import os
import argparse
import asyncio
import pandas as pd
import ast
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
from tqdm.asyncio import tqdm  # pip install tqdm

# --- КОНФИГ ---
API_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"

DATA_DIR = os.getenv("DATA_DIR", "data")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions_clean.csv")
IDS_CSV = "final/submission_ids.csv"
OUTPUT_CSV = "final/final_su.csv"

# ВАЖНО: Имя коллекции как в debug скрипте
COLLECTION_NAME = "documents1" 

# Количество одновременных потоков (для T4 и vLLM 30-50 оптимально)
CONCURRENT_REQUESTS = 100 

# Тестовый режим (True = обработать только TEST_LIMIT вопросов)
TEST_MODE = False
TEST_LIMIT = 50 

# Клиенты
# Qdrant используем асинхронный для неблокирующего I/O
client_qdrant = AsyncQdrantClient(url="http://localhost:6333")
aclient = AsyncOpenAI(base_url=API_URL, api_key=API_KEY)

async def get_text_from_qdrant(web_id) -> str:
    """
    Извлекает текст чанков асинхронно.
    CRITICAL FIX: Принудительно конвертируем ID в строку (str),
    так как в базе они лежат как строки.
    """
    try:
        points, _ = await client_qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", 
                        # ВАЖНО: str(web_id) исправляет проблему "Int vs String"
                        match=models.MatchValue(value=str(web_id))
                    )
                ]
            ),
            limit=10, # Берем до 10 чанков одного документа (если он большой)
            with_payload=True, 
            with_vectors=False
        )
        
        if points:
            # Склеиваем тексты чанков
            texts = [p.payload.get('text', '') for p in points]
            return "\n".join(texts)
        return ""
    except Exception as e:
        # Логируем редко, чтобы не спамить
        return ""

async def process_row(row, doc_cache, semaphore):
    async with semaphore:
        q_id = row['q_id']
        query = row['query']
        ids_str = str(row.get('retrieved_ids', '[]'))
        
        # 1. Парсинг ID документов
        try:
            doc_ids = ast.literal_eval(ids_str)
            if not isinstance(doc_ids, list): doc_ids = []
        except:
            doc_ids = []
        
        # 2. Сборка контекста
        context_parts = []
        for d_id in doc_ids:
            # Ключ для кэша - строка, чтобы не путаться
            cache_key = str(d_id)

            if cache_key not in doc_cache:
                # Делаем запрос в Qdrant
                found_text = await get_text_from_qdrant(cache_key)
                doc_cache[cache_key] = found_text
            
            if doc_cache[cache_key]:
                context_parts.append(doc_cache[cache_key])
            
        # Ограничиваем контекст (8k токенов модели ~ 20-25k символов, но берем 6k для скорости)
        full_context = "\n\n".join(context_parts)[:2500]
        
        # 3. Генерация
        if not full_context.strip():
            # Если контекст пустой, модель не сможет ответить
            return {"q_id": q_id, "answer": "Информации недостаточно"}

        system_prompt = (
            "Ты — ассистент Альфа-Банка. "
            "Отвечай на вопрос клиента, используя ИСКЛЮЧИТЕЛЬНО предоставленный контекст. "
						"оцени контекст на релевантность от 0 до 10, где 10 - идеальный контекст. "
            "Если релевантность контекста меньше 4 — пиши 'Информации недостаточно' без дополнительных пояснений. "
            "Ответ должен быть кратким (до 3 предложений). "
            "ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ."
        )
        
        try:
            response = await aclient.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Контекст:\n{full_context}\n\nВопрос: {query}"}
                ],
                temperature=0.0,
                max_tokens=120,
            )
            ans = response.choices[0].message.content.strip()
            return {"q_id": q_id, "answer": ans}
        except Exception as e:
            print(f"LLM Error q_id={q_id}: {e}")
            return {"q_id": q_id, "answer": "Ошибка генерации"}

async def main(test_mode=False):
    print("--- ЗАПУСК ГЕНЕРАЦИИ (FINAL FIX) ---")
    
    # 1. Загрузка данных
    if not os.path.exists(QUESTIONS_CSV) or not os.path.exists(IDS_CSV):
        print("❌ Ошибка: Не найдены входные файлы (questions.csv или submission_ids.csv)")
        return

    q_df = pd.read_csv(QUESTIONS_CSV)
    ids_df = pd.read_csv(IDS_CSV)
    
    # Мердж
    df = pd.merge(q_df, ids_df, on='q_id', how='left')
    if 'answer' in df.columns and 'retrieved_ids' not in df.columns:
        df.rename(columns={'answer': 'retrieved_ids'}, inplace=True)
    
    if test_mode:
        print(f"⚠️ ТЕСТОВЫЙ РЕЖИМ: Берем только первые {TEST_LIMIT} вопросов")
        df = df.head(TEST_LIMIT)
    
    print(f"Вопросов к обработке: {len(df)}")
    
    # 2. Подготовка
    doc_cache = {} # Кэш текстов документов, чтобы не дергать базу лишний раз
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = []
    
    # 3. Создание задач
    for _, row in df.iterrows():
        tasks.append(process_row(row, doc_cache, semaphore))
    
    # 4. Исполнение
    results = await tqdm.gather(*tasks, desc="Генерация ответов")
    
    # 5. Сохранение
    final_df = pd.DataFrame(results).sort_values(by='q_id')
    
    # Проверка на пустые ответы перед сохранением
    empty_count = len(final_df[final_df['answer'] == "Информации недостаточно"])
    print(f"📊 Статистика: Всего {len(final_df)}, 'Информации недостаточно': {empty_count}")
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Файл сохранен: {OUTPUT_CSV}")

if __name__ == "__main__":
    # Парсинг аргументов
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "--тест", action="store_true", help="Запустить в тестовом режиме (50 вопросов)")
    args = parser.parse_args()

    # Исправление для event loop в некоторых средах
    try:
        asyncio.run(main(test_mode=args.test))
    except KeyboardInterrupt:
        print("Остановлено пользователем")