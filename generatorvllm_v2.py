"""
Optimized RAG Generator v3
Target: 200+ tokens/sec throughput
Key optimizations:
1. Batch prefetch all documents BEFORE generation
2. Few-shot prompt for better style matching
3. Parallel document fetching with asyncio.gather
4. Retry decorator for LLM errors
5. Model: Mistral 7B v0.3 AWQ
6. repetition_penalty: 1.1 for quality
"""

import os
import argparse
import asyncio
import re
import pandas as pd
import ast
from functools import wraps
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
from tqdm.asyncio import tqdm

# --- КОНФИГ ---
API_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 0.5

DATA_DIR = os.getenv("DATA_DIR", "data")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions_clean.csv")
IDS_CSV = "final/submission_ids.csv"
OUTPUT_CSV = "final/final_su.csv"

COLLECTION_NAME = "documents1"

# ОПТИМИЗАЦИЯ: Увеличиваем concurrency, т.к. prefetch убирает bottleneck
CONCURRENT_REQUESTS = 256
QDRANT_BATCH_SIZE = 100  # Batch fetching from Qdrant

# Тестовый режим
TEST_MODE = False
TEST_LIMIT = 50

# Клиенты
client_qdrant = AsyncQdrantClient(url="http://localhost:6333")
aclient = AsyncOpenAI(base_url=API_URL, api_key=API_KEY)

# Улучшенный промпт с few-shot примерами для лучшего стиля
SYSTEM_PROMPT = """Ты — ассистент поддержки Альфа-Банка. Отвечай на вопрос клиента строго по контексту.

Правила:
- Ответ: 2-3 предложения, кратко и по делу
- Нет информации в контексте → отвечай только: "Информации недостаточно"
- Язык: только русский

Примеры:
Вопрос: Как узнать номер счета?
Ответ: Номер счета указан в разделе договора с реквизитами. Также его можно найти в мобильном приложении в разделе "Счета".

Вопрос: Почему не приходят СМС?
Ответ: Возможно, вы включили подтверждение через секретный код в приложении. Проверьте настройки уведомлений в профиле."""


def async_retry(max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Decorator for async retry on failure"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_error
        return wrapper
    return decorator


async def fetch_single_doc(doc_id: str) -> tuple[str, str]:
    """Fetch single document, return (doc_id, text)"""
    try:
        points, _ = await client_qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=20,  # Увеличено для полноты контекста
            with_payload=True,
            with_vectors=False
        )
        if points:
            texts = [p.payload.get('text', '') for p in points]
            return (doc_id, "\n".join(texts))
        return (doc_id, "")
    except Exception:
        return (doc_id, "")


async def prefetch_all_documents(all_doc_ids: set[str]) -> dict[str, str]:
    """
    ОПТИМИЗАЦИЯ 2: Prefetch ALL documents in parallel BEFORE generation.
    This eliminates per-request Qdrant latency.
    """
    print(f"📥 Prefetching {len(all_doc_ids)} unique documents...")
    
    # Batch into chunks to avoid overwhelming Qdrant
    doc_list = list(all_doc_ids)
    cache = {}
    
    for i in range(0, len(doc_list), QDRANT_BATCH_SIZE):
        batch = doc_list[i:i + QDRANT_BATCH_SIZE]
        tasks = [fetch_single_doc(doc_id) for doc_id in batch]
        results = await asyncio.gather(*tasks)
        for doc_id, text in results:
            cache[doc_id] = text
        
        # Progress indicator
        if (i + QDRANT_BATCH_SIZE) % 500 == 0:
            print(f"  Fetched {min(i + QDRANT_BATCH_SIZE, len(doc_list))}/{len(doc_list)}")
    
    print(f"✅ Prefetch complete. Cache size: {len(cache)}")
    return cache


async def process_row(row, doc_cache: dict, semaphore: asyncio.Semaphore):
    """Process single question - now cache is pre-populated"""
    async with semaphore:
        q_id = row['q_id']
        query = row['query']
        ids_str = str(row.get('retrieved_ids', '[]'))
        
        try:
            doc_ids = ast.literal_eval(ids_str)
            if not isinstance(doc_ids, list):
                doc_ids = []
        except:
            doc_ids = []
        
        # ОПТИМИЗАЦИЯ 3: Просто читаем из кэша, никаких await
        context_parts = []
        for d_id in doc_ids:
            cache_key = str(d_id)
            text = doc_cache.get(cache_key, "")
            if text:
                context_parts.append(text)
        
        # ОПТИМИЗАЦИЯ 4: Меньше контекста = быстрее prefill
        # 4000 символов ~ 1200-1500 токенов — достаточно для ответа
        full_context = "\n\n".join(context_parts)[:5000]
        
        if not full_context.strip():
            return {"q_id": q_id, "answer": "Информации недостаточно"}
        
        try:
            ans = await generate_answer(query, full_context)
            return {"q_id": q_id, "answer": ans}
        except Exception as e:
            print(f"LLM Error q_id={q_id}: {e}")
            return {"q_id": q_id, "answer": "Ошибка генерации"}


@async_retry(max_retries=MAX_RETRIES)
async def generate_answer(query: str, context: str) -> str:
    """Generate answer with retry logic"""
    response = await aclient.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос клиента: {query}"}
        ],
        temperature=0.1,
        max_tokens=120,
        extra_body={"min_p": 0.05, "repetition_penalty": 1.1}
    )
    ans = response.choices[0].message.content.strip()
    
    # Фильтр китайского
    if any("\u4e00" <= char <= "\u9fff" for char in ans):
        return "Информации недостаточно"
    
    return ans


async def main(test_mode=False):
    print("=== OPTIMIZED RAG GENERATOR v3 ===")
    
    # 1. Загрузка данных
    if not os.path.exists(QUESTIONS_CSV) or not os.path.exists(IDS_CSV):
        print("❌ Ошибка: Не найдены входные файлы")
        return

    q_df = pd.read_csv(QUESTIONS_CSV)
    ids_df = pd.read_csv(IDS_CSV)
    
    df = pd.merge(q_df, ids_df, on='q_id', how='left')
    if 'answer' in df.columns and 'retrieved_ids' not in df.columns:
        df.rename(columns={'answer': 'retrieved_ids'}, inplace=True)
    
    if test_mode:
        print(f"⚠️ ТЕСТОВЫЙ РЕЖИМ: {TEST_LIMIT} вопросов")
        df = df.head(TEST_LIMIT)
    
    print(f"Вопросов к обработке: {len(df)}")
    
    # 2. ОПТИМИЗАЦИЯ: Собираем ВСЕ уникальные doc_ids и prefetch'им их
    all_doc_ids = set()
    for _, row in df.iterrows():
        ids_str = str(row.get('retrieved_ids', '[]'))
        try:
            doc_ids = ast.literal_eval(ids_str)
            if isinstance(doc_ids, list):
                all_doc_ids.update(str(d) for d in doc_ids)
        except:
            pass
    
    # Prefetch all documents
    doc_cache = await prefetch_all_documents(all_doc_ids)
    
    # 3. Генерация (теперь без Qdrant latency)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [process_row(row, doc_cache, semaphore) for _, row in df.iterrows()]
    
    print(f"🚀 Запуск генерации с concurrency={CONCURRENT_REQUESTS}")
    results = await tqdm.gather(*tasks, desc="Генерация")
    
    # 4. Сохранение
    final_df = pd.DataFrame(results).sort_values(by='q_id')
    empty_count = len(final_df[final_df['answer'].str.contains("недостаточно", na=False)])
    print(f"📊 Статистика: Всего {len(final_df)}, 'Информации недостаточно': {empty_count}")
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Файл сохранен: {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "--тест", action="store_true")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(test_mode=args.test))
    except KeyboardInterrupt:
        print("Остановлено")
