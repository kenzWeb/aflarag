import os
import asyncio
import pandas as pd
import ast
from openai import AsyncOpenAI  # Нужен `pip install openai` версии 1.x+
from typing import List

# Настройки
API_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-AWQ"

# Файлы
DATA_DIR = os.getenv("DATA_DIR", "data")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions_clean.csv")
IDS_CSV = "final/submission_ids.csv"
OUTPUT_CSV = "final/final_su.csv"

# Настройки параллелизма
CONCURRENT_REQUESTS = 50  # Сколько запросов слать одновременно (для T4 норм 30-50)

# Инициализация Qdrant клиента (синхронный ок, он быстрый)
from qdrant_client import QdrantClient, models
client_qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "documents1"

# Асинхронный клиент LLM
aclient = AsyncOpenAI(base_url=API_URL, api_key=API_KEY)

def get_text_from_qdrant(web_id: int) -> str:
    """Синхронная вставка из Qdrant (быстрая)"""
    try:
        points, _ = client_qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=web_id))]),
            limit=5, with_payload=True, with_vectors=False
        )
        return "\n".join([p.payload.get('text', '') for p in points]) if points else ""
    except: return ""

async def process_row(row, doc_cache, semaphore):
    async with semaphore:  # Ограничиваем количество одновременных запросов
        q_id = row['q_id']
        query = row['query']
        ids_str = str(row.get('retrieved_ids', '[]'))
        
        # Сборка контекста
        try: doc_ids = ast.literal_eval(ids_str)
        except: doc_ids = []
        
        context_parts = []
        for d_id in doc_ids:
            try:
                did = int(d_id)
                if did not in doc_cache:
                    doc_cache[did] = get_text_from_qdrant(did)
                if doc_cache[did]:
                    context_parts.append(doc_cache[did])
            except: continue
            
        full_context = "\n\n".join(context_parts)[:6000]
        
        if not full_context.strip():
            return {"q_id": q_id, "answer": "Информации недостаточно"}

        # Промпт
        system_prompt = "Ты — ассистент Альфа-Банка. Отвечай кратко (до 3 предложений) и ТОЛЬКО по контексту. Если информации нет — пиши 'Информации недостаточно'."
        
        try:
            response = await aclient.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Контекст:\n{full_context}\n\nВопрос: {query}"}
                ],
                temperature=0.0,
                max_tokens=250,
            )
            return {"q_id": q_id, "answer": response.choices[0].message.content.strip()}
        except Exception as e:
            print(f"Error {q_id}: {e}")
            return {"q_id": q_id, "answer": "Ошибка"}

async def main():
    print("Loading data...")
    q_df = pd.read_csv(QUESTIONS_CSV)
    ids_df = pd.read_csv(IDS_CSV)
    df = pd.merge(q_df, ids_df, on='q_id', how='left')
    if 'answer' in df.columns and 'retrieved_ids' not in df.columns:
        df.rename(columns={'answer': 'retrieved_ids'}, inplace=True)

    # Кэш документов (общий для всех)
    doc_cache = {}
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    tasks = []
    print(f"Starting async processing of {len(df)} questions...")
    
    # Создаем задачи
    for _, row in df.iterrows():
        tasks.append(process_row(row, doc_cache, semaphore))
    
    # Запускаем все сразу с прогресс-баром
    # (pip install tqdm если нет)
    from tqdm.asyncio import tqdm
    results = await tqdm.gather(*tasks)
    
    # Сохраняем
    final_df = pd.DataFrame(results).sort_values(by='q_id')
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())