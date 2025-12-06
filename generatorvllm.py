import os
import time
import pandas as pd
from openai import OpenAI
from typing import List, Dict
import ast

# Настройки
API_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY" # vLLM не требует ключа локально
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-AWQ" # То же имя, что и при запуске докера

# Данные
DATA_DIR = os.getenv("DATA_DIR", "data")
WEBSITES_CSV = os.path.join(DATA_DIR, "websites.csv")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions_clean.csv")
IDS_CSV = "final/submission_ids.csv" # Файл с ID из предыдущего шага
OUTPUT_CSV = "final/final_submission.csv"

# Инициализация клиента
client = OpenAI(base_url=API_URL, api_key=API_KEY)

def load_docs():
    print("Loading docs...")
    df = pd.read_csv(WEBSITES_CSV).dropna(subset=['web_id', 'text'])
    store = {}
    for _, row in df.iterrows():
        try:
            store[int(row['web_id'])] = str(row['text']).strip()
        except: pass
    return store

def generate_answer(query: str, context: str):
    system_prompt = """Ты — ассистент Альфа-Банка.
ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе контекста. Если информации нет — пиши "Информации недостаточно".
2. Ответ должен быть кратким (до 3 предложений).
3. Не используй фразы "В тексте сказано". Сразу к сути."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
            ],
            temperature=0.0, # Детерминизм
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "Ошибка генерации"

def main():
    doc_store = load_docs()
    questions = pd.read_csv(QUESTIONS_CSV)
    ids_df = pd.read_csv(IDS_CSV)
    
    # Мержим вопросы и ID
    df = pd.merge(questions, ids_df, on='q_id', how='left')
    # Если колонка называется answer в ids файле, переименуем
    if 'answer' in df.columns and 'retrieved_ids' not in df.columns:
        df.rename(columns={'answer': 'retrieved_ids'}, inplace=True)

    results = []
    
    print(f"Start processing {len(df)} questions via vLLM...")
    
    for idx, row in df.iterrows():
        q_id = row['q_id']
        query = row['query']
        ids_str = str(row.get('retrieved_ids', '[]'))
        
        # Парсим ID
        try:
            doc_ids = ast.literal_eval(ids_str)
            if not isinstance(doc_ids, list): doc_ids = []
        except: doc_ids = []
        
        # Собираем контекст
        context_parts = [doc_store[d] for d in doc_ids if d in doc_store]
        full_context = "\n\n".join(context_parts)[:6000] # Ограничим контекст 6к знаков
        
        # Генерация
        if not full_context:
            ans = "Информации недостаточно"
        else:
            ans = generate_answer(query, full_context)
            
        results.append({"q_id": q_id, "answer": ans})
        
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(df)}")

    # Сохраняем
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    main()