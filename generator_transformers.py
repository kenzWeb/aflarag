"""
Direct High-Performance Generator using Transformers + BitsAndBytes
Optimized for 2x RTX 5090 (Batch Processing)
Model: Qwen/Qwen2.5-32B-Instruct (loaded in 4-bit NF4)
"""

import os
import ast
import asyncio
import torch
import pandas as pd
from qdrant_client import AsyncQdrantClient, models
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct" 

# Настройки генерации
BATCH_SIZE = 8           # Размер пачки (8-16 для 32B на 2x5090)
MAX_NEW_TOKENS = 200     # Длина ответа
TEMPERATURE = 0.1

# Пути
DATA_DIR = os.getenv("DATA_DIR", "data")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions_clean.csv")
IDS_CSV = "final/submission_ids.csv"
OUTPUT_CSV = "final/final_su.csv"
COLLECTION_NAME = "documents1"
QDRANT_URL = "http://158.160.208.30:6333"

SYSTEM_PROMPT = """Ты — эксперт поддержки Альфа-Банка.
Отвечай на вопрос клиента, используя ТОЛЬКО предоставленный контекст.
Если информации недостаточно, напиши: "Информации недостаточно".

Требования:
1. Используй факты только из контекста.
2. Ответ: 2-4 предложения, кратко.
3. Язык: Русский.
"""

# --- Qdrant ---

async def fetch_single_doc(client, doc_id: str):
    try:
        points, _ = await client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=str(doc_id)))]
            ),
            limit=20,
            with_payload=True,
            with_vectors=False
        )
        if points:
            return str(doc_id), "\n".join([p.payload.get('text', '') for p in points])
    except:
        pass
    return str(doc_id), ""

async def prefetch_documents(all_doc_ids):
    print(f"📥 Загрузка {len(all_doc_ids)} документов...")
    client = AsyncQdrantClient(url=QDRANT_URL)
    doc_cache = {}
    batch_size = 200
    ids_list = list(all_doc_ids)
    
    for i in range(0, len(ids_list), batch_size):
        batch = ids_list[i:i+batch_size]
        results = await asyncio.gather(*[fetch_single_doc(client, d_id) for d_id in batch])
        for d_id, text in results:
            doc_cache[d_id] = text
        
    await client.close()
    print(f"✅ Загружено {len(doc_cache)} документов")
    return doc_cache

def prepare_data(df, doc_cache):
    prompts = []
    q_ids = []
    
    for _, row in df.iterrows():
        q_id = row['q_id']
        query = row['query']
        
        try:
            doc_ids = ast.literal_eval(str(row.get('retrieved_ids', '[]')))
            if not isinstance(doc_ids, list): doc_ids = []
        except:
            doc_ids = []
            
        context_texts = [doc_cache.get(str(d_id), "") for d_id in doc_ids if doc_cache.get(str(d_id))]
        full_context = "\n\n".join(context_texts)[:10000]
        
        if not full_context.strip():
            full_context = "Нет информации."
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{full_context}\n\nВопрос: {query}"}
        ]
        prompts.append(messages)
        q_ids.append(q_id)
        
    return prompts, q_ids

# --- Инференс ---

def run_inference(prompts_messages):
    print(f"🚀 Загрузка {MODEL_NAME}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    
    # Токенизация
    text_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts_messages
    ]
    
    all_answers = []
    total = len(text_prompts)
    
    print(f"🔥 Генерация ({total} запросов, batch={BATCH_SIZE})...")
    
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Генерация"):
        batch_prompts = text_prompts[i : i + BATCH_SIZE]
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=8192
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        batch_answers = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_answers.extend(batch_answers)
        
    return all_answers

# --- MAIN ---

def main():
    if not os.path.exists(QUESTIONS_CSV) or not os.path.exists(IDS_CSV):
        print("❌ Файлы не найдены")
        return

    q_df = pd.read_csv(QUESTIONS_CSV)
    ids_df = pd.read_csv(IDS_CSV)
    df = pd.merge(q_df, ids_df, on='q_id', how='left')
    
    # TEST: раскомментируйте для теста
    # df = df.head(20)
    
    all_doc_ids = set()
    for _, row in df.iterrows():
        try:
            d_ids = ast.literal_eval(str(row.get('retrieved_ids', '[]')))
            if isinstance(d_ids, list):
                all_doc_ids.update(str(x) for x in d_ids)
        except: pass
        
    doc_cache = asyncio.run(prefetch_documents(all_doc_ids))
    prompts, q_ids = prepare_data(df, doc_cache)
    
    answers = run_inference(prompts)
    
    results = []
    for q_id, ans in zip(q_ids, answers):
        ans_clean = ans.strip()
        if "Информации недостаточно" in ans_clean or any("\u4e00" <= c <= "\u9fff" for c in ans_clean):
            ans_clean = "Информации недостаточно"
        results.append({"q_id": q_id, "answer": ans_clean})
        
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Готово! {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
