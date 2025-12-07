"""
Direct High-Performance Generator for 2x RTX 5090
Uses vLLM offline inference engine directly (no server required).
Model: Qwen2.5-72B-Instruct-AWQ (Max quality/speed balance for 2x32GB)
"""

import os
import ast
import asyncio
import pandas as pd
from qdrant_client import AsyncQdrantClient, models
from vllm import LLM, SamplingParams

# --- КОНФИГУРАЦИЯ ---
# Используем 72B модель, так как у нас 64GB VRAM. Она влезает в Int4 (AWQ).
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"
TENSOR_PARALLEL_SIZE = 2  # Распараллеливание на 2 видеокарты
MAX_MODEL_LEN = 16384     # У Qwen огромный контекст, можем позволить больше
GPU_MEM_UTIL = 0.95       # Забиваем память по максимуму

# Настройки генерации
SAMPLING_PARAMS = SamplingParams(
    temperature=0.1,
    max_tokens=300,        # Ответы могут быть подробнее
    min_p=0.05,
    repetition_penalty=1.1,
    stop=["<|endoftext|>", "<|im_end|>"]
)

# Пути (поправьте под свою структуру)
DATA_DIR = os.getenv("DATA_DIR", "data")
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions_clean.csv")
IDS_CSV = "final/submission_ids.csv"
OUTPUT_CSV = "final/final_su.csv"
COLLECTION_NAME = "documents1"
QDRANT_URL = "http://localhost:6333" # Qdrant должен быть запущен (можно в докере или локально)

SYSTEM_PROMPT = """Ты — эксперт поддержки Альфа-Банка.
Твоя задача: ответить на вопрос клиента, используя ТОЛЬКО предоставленный контекст.
Если информации в контексте недостаточно для полного ответа, напиши строго: "Информации недостаточно".

Требования к ответу:
1. Используй цитаты и факты только из контекста.
2. Стиль: вежливый, профессиональный, лаконичный (2-4 предложения).
3. Структурируй ответ, если фактов несколько.
4. Язык: Русский.
"""

# --- ЧАСТЬ 1: Работа с данными и Qdrant (Async) ---

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
    print(f"📥 Скачивание {len(all_doc_ids)} документов из Qdrant...")
    client = AsyncQdrantClient(url=QDRANT_URL)
    
    tasks = []
    doc_cache = {}
    
    # Разбиваем на чанки, чтобы не убить сеть/Qdrant
    batch_size = 200
    ids_list = list(all_doc_ids)
    
    for i in range(0, len(ids_list), batch_size):
        batch = ids_list[i:i+batch_size]
        batch_tasks = [fetch_single_doc(client, d_id) for d_id in batch]
        results = await asyncio.gather(*batch_tasks)
        for d_id, text in results:
            doc_cache[d_id] = text
        print(f"  Загружено {min(i+batch_size, len(ids_list))}/{len(ids_list)}")
        
    await client.close()
    print(f"✅ Документы загружены. Всего: {len(doc_cache)}")
    return doc_cache

def prepare_prompts(df, doc_cache):
    print("📝 Подготовка промптов...")
    prompts = []
    indices = []
    
    for idx, row in df.iterrows():
        q_id = row['q_id']
        query = row['query']
        
        # Получаем ID документов
        try:
            doc_ids = ast.literal_eval(str(row.get('retrieved_ids', '[]')))
            if not isinstance(doc_ids, list): doc_ids = []
        except:
            doc_ids = []
            
        # Собираем контекст
        context_texts = [doc_cache.get(str(d_id), "") for d_id in doc_ids if doc_cache.get(str(d_id))]
        full_context = "\n\n".join(context_texts)[:20000] # Большой контекст для 72B модели
        
        if not full_context.strip():
            # Если нет контекста, помечаем для быстрой заглушки (но vllm прогоним для унификации или пропустим)
            full_context = "Нет информации."
        
        # Формат ChatML (Qwen его любит)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст:\n{full_context}\n\nВопрос: {query}"}
        ]
        
        # В vLLM можно подавать список сообщений, но лучше отформатированный текст
        # Qwen2.5 использует ChatML. vLLM сам применит chat_template если передать tokenizer
        # Но для простоты передадим messages в generate (vLLM >= 0.6.0 поддерживает entry с messages)
        # Если старый vLLM, нужно форматировать вручную. Будем надеяться на свежий vLLM.
        # Для надежности используем apply_chat_template через токенизатор ПОЗЖЕ, 
        # но пока соберем просто данные.
        
        prompts.append(messages)
        indices.append(idx)
        
    return prompts, indices

# --- ЧАСТЬ 2: Инференс (Sync) ---

def run_inference(prompts):
    print(f"🚀 Загрузка модели {MODEL_NAME} на {TENSOR_PARALLEL_SIZE} GPU...")
    
    # Инициализация движка
    llm = LLM(
        model=MODEL_NAME,
        quantization="awq",
        dtype="float16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False, # True для torch graph (быстрее), False если проблемы
        trust_remote_code=True
    )
    
    # Генерация
    # vLLM принимает prompt_token_ids или prompts. 
    # Но лучше всего он работает с tokenizer.apply_chat_template.
    # Сделаем это через сам LLM.
    
    tokenizer = llm.get_tokenizer()
    text_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) 
        for p in prompts
    ]
    
    print(f"🔥 Запуск генерации для {len(text_prompts)} запросов...")
    outputs = llm.generate(text_prompts, SAMPLING_PARAMS)
    
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        results.append(generated_text)
        
    return results

# --- MAIN ---

def main():
    # 1. Загрузка данных
    if not os.path.exists(QUESTIONS_CSV) or not os.path.exists(IDS_CSV):
        print("❌ Файлы данных не найдены")
        return

    q_df = pd.read_csv(QUESTIONS_CSV)
    ids_df = pd.read_csv(IDS_CSV)
    df = pd.merge(q_df, ids_df, on='q_id', how='left')
    
    # ТЕСТ (раскомментировать для теста)
    # df = df.head(50) 
    
    # 2. Сбор всех ID документов
    all_doc_ids = set()
    for _, row in df.iterrows():
        try:
            d_ids = ast.literal_eval(str(row.get('retrieved_ids', '[]')))
            if isinstance(d_ids, list):
                all_doc_ids.update(str(x) for x in d_ids)
        except: pass
        
    # 3. Скачивание документов (нужен запущенный Qdrant!)
    # Qdrant можно оставить в докере: docker compose up -d qdrant
    doc_cache = asyncio.run(prefetch_documents(all_doc_ids))
    
    # 4. Подготовка промптов
    prompts, indices = prepare_prompts(df, doc_cache)
    
    # 5. Инференс
    answers = run_inference(prompts)
    
    # 6. Сборка результатов
    results_data = []
    for idx, ans in zip(indices, answers):
        q_id = df.loc[idx, 'q_id']
        
        # Фильтр
        if "Информации недостаточно" in ans or any("\u4e00" <= c <= "\u9fff" for c in ans):
            final_ans = "Информации недостаточно"
        else:
            final_ans = ans
            
        results_data.append({"q_id": q_id, "answer": final_ans})
        
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Готово! Сохранено в {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
