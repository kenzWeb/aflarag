"""
Retrieval ID Submission Generator
Purpose: Generate submission with web_ids ONLY.
Output Format: q_id, [id1, id2, id3, id4, id5]
"""

import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List

import pandas as pd
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Try importing Reranker
try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    # Embedding
    EMBEDDING_MODEL = "ai-forever/FRIDA"
    EMBEDDING_DIM = 1536
    
    # Reranker
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANK_TOP_K_INPUT = 120   # Берем 100 кандидатов
    RERANK_TOP_K_OUTPUT = 5    # Оставляем топ-5 ID
    
    # Vector DB
    COLLECTION_NAME = "documents1" 
    
    # Output
    OUTPUT_FILE = "final/submission_ids.csv"

# ==========================================
# MODELS
# ==========================================

_embed_model = None
_embed_lock = threading.Lock()

def get_frida_model():
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            logger.info(f"Loading Embedding Model: {Config.EMBEDDING_MODEL}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _embed_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)
    return _embed_model

def embed_text_frida(text: str, is_query: bool = True) -> List[float]:
    model = get_frida_model()
    prompt_name = "search_query" if is_query else "search_document"
    if not text or not text.strip():
        return [0.0] * Config.EMBEDDING_DIM
    with torch.no_grad():
        embedding = model.encode(text.strip(), prompt_name=prompt_name, convert_to_numpy=True)
    return embedding.tolist()

class Reranker:
    def __init__(self, model_name=Config.RERANKER_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = None
        if FlagReranker:
            self.reranker = FlagReranker(model_name, use_fp16=True, device=self.device)
            logger.info(f"Reranker loaded on {self.device}")
        else:
            logger.warning("⚠️ FlagReranker not installed!")

    def compute_scores(self, query: str, texts: List[str]) -> List[float]:
        """Compute raw scores for query-text pairs"""
        if not self.reranker or not texts:
            return [0.0] * len(texts)
            
        pairs = [[query, t] for t in texts]
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            scores = self.reranker.compute_score(batch)
            if isinstance(scores, float): scores = [scores]
            all_scores.extend(scores)
            
        return all_scores

# ==========================================
# RETRIEVAL LOGIC
# ==========================================

_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client

def retrieve_candidates(query: str, top_k: int) -> List[Dict]:
    query_vec = embed_text_frida(query, is_query=True)
    client = get_qdrant_client()
    
    try:
        response = client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True
        )
        hits = response.points
    except Exception as e:
        logger.error(f"Qdrant error: {e}")
        return []
    
    results = []
    for hit in hits:
        payload = hit.payload
        if payload:
            results.append({
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id"), # Это и есть web_id
                "score": hit.score # Векторный скор
            })
    return results

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # Paths
    data_dir = os.getenv("DATA_DIR", "data")
    questions_path = os.path.join(data_dir, "questions_clean.csv")
    
    if not os.path.exists(questions_path):
        logger.error(f"Questions file not found at {questions_path}")
        return
        
    questions_df = pd.read_csv(questions_path)
    logger.info(f"Loaded {len(questions_df)} questions.")
    
    # Init Models
    get_frida_model()
    reranker = Reranker()
    
    results = []
    start_time = time.time()
    
    # Loop
    for idx, row in questions_df.iterrows():
        q_id = row['q_id']
        query = str(row['query']).strip()
        
        # 1. Retrieval (Top-100)
        # Получаем список словарей {'text':..., 'doc_id':...}
        candidates = retrieve_candidates(query, top_k=Config.RERANK_TOP_K_INPUT)
        
        if not candidates:
            results.append({"q_id": q_id, "answer": "[]"})
            continue

        # 2. Reranking Logic
        # Извлекаем тексты для реранкера
        cand_texts = [c['text'] for c in candidates]
        
        # Считаем новые скоры
        rerank_scores = reranker.compute_scores(query, cand_texts)
        
        # Обновляем скоры в словарях кандидатов
        for i, c in enumerate(candidates):
            c['rerank_score'] = rerank_scores[i]
            
        # Сортируем по новому скору реранкера
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Берем топ-5 лучших
        top_candidates = candidates[:Config.RERANK_TOP_K_OUTPUT]
        
        # 3. Extract web_ids
        # Преобразуем в int для чистоты (если они были строками) и формируем список
        web_ids = []
        for c in top_candidates:
            try:
                # doc_id может быть строкой "123.0" или "123", приводим к int
                w_id = int(float(str(c['doc_id'])))
                web_ids.append(w_id)
            except (ValueError, TypeError):
                # Если вдруг doc_id не число, оставляем как есть или пропускаем
                if c['doc_id']:
                    web_ids.append(c['doc_id'])

        # Формируем строку вида "[253, 704, 24]"
        # Важно: просто str(list) создаст нужный формат
        answer_str = str(web_ids)

        results.append({
            "q_id": q_id,
            "answer": answer_str
        })
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(questions_df)} questions...")

    # 4. Save
    output_path = Config.OUTPUT_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ IDs submission saved to {output_path}")
    logger.info(f"⏱️ Time taken: {elapsed:.1f}s")

if __name__ == "__main__":
    main()