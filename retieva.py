"""
Retrieval-Only Submission Generator
Purpose: Debug retrieval quality without waiting for LLM generation.
Output: Submission CSV where 'answer' contains the retrieved context chunks.
Pipeline: Frida Retrieval (Top-100) -> BGE-M3 Reranking (Top-5) -> Context Concatenation
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

# Try importing Reranker (Critical for this pipeline)
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
    RERANK_TOP_K_INPUT = 100   # Широкий поиск кандидатов
    RERANK_TOP_K_OUTPUT = 5    # Топ-5 самых релевантных для контекста
    
    # Vector DB
    # Убедитесь, что имя коллекции совпадает с тем, куда вы загружали данные!
    COLLECTION_NAME = "documents1" 
    
    # Output
    OUTPUT_FILE = "final/retrieval_debug.csv"

# ==========================================
# MODELS
# ==========================================

_embed_model = None
_embed_lock = threading.Lock()

def get_frida_model():
    """Singleton for Embedding Model"""
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            logger.info(f"Loading Embedding Model: {Config.EMBEDDING_MODEL}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _embed_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)
    return _embed_model

def embed_text_frida(text: str, is_query: bool = True) -> List[float]:
    """Generate embedding with correct prompt prefix."""
    model = get_frida_model()
    prompt_name = "search_query" if is_query else "search_document"
    
    if not text or not text.strip():
        return [0.0] * Config.EMBEDDING_DIM

    with torch.no_grad():
        embedding = model.encode(
            text.strip(), 
            prompt_name=prompt_name,
            convert_to_numpy=True
        )
    return embedding.tolist()

class Reranker:
    def __init__(self, model_name=Config.RERANKER_MODEL):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = None
        if FlagReranker:
            self.reranker = FlagReranker(model_name, use_fp16=True, device=self.device)
            logger.info(f"Reranker loaded on {self.device}")
        else:
            logger.warning("⚠️ FlagReranker not installed! Results will be suboptimal.")

    def rerank(self, query: str, candidates: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.reranker or not candidates:
            # Fallback if no reranker
            return [{"text": c, "score": 0.0} for c in candidates[:top_k]]
            
        pairs = [[query, c] for c in candidates]
        
        # Batch processing to avoid OOM on large lists
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            scores = self.reranker.compute_score(batch)
            if isinstance(scores, float): scores = [scores]
            all_scores.extend(scores)
            
        scored = [{"text": c, "score": s} for c, s in zip(candidates, all_scores)]
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        return scored[:top_k]

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
        hits = client.search(
            collection_name=Config.COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True
        )
    except Exception as e:
        logger.error(f"Qdrant error: {e}")
        return []
    
    results = []
    for hit in hits:
        payload = hit.payload
        if payload:
            results.append({
                "text": payload.get("text", ""),
                "score": hit.score,
                "doc_id": payload.get("doc_id")
            })
    return results

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # Paths
    data_dir = os.getenv("DATA_DIR", "data_final")
    questions_path = os.path.join(data_dir, "questions_clean.csv")
    
    # 1. Load Data
    if not os.path.exists(questions_path):
        logger.error(f"Questions file not found at {questions_path}")
        return
        
    questions_df = pd.read_csv(questions_path)
    logger.info(f"Loaded {len(questions_df)} questions for retrieval check.")
    
    # 2. Init Models
    get_frida_model()
    reranker = Reranker()
    
    results = []
    start_time = time.time()
    
    # 3. Processing Loop
    # No batching needed for logic logic, simple iteration is fine for retrieval dump
    for idx, row in questions_df.iterrows():
        q_id = row['q_id']
        query = str(row['query']).strip()
        
        # A. Retrieval (Top-100)
        candidates = retrieve_candidates(query, top_k=Config.RERANK_TOP_K_INPUT)
        cand_texts = [c['text'] for c in candidates]
        
        # B. Reranking (Top-5)
        reranked = reranker.rerank(query, cand_texts, top_k=Config.RERANK_TOP_K_OUTPUT)
        
        # C. Form Context (The "Answer")
        # Собираем контекст так же, как он пойдет в LLM
        final_texts = [r['text'] for r in reranked]
        
        if not final_texts:
            answer_text = "Информации недостаточно (Context Empty)"
        else:
            # Склеиваем через разделитель для наглядности в CSV
            answer_text = "\n\n--- CHUNK ---\n\n".join(final_texts)
            
            # Добавляем скоры для отладки (опционально)
            # answer_text += f"\n\n[Best Score: {reranked[0]['score']:.4f}]"

        results.append({
            "q_id": q_id,
            "answer": answer_text
        })
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(questions_df)} questions...")

    # 4. Save
    output_path = Config.OUTPUT_FILE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Retrieval submission saved to {output_path}")
    logger.info(f"⏱️ Time taken: {elapsed:.1f}s")

if __name__ == "__main__":
    main()