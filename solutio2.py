"""
python solution.py --ingest для чанкинга 
python solution.py --process для генерации сабмита
"""

import logging
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================
# Embeddings Logic 
# ==========================================

class FridaEmbeddingConfig:
    """
    Configuration for FRIDA embeddings - specialized Russian model.
    """
    
    def __init__(self):
        self.model_name = "ai-forever/FRIDA"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dim = 1536  
        self.max_length = 512
        self.batch_size = 6  
        
        logger.info(f"FRIDA embedding config: model={self.model_name}, device={self.device}")
    
    def validate(self):
        """Validate configuration."""
        if not self.model_name:
            raise ValueError("Model name is required")
        if self.embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

# Global model instances
_model = None
_model_config = None
_model_lock = None

def get_frida_model():
    """Get or initialize FRIDA model (thread-safe)."""
    global _model, _model_config, _model_lock
    
    if _model_lock is None:
        _model_lock = threading.Lock()
    
    if _model is None:
        with _model_lock:
            # Double-check pattern
            if _model is not None:
                return _model
                
            try:
                _model_config = FridaEmbeddingConfig()
                _model_config.validate()
                
                logger.info(f"Loading FRIDA embedding model: {_model_config.model_name}")
                logger.info(f"Using device: {_model_config.device}")
                
                # Load FRIDA model with CLS pooling (default)
                _model = SentenceTransformer(_model_config.model_name, device=_model_config.device)
                
                logger.info(f"FRIDA embedding model loaded successfully")
                logger.info(f"Model parameters: {sum(p.numel() for p in _model.parameters()):,}")
                
            except ImportError:
                raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
            except Exception as e:
                raise RuntimeError(f"Failed to load FRIDA embedding model: {e}")
    
    return _model

def embed_text_frida(text: str, is_query: bool = True) -> List[float]:
    """
    Generate embedding for a single text using FRIDA model with optimized prefixes.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    model = get_frida_model()
    config = _model_config
    
    try:
        # Clean text
        cleaned = text.strip()
        
        # Use optimized prompts for FRIDA
        if is_query:
            # For queries, use search_query prefix
            prompt_name = "search_query"
        else:
            # For documents, use search_document prefix
            prompt_name = "search_document"
        
        # Generate embedding with prompt
        with torch.no_grad():
            embedding = model.encode(
                cleaned, 
                prompt_name=prompt_name,
                convert_to_numpy=True  # Use numpy instead of tensor
            )
            
            # Convert to list
            embedding_list = embedding.tolist()
        
        if len(embedding_list) != config.embedding_dim:
            logger.warning(f"Embedding dimension mismatch: expected {config.embedding_dim}, got {len(embedding_list)}")
        
        return embedding_list
        
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}")

def embed_texts_frida(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches using FRIDA model.
    """
    if not texts:
        return []
    
    # Filter out empty texts
    valid_indices = [(i, text.strip()) for i, text in enumerate(texts) if text and text.strip()]
    
    if not valid_indices:
        return []
    
    model = get_frida_model()
    config = _model_config
    
    embeddings = [None] * len(texts)  # Preserve original order
    
    try:
        # Process in batches
        batch_size = min(config.batch_size, len(valid_indices))
        
        for i in range(0, len(valid_indices), batch_size):
            batch = valid_indices[i:i + batch_size]
            batch_texts = [text for _, text in batch]
            batch_indices = [idx for idx, _ in batch]
            
            # Use appropriate prompt for FRIDA
            prompt_name = "search_query" if is_query else "search_document"
            
            # Generate embeddings for batch
            with torch.no_grad():
                batch_embeddings = model.encode(
                    batch_texts,
                    prompt_name=prompt_name,
                    convert_to_numpy=True,
                    batch_size=len(batch_texts)
                )
                
                # Convert to list and place in original order
                batch_embeddings_list = batch_embeddings.tolist()
                
                for idx, embedding in zip(batch_indices, batch_embeddings_list):
                    embeddings[idx] = embedding
        
        # Filter out None values (empty texts)
        final_embeddings = [emb for emb in embeddings if emb is not None]
        
        return final_embeddings
        
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise RuntimeError(f"Batch embedding generation failed: {e}")

# ==========================================
# Vector Store Logic 
# ==========================================

@dataclass
class QdrantConfig:
    """
    Configuration for Qdrant vector store.
    """
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    collection_name: str = "documents1"
    vector_size: int = 1536
    distance: str = "Cosine"

_qdrant_config = QdrantConfig()
_qdrant_client: Optional[QdrantClient] = None

def get_qdrant_client() -> QdrantClient:
    """
    Get a singleton Qdrant client instance.
    """
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=_qdrant_config.url,
            api_key=_qdrant_config.api_key,
        )
    return _qdrant_client

def ensure_collection() -> None:
    """
    Ensure that the target collection exists with the expected configuration.
    """
    client = get_qdrant_client()
    
    collections = client.get_collections().collections
    existing = {c.name for c in collections}

    if _qdrant_config.collection_name not in existing:
        logger.info(f"Creating collection: {_qdrant_config.collection_name}")
        client.create_collection(
            collection_name=_qdrant_config.collection_name,
            vectors_config=qmodels.VectorParams(
                size=_qdrant_config.vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
    else:
        logger.info(f"Collection {_qdrant_config.collection_name} already exists")

def upsert_chunks(chunks: Iterable[Dict[str, Any]]) -> None:
    """
    Upsert multiple chunks into Qdrant.
    """
    client = get_qdrant_client()
    points = []
    
    for c in chunks:
        if "id" not in c or "vector" not in c:
            raise ValueError("Each chunk must have 'id' and 'vector' fields")
        payload = c.get("payload") or {}
        
        points.append(
            qmodels.PointStruct(
                id=c["id"],
                vector=c["vector"],
                payload=payload,
            )
        )

    if not points:
        return

    client.upsert(
        collection_name=_qdrant_config.collection_name,
        points=points,
    )

# ==========================================
# Splitter Logic 
# ==========================================

def _approx_token_len(text: str) -> int:
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))

def _split_by_delimiter(text: str, delimiter: str) -> List[str]:
    if not text:
        return []
    if delimiter in ("\n\n", "\n"):
        parts = [p.strip() for p in text.split(delimiter)]
        return [p for p in parts if p]
    
    pattern = f"({re.escape(delimiter)})"
    raw_parts = re.split(pattern, text)
    segments = []
    current = ""
    for part in raw_parts:
        if part == delimiter:
            current += part
            seg = current.strip()
            if seg:
                segments.append(seg)
            current = ""
        else:
            current += part
    if current.strip():
        segments.append(current.strip())
    return [s for s in segments if s]

def _hierarchical_split(text: str, delimiters_levels: List[List[str]]) -> List[str]:
    if not text:
        return []
    segments = [text]
    for level_delims in delimiters_levels:
        new_segments = []
        for seg in segments:
            if _approx_token_len(seg) <= 512:
                new_segments.append(seg)
                continue
            to_process = [seg]
            for d in level_delims:
                next_parts = []
                for piece in to_process:
                    split_parts = _split_by_delimiter(piece, d)
                    if split_parts:
                        next_parts.extend(split_parts)
                    else:
                        next_parts.append(piece)
                to_process = next_parts
            for p in to_process:
                p = p.strip()
                if p:
                    new_segments.append(p)
        segments = new_segments
    return [s for s in segments if s.strip()]

def recursive_split(
    text: str,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 50,
) -> List[str]:
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
        
    approx_len = _approx_token_len(text)
    if approx_len <= chunk_size_tokens:
        return [text]

    delimiters_levels = [["\n\n"], [".", "?", "!"]]
    units = _hierarchical_split(text, delimiters_levels)
    if not units:
        return [text]

    chunks = []
    current_units = []
    current_tokens = 0

    def flush_current():
        nonlocal current_units, current_tokens
        if not current_units:
            return
        chunk_text = " ".join(u.strip() for u in current_units if u.strip()).strip()
        if chunk_text:
            chunks.append(chunk_text)
        current_units = []
        current_tokens = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        unit_tokens = _approx_token_len(unit)

        if unit_tokens > chunk_size_tokens * 1.5:
            words = re.findall(r"\w+|[^\w\s]", unit)
            start = 0
            while start < len(words):
                end = min(start + chunk_size_tokens, len(words))
                sub = " ".join(words[start:end])
                if sub.strip():
                    if current_tokens + _approx_token_len(sub) > chunk_size_tokens:
                        flush_current()
                    current_units.append(sub)
                    current_tokens += _approx_token_len(sub)
                    flush_current()
                start = end
            continue

        if current_tokens + unit_tokens <= chunk_size_tokens:
            current_units.append(unit)
            current_tokens += unit_tokens
        else:
            flush_current()
            if chunk_overlap_tokens > 0 and chunks:
                last_chunk = chunks[-1]
                last_tokens = re.findall(r"\w+|[^\w\s]", last_chunk)
                if last_tokens:
                    overlap_slice = last_tokens[-chunk_overlap_tokens:]
                    overlap_text = " ".join(overlap_slice).strip()
                    if overlap_text:
                        current_units = [overlap_text]
                        current_tokens = _approx_token_len(overlap_text)
                    else:
                        current_units = []
                        current_tokens = 0
                else:
                    current_units = []
                    current_tokens = 0
            else:
                current_units = []
                current_tokens = 0
            
            if unit_tokens <= chunk_size_tokens:
                current_units.append(unit)
                current_tokens += unit_tokens
            else:
                words = re.findall(r"\w+|[^\w\s]", unit)
                start = 0
                while start < len(words):
                    end = min(start + chunk_size_tokens, len(words))
                    sub = " ".join(words[start:end])
                    if sub.strip():
                        if current_tokens + _approx_token_len(sub) > chunk_size_tokens:
                            flush_current()
                        current_units.append(sub)
                        current_tokens += _approx_token_len(sub)
                        flush_current()
                    start = end

    flush_current()
    return [c.strip() for c in chunks if c and c.strip()]

# ==========================================
# Ingestion Logic
# ==========================================

def _extract_title(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")[:5]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 10:
            continue
        if line.endswith(":") or line.endswith("?") or line.istitle():
            return line
        if len(line) <= 100 and line[0].isupper():
            return line
    
    first_sentence = text.split(".")[0].split("!")[0].split("?")[0].strip()
    if len(first_sentence) <= 100:
        return first_sentence
    return text[:50] + "..."

def batch_ingest_documents(batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {"processed": 0, "failed": 0, "total_chunks": 0}
    
    all_texts = []
    all_sources = []
    all_doc_ids = []

    for item in batch_data:
        text, source, doc_id = item
        if text and text.strip():
            all_texts.append(text.strip())
            all_sources.append(source)
            all_doc_ids.append(doc_id)

    if not all_texts:
        return stats

    try:
        all_chunks = []
        chunk_to_doc_mapping = []
        
        for i, text in enumerate(all_texts):
            chunks = recursive_split(
                text,
                chunk_size_tokens=500,
                chunk_overlap_tokens=50,
            )
            if chunks:
                all_chunks.extend(chunks)
                chunk_to_doc_mapping.extend(
                    [(i, chunk_idx) for chunk_idx in range(len(chunks))]
                )

        if not all_chunks:
            return stats

        logger.debug(f"Embedding {len(all_chunks)} chunks...")
        vectors = embed_texts_frida(all_chunks, is_query=False)
        
        points = []
        for chunk_idx, (text, vec) in enumerate(zip(all_chunks, vectors)):
            if not text.strip():
                continue

            doc_idx, chunk_local_idx = chunk_to_doc_mapping[chunk_idx]
            doc_id = all_doc_ids[doc_idx]
            source = all_sources[doc_idx]

            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-{chunk_local_idx}")
            )
            points.append(
                {
                    "id": point_id,
                    "vector": vec,
                    "payload": {
                        "doc_id": doc_id,
                        "chunk_index": chunk_local_idx,
                        "text": text.strip(),
                        "source": source,
                    },
                }
            )

        if points:
            upsert_chunks(points)
            stats["processed"] = len(all_texts)
            stats["total_chunks"] = len(points)

    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        stats["failed"] = len(all_texts)

    return stats

def ingest_websites_data(csv_path: str, batch_size: int = 25) -> Dict[str, Any]:
    logger.info(f"Starting ingestion from {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} websites from CSV")
    
    df = df.dropna(subset=["text", "web_id"])
    df = df[df["text"].str.strip() != ""]
    
    stats = {"total_websites": len(df), "processed": 0, "failed": 0, "total_chunks": 0}
    
    documents_data = []
    for idx, row in df.iterrows():
        text = row["text"].strip()
        if text:
            source = f"web_id:{row['web_id']}|url:{row.get('url', '')}|kind:{row.get('kind', '')}"
            doc_id = str(row["web_id"])
            documents_data.append((text, source, doc_id))

    logger.info(f"Processing {len(documents_data)} documents in batches of {batch_size}...")
    
    batches = [
        documents_data[i : i + batch_size]
        for i in range(0, len(documents_data), batch_size)
    ]

    ensure_collection()

    for batch_idx, batch_data in enumerate(batches):
        try:
            batch_stats = batch_ingest_documents(batch_data)
            stats["processed"] += batch_stats["processed"]
            stats["failed"] += batch_stats["failed"]
            stats["total_chunks"] += batch_stats["total_chunks"]

            processed_total = stats["processed"] + stats["failed"]
            if (batch_idx + 1) % 5 == 0:
                logger.info(
                    f"Progress: {processed_total}/{len(documents_data)} documents "
                    f"(chunks: {stats['total_chunks']})"
                )

        except Exception as e:
            logger.error(f"Batch {batch_idx} processing failed: {e}")
            stats["failed"] += len(batch_data)

    logger.info(f"Ingestion complete. Stats: {stats}")
    return stats

# ==========================================
# Main Pipeline Logic 
# ==========================================

def retrieve_baseline(query: str, top_k: int = 5) -> list:
    """
    Ultra-simple retrieval: semantic search + extract unique web_ids.
    Robust to QdrantClient version changes (handles missing .search method).
    """
    # 1. Get query embedding
    query_vec = embed_text_frida(query, is_query=True)

    # 2. Semantic search - get more chunks to ensure we have enough unique web_ids
    client = get_qdrant_client()
    results = []
    
    try:
        # Check available methods dynamically to handle version mismatches
        if hasattr(client, 'search'):
            # Classic API (v0.10.x - v1.9.x+)
            results = client.search(
                collection_name=_qdrant_config.collection_name,
                query_vector=query_vec,
                limit=50,
                with_payload=True,
            )
        elif hasattr(client, 'query_points'):
            # Newer API (v1.10+) or Unified API
            # Note: query_points returns a QueryResponse, so we access .points
            response = client.query_points(
                collection_name=_qdrant_config.collection_name,
                query=query_vec, # Pass vector as query
                limit=50,
                with_payload=True,
            )
            results = response.points
        else:
            # Fallback to direct HTTP API if high-level methods fail
            logger.warning("High-level search methods missing. Trying HTTP API.")
            if hasattr(client, 'http') and hasattr(client.http, 'points_api'):
                search_request = qmodels.SearchRequest(
                    vector=query_vec,
                    limit=50,
                    with_payload=True
                )
                response = client.http.points_api.search_points(
                    collection_name=_qdrant_config.collection_name,
                    search_request=search_request
                )
                results = response.result
            else:
                raise AttributeError("Could not find a valid search method on QdrantClient")

    except Exception as e:
        logger.error(f"Retrieval failed for query: {query[:50]}... Error: {e}")
        return [-1] * top_k

    # 3. Extract unique web_ids in order
    web_ids = []
    seen = set()

    for result in results:
        # Handle difference in object structure between versions if necessary
        # Usually result is a ScoredPoint with .payload
        payload = getattr(result, 'payload', None)
        if not payload:
            continue
            
        web_id = payload.get("doc_id")
        if web_id and web_id not in seen:
            try:
                web_ids.append(int(web_id))
                seen.add(web_id)
                if len(web_ids) >= top_k:
                    break
            except (ValueError, TypeError):
                continue

    # 4. Pad with -1 if needed
    while len(web_ids) < top_k:
        web_ids.append(-1)

    return web_ids[:top_k]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Standalone RAG Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion")
    parser.add_argument("--process", action="store_true", help="Run question processing")
    args = parser.parse_args()

    # If no arguments provided, run both by default (or ask user)
    # For now, let's default to just processing if no args, to be safe.
    if not args.ingest and not args.process:
        logger.info("No arguments provided. Usage: python solution.py --ingest --process")
        logger.info("Defaulting to --process only.")
        args.process = True

    # Paths
    websites_csv = Path("data/websites_updated.csv")
    questions_csv = Path("data/questions_clean.csv")
    output_csv = Path("final/submission.csv")

    if args.ingest:
        if not websites_csv.exists():
            logger.error(f"❌ Websites file not found: {websites_csv}")
            return
        ingest_websites_data(str(websites_csv))

    if args.process:
        if not questions_csv.exists():
            logger.error(f"❌ Questions file not found: {questions_csv}")
            return

        # Create output directory
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Load questions
        logger.info(f"Loading questions from {questions_csv}...")
        questions_df = pd.read_csv(questions_csv)
        logger.info(f"Loaded {len(questions_df)} questions")

        # Process questions
        results = []
        start_time = time.time()

        for idx, row in questions_df.iterrows():
            q_id = row["q_id"]
            query = str(row["query"]).strip()

            try:
                # Retrieve top-5 web_ids
                web_ids = retrieve_baseline(query, top_k=5)
                results.append({"q_id": q_id, "web_list": str(web_ids)})

                if (idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (idx + 1)
                    remaining = avg_time * (len(questions_df) - idx - 1)
                    logger.info(
                        f"Processed {idx + 1}/{len(questions_df)} "
                        f"({(idx + 1) / len(questions_df) * 100:.1f}%) "
                        f"- ETA: {remaining / 60:.1f} min"
                    )

            except Exception as e:
                logger.error(f"Failed to process question {q_id}: {e}")
                results.append({"q_id": q_id, "web_list": str([-1, -1, -1, -1, -1])})

        # Save submission
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(output_csv, index=False)

        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ Submission saved to {output_csv}")
        logger.info(f"⏱️  Total time: {total_time / 60:.1f} minutes")
        logger.info(f"⚡ Avg per question: {total_time / len(questions_df):.2f} seconds")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
