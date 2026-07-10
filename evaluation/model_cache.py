"""Process-wide singleton cache for the heavy retrieval models.

The RAG eval used to instantiate an embedding model and a reranker for *every*
example across a thread pool (eval_rag.process_example -> SimpleHybridSearcher).
On a 16GB machine that stacked dozens of ~1GB model copies on top of Ollama's
resident LLM and OOM-killed the run mid-study.

These helpers load each distinct model (keyed by name + config) once and hand
back the shared instance. PyTorch CPU forward passes are read-only on the model
weights, so sharing one instance across worker threads is safe; the lock only
guards first-time construction.
"""
import threading

_lock = threading.Lock()
_embed_cache = {}
_rerank_cache = {}


def get_embed_model(model_name, **kwargs):
    key = (model_name, tuple(sorted(kwargs.items())))
    with _lock:
        model = _embed_cache.get(key)
        if model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            model = HuggingFaceEmbedding(model_name=model_name, **kwargs)
            _embed_cache[key] = model
        return model


def get_reranker(model_name, top_n):
    key = (model_name, top_n)
    with _lock:
        reranker = _rerank_cache.get(key)
        if reranker is None:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            reranker = SentenceTransformerRerank(top_n=top_n, model=model_name)
            _rerank_cache[key] = reranker
        return reranker
