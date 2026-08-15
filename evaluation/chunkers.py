"""
Chonkie chunking adapter for LaRA's RAG path.

Single seam: get_nodes(text, strategy) -> list[TextNode]. Swaps in for the
SentenceSplitter + IngestionPipeline block in eval_rag.py's process_example().
SimpleHybridSearcher re-embeds nodes itself (VectorStoreIndex(nodes, embed_model=...)
in search/simpleHybridSearcher.py), so nodes only need .text set here, not embeddings.

chunk_size/chunk_overlap are held constant (600/100, matching the SentenceSplitter
baseline LaRA already used) across all 5 strategies so the comparison isolates
chunking strategy, not chunk granularity. Recursive/Semantic/Fast chunkers don't take
an overlap parameter (chonkie doesn't expose one for them); that's a real difference
between strategies, not an oversight.
"""
from llama_index.core.schema import TextNode
from chonkie import TokenChunker, SentenceChunker, RecursiveChunker, SemanticChunker, FastChunker

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100

STRATEGIES = ["token", "sentence", "recursive", "semantic", "fast"]


def _build_chunker(strategy, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    if strategy == "token":
        return TokenChunker(tokenizer="character", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "sentence":
        return SentenceChunker(tokenizer="character", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if strategy == "recursive":
        return RecursiveChunker(tokenizer="character", chunk_size=chunk_size)
    if strategy == "semantic":
        # embedding_model defaults to chonkie's own minishlab/potion-base-32M (model2vec,
        # CPU-light); reuse across runs so the same strategy is deterministic and cheap.
        return SemanticChunker(chunk_size=chunk_size)
    if strategy == "fast":
        return FastChunker(chunk_size=chunk_size)
    raise ValueError(f"unknown chunking strategy: {strategy!r} (choose from {STRATEGIES})")


def get_nodes(text, strategy, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    chunker = _build_chunker(strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker(text)
    return [TextNode(text=c.text) for c in chunks]
