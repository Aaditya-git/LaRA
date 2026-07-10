# LaRA x ChunkResearch: pluggable Chonkie chunkers (scope + handoff)

Status: SCOPED, not yet implemented. Decision made to keep everything inside the LaRA repo.
Last working session ended here. Pick up from "Next step" at the bottom.

## The goal in one sentence

Bring ChunkResearch's 7 Chonkie chunking strategies into LaRA's RAG path so we can
measure how chunking strategy affects RAG accuracy on long book/paper/financial
documents (and on the RAG-vs-long-context axis). This extends the professor's actual
research question (chunking affects RAG accuracy), currently only tested on short
medical MCQs, to a new dataset and domain.

## Why this is the right integration (and what we are NOT doing)

LaRA and ChunkResearch measure different things and do NOT merge as-is:

| | ChunkResearch | LaRA |
|---|---|---|
| Domain | Medical MCQ (MedQA) | Long-doc free-form QA (book/paper/financial) |
| Answer | single letter A/B/C/D/Z | open-ended text |
| Scoring | exact letter match (deterministic) | LLM-as-judge |
| Retrieval src | pre-ingested Mongo collections | one document supplied per question |
| Research var | 7 chunking strategies (Chonkie) | RAG vs long-context |

The ONLY thing crossing over is the chunking step: ChunkResearch's chunking is
essentially `chonkie.chunk_with(strategy)` plus an embedder. Everything else in
ChunkResearch (MongoDB, LangGraph, the letter prompt, Pydantic LetterModel, Langfuse)
is MedQA scaffolding we do NOT carry over. LaRA already has its own retriever,
reranker, generator, and scorer; we keep all of those.

We are NOT importing Mongo. We reuse Chonkie's chunkers but keep LaRA's BGE embedding
+ hybrid (vector+BM25) search + reranker, so the chunking strategies are compared on
equal footing inside LaRA's own retrieval.

## How LaRA's RAG path works (the map)

- Chunking lives in ONE place: `evaluation/eval_rag.py`, function `process_example()`,
  lines ~96-108. The `SentenceSplitter(...)` block + `IngestionPipeline` turn the raw
  document text into `nodes`. This is the seam to replace.
- Retrieval lives in `evaluation/search/`:
  - `search/simpleHybridSearcher.py`: `load_retriever()` builds the vector retriever
    (VectorStoreIndex + HuggingFaceEmbedding/BGE) and the BM25 retriever, fuses them
    into `SimpleHybridRetriever`; `load_node_postprocessors()` is the reranker
    (SentenceTransformerRerank).
  - `search/simpleHybridRetriever.py`: the vector+BM25 fusion logic.
  - `search/baseSearcher.py`: defines `.process()` (retrieve then rerank).
- Scoring: `evaluation/compute_score_llm.py` (LLM-as-judge).

Important: the hybrid searcher RE-EMBEDS nodes itself in `load_retriever`
(VectorStoreIndex(nodes, embed_model=...)). So the embed step in eval_rag's current
IngestionPipeline is redundant; when we swap chunkers we only need to PRODUCE nodes,
not pre-embed them.

## Dataset metadata we learned (so future-me does not re-derive it)

Each query record: keys = type, level, length, file, question, answer, plus one of:
- `context_order` (location/reasoning tasks): WHERE the single answer-evidence sits in
  the document. Values: 0=start third, 1=middle third, 2=end third, "full"=spread
  across whole doc. The scorer buckets accuracy by this and writes per-position
  results to `prediction/result/<model>_order.jsonl`.
- `comp_parts` (comp task): a PAIR of thirds the answer must combine. Only three
  combos occur: [0,1] start+middle, [0,2] start+end (hardest, far apart), [1,2]
  middle+end. Currently NOT bucketed by the scorer (optional future extension).

These thirds (0/1/2) are NOT produced by any chunker. They are ground-truth labels
baked into `datasets/query/*.jsonl` at dataset-creation time, based on the character/
token position of the evidence in the whole continuous document. The runtime chunker
(SentenceSplitter / Chonkie) shreds the doc into ~600-char chunks independently and
is unaware of the thirds. A [0,2] comp question is the stress case: the two needed
chunks sit at opposite ends, so the chunker+retriever must surface both.

## Implementation plan (all inside LaRA)

1. NEW FILE `evaluation/chunkers.py` (~50 lines). Adapter: strategy name -> function
   `text -> list[llama_index TextNode]`. Runs the Chonkie chunker, wraps each chunk's
   `.text` in a `TextNode`. Sketch:
   ```python
   from chonkie import TokenChunker, SentenceChunker, RecursiveChunker, SemanticChunker, ...
   from llama_index.core.schema import TextNode
   def get_nodes(text, strategy, embed_model_name):
       chunker = _build(strategy, embed_model_name)   # one of the 7
       return [TextNode(text=c.text) for c in chunker.chunk(text)]
   ```
2. EDIT `evaluation/eval_rag.py` (~15 lines):
   - add `--chunker` CLI arg, default `sentence` (reproduces today's behavior);
   - replace lines ~96-108 (SentenceSplitter + IngestionPipeline) with
     `nodes = get_nodes(eg_txt, args.chunker, EMBED_MODEL)`;
   - put `{chunker}` in the output filename so strategies do not overwrite each other:
     `rag_preds_{eval_model}_{chunker}_{length}_{context}_{query}.jsonl`.
3. EDIT `evaluation/compute_score_llm.py` (~6 lines): add matching `--chunker` arg,
   thread it into `data_path` and into the written result tag
   (`rag_{model}_{chunker}_{length}_{context}_{query}: <acc>`). Result: one CSV with one
   row per chunker = the per-strategy accuracy table (the research output).
4. DEPS: add `chonkie`. Reuse `minishlab/potion-retrieval-32M` (ChunkResearch's
   embedder) for strategies that need an embedder, for comparability.

## Run recipe (after implementing)
```bash
cd evaluation
for c in token sentence recursive semantic late; do
  OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  OLLAMA_NUM_CTX=8192 LARA_LIMIT=3 LARA_WORKERS=1 \
  ../venv/bin/python eval_rag.py --chunker $c \
    --query_type reasoning --context_type book --context_length 32k --eval_model llama3.2
done
# then score per chunker; one CSV, one row per chunker
```

## Caveats (carried from prior work)
- Hardware (16GB M2): token/sentence/recursive/fast are cheap and fine. semantic/late
  need an embed pass (light with potion-32M). neural pulls a separate model, heaviest.
  Scope first run to the 4 cheap + semantic.
- The weak LOCAL judge still gates real numbers: with a 3B/7B judge every strategy
  reads ~0.0 and you cannot rank them. This integration builds the machinery now;
  trustworthy per-chunker rankings wait on a strong API judge (the ask already flagged
  to the professor in PROGRESS_NOTES.md).
- Scope stays at 32k; 128k does not fit in RAM (unrelated to chunking).

## Optional follow-on
Break `comp` accuracy down by `comp_parts` in the scorer (like `context_order` is for
location/reasoning), so you can show which chunker best preserves cross-section
comparisons, especially the far-apart [0,2] case.

## NEXT STEP when resuming
Implement step 1-4 above (start with `evaluation/chunkers.py`). User chose: keep it all
in the LaRA repo. Then do a smoke run on the 4 cheap chunkers + semantic at 32k/book/
reasoning with llama3.2. Remember the judge will read ~0.0; that is expected until a
strong judge is wired in.
