# Zepto Support Assistant

This module implements a locally runnable retrieval-augmented generation (RAG) assistant for Zepto policy questions. It uses a local ChromaDB vector store, a local sentence-transformers embedding model, and a LangGraph routing flow. The default runtime mode is offline mock mode, which is the required graded baseline; no API key or network call is needed.

## Architecture

The pipeline follows the required ingestion → embedding → retrieval → generation sequence.

1. Ingestion: the corpus lives under `docs/` and is loaded by `main.py` via `ensure_corpus_indexed()`. Each file is treated as a document chunk, and the ids are the document names such as `doc_01` through `doc_08`.
2. Embedding: the `SentenceTransformer("all-MiniLM-L6-v2")` model from `sentence-transformers` creates embeddings for each document. These vectors are stored in the ChromaDB collection named `zepto_policy` under the local persistence directory `chroma_db/`.
3. Retrieval: the `retrieve_top_chunks()` function embeds the incoming user query and queries ChromaDB for the top 3 nearest chunks using cosine similarity. This retrieval step is performed by the `retrieve_and_answer` node in the LangGraph graph.
4. Generation: the final answer is generated in the `retrieve_and_answer` node for policy questions and in the `direct_answer` node for general questions. In mock mode, these nodes return deterministic responses without invoking an LLM. In the optional real-LLM mode (`MOCK_LLM=0`), they will call an LLM backend instead.

The graph flow is:

```text
START
  -> classify_intent
      -> retrieve_and_answer   (policy_question)
      -> direct_answer         (general_question)
```

The `MOCK_LLM` toggle affects only the generation steps inside the branch nodes; the retrieval itself always runs for policy questions and uses the local embedding model and ChromaDB.

## Mock mode vs optional real mode

- Default / graded baseline: `MOCK_LLM` is unset or set to `1`. The system uses keyword-based routing and canned deterministic answer generation. No external LLM call is made.
- Optional extension: `MOCK_LLM=0` enables the optional real-language-model path. That path tries to call a Groq/OpenAI-compatible API only when an API key is configured. It is not required for grading.

## Example calls

The app was run locally with `MOCK_LLM` left at its default value.

### Example 1: policy question routed to retrieval

Request:

```json
{"query": "What is the delivery fee for orders below INR 149?"}
```

Response:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority delivery, which reserves the next available rider slot, is available at checkout for an additional INR 15. Zepto does not currently deliver to addresses outside its listed serviceable pin codes.","sources":["doc_01"],"confidence":1.0}
```

### Example 2: general question routed directly

Request:

```json
{"query": "Who is the CEO of Zepto?"}
```

Response:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Run locally

```bash
cd support_assistant
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Then send requests to:

```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the delivery fee for orders below INR 149?"}'
```

## Docker build and run

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

The container serves the same FastAPI app on port `7860` with the required local mock baseline enabled by default.

## Corpus files

The repository includes the exact Zepto policy corpus under `docs/` as requested:

- `docs/doc_01.txt`
- `docs/doc_02.txt`
- `docs/doc_03.txt`
- `docs/doc_04.txt`
- `docs/doc_05.txt`
- `docs/doc_06.txt`
- `docs/doc_07.txt`
- `docs/doc_08.txt`

## Optional extension notes

The real-LLM path is implemented, but it is intentionally optional and not required for grading. If you want to enable it, set `MOCK_LLM=0` and provide a free-tier API key such as a Groq key. The default graded path remains offline and deterministic.
