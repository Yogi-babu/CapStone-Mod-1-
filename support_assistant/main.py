import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.errors import NotFoundError
from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict

try:
    from groq import Groq  # type: ignore[import-not-found]
except ModuleNotFoundError:  # optional dependency for non-mock mode
    Groq = None

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policy"
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


class AskRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float
    retrieved_chunks: list[str]


STRUCTURED_PROMPT_TEMPLATE = """
Role: You are Zepto's support assistant. You answer questions using only Zepto policy information from the provided context.
Context: The following retrieved policy excerpts are the only source of truth. Use them to answer the question exactly and do not invent any details.
Task: Answer the question clearly and briefly, citing the relevant Zepto policy when it is present in the context.
Format: Return a single valid JSON object with keys: answer, sources, confidence.
Length: Keep the answer concise but complete, typically 1-3 sentences.
Negative constraint: Do not answer using information not present in the provided context. Do not speculate, infer unstated policies, or mention unsupported claims.
Few-shot example:
User question: "What is the delivery fee for orders below INR 149?"
Context: "Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee."
Output: {"answer": "Orders below INR 149 incur a flat INR 25 delivery fee.", "sources": ["doc_01"], "confidence": 1.0}
"""


def is_mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "1").strip() not in {"0", "false", "False", "FALSE"}


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except (ValueError, NotFoundError):
        return client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def ensure_corpus_indexed() -> None:
    collection = get_collection()
    existing_ids = set(collection.get(include=["ids"]) ["ids"]) if collection.count() > 0 else set()
    if existing_ids:
        return

    files = sorted(DOCS_DIR.glob("doc_*.txt"))
    if not files:
        raise FileNotFoundError(f"No corpus docs found in {DOCS_DIR}")

    documents = []
    ids = []
    metadata = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        doc_id = file_path.stem
        ids.append(doc_id)
        documents.append(text)
        metadata.append({"source_file": file_path.name})

    collection.upsert(documents=documents, ids=ids, metadatas=metadata)


def retrieve_top_chunks(query: str, top_k: int = 3) -> dict:
    collection = get_collection()
    embedding = get_embedder().encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return {
        "ids": results["ids"][0],
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }


def classify_intent_node(state: GraphState) -> GraphState:
    query = state["query"]
    lowered = query.lower()
    if any(keyword in lowered for keyword in POLICY_KEYWORDS):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"
    return state


def route_intent(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def generate_real_llm_response(query: str, context_chunks: list[str], sources: list[str]) -> dict:
    context_text = "\n---\n".join(context_chunks) if context_chunks else "No retrieved policy context provided."
    prompt = f"""
{STRUCTURED_PROMPT_TEMPLATE}
User question: {query}
Context: {context_text}
"""
    try:
        if Groq is None:
            raise RuntimeError("Groq package is not installed for real-LLM mode.")

        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No LLM API key configured for real-LLM mode.")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=250,
        )
        text = response.choices[0].message.content
        try:
            parsed = __import__("json").loads(text)
            answer = parsed.get("answer", "Unable to answer using the provided policy context.")
            return {
                "answer": answer,
                "sources": parsed.get("sources", sources),
                "confidence": float(parsed.get("confidence", 0.5)),
            }
        except Exception:
            return {
                "answer": text.strip(),
                "sources": sources,
                "confidence": 0.5,
            }
    except Exception as exc:
        return {
            "answer": f"Real LLM path unavailable: {exc}",
            "sources": sources,
            "confidence": 0.0,
        }


def retrieve_and_answer_node(state: GraphState) -> GraphState:
    query = state["query"]
    retrieved = retrieve_top_chunks(query, top_k=3)
    state["retrieved_chunks"] = retrieved["documents"]
    state["sources"] = retrieved["ids"]

    if is_mock_mode():
        top_chunk = retrieved["documents"][0]
        snippet = top_chunk[:200]
        state["answer"] = f"Based on the retrieved context: {snippet}"
        state["confidence"] = 1.0
        return state

    answer_payload = generate_real_llm_response(query, retrieved["documents"], retrieved["ids"])
    state["answer"] = answer_payload["answer"]
    state["sources"] = answer_payload["sources"]
    state["confidence"] = float(answer_payload["confidence"])
    return state


def direct_answer_node(state: GraphState) -> GraphState:
    if is_mock_mode():
        state["answer"] = "I can only answer questions about Zepto policies right now."
        state["sources"] = []
        state["confidence"] = 1.0
        return state

    prompt = f"{STRUCTURED_PROMPT_TEMPLATE}\nUser question: {state['query']}\n"
    response = generate_real_llm_response(state["query"], [], [])
    state["answer"] = response["answer"]
    state["sources"] = []
    state["confidence"] = float(response["confidence"])
    return state


ensure_corpus_indexed()

builder = StateGraph(GraphState)
builder.add_node("classify_intent", classify_intent_node)
builder.add_node("retrieve_and_answer", retrieve_and_answer_node)
builder.add_node("direct_answer", direct_answer_node)
builder.set_entry_point("classify_intent")
builder.add_conditional_edges("classify_intent", route_intent, {
    "retrieve_and_answer": "retrieve_and_answer",
    "direct_answer": "direct_answer",
})
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()


app = FastAPI(title="Zepto Support Assistant")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: AskRequest) -> AnswerResponse:
    result = graph.invoke({
        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "retrieved_chunks": [],
    })
    return AnswerResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=float(result.get("confidence", 1.0)),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
