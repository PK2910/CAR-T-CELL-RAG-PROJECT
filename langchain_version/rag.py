"""
The same RAG system, built with LangChain (side-by-side with the plain-Python version in src/).

Same data, same settings, different framework:
  vector search   QdrantVectorStore + HuggingFaceEmbeddings (bge-small, same vectors)
  keyword search  BM25Retriever, given our own word-splitting (src/bm25.py: "ALL" != "all")
  hybrid merge    EnsembleRetriever (reciprocal rank fusion, c = 60) over each retriever's top 50
  answer          ChatPromptTemplate | ChatAnthropic (Claude Sonnet 5.5, same prompt and settings)
  safety          our guardrails (src/guardrails.py), wrapped around the chain

Run:  python langchain_version/rag.py "your question"     (needs the API key in .env; you run this)
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_classic.retrievers import EnsembleRetriever  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_qdrant import QdrantVectorStore  # noqa: E402

from bm25 import tokenize  # noqa: E402
from embed import CHUNKS_FILE, EMBEDDING_MODEL, QUERY_PREFIX  # noqa: E402
from guardrails import PERSONAL_NOTE, check_answer, is_personal  # noqa: E402
from retrieve import RRF_K, device, load_client  # noqa: E402

LC_COLLECTION = "cart_all_langchain"  # created by build_index.py
CANDIDATES = 50                      # each retriever returns its top 50; the ensemble merges them
MODEL = "claude-sonnet-5-5"
PRICE_PER_MTOK = {"input": 2.00, "output": 10.00}


@lru_cache(maxsize=1)
def load_documents() -> list[Document]:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return [Document(page_content=c["text"], metadata=c) for c in chunks]


@lru_cache(maxsize=1)
def load_retriever() -> EnsembleRetriever:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device()},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_PREFIX},  # bge's question prefix
    )
    vector_store = QdrantVectorStore(client=load_client(), collection_name=LC_COLLECTION, embedding=embeddings)
    vector = vector_store.as_retriever(search_kwargs={"k": CANDIDATES})
    keyword = BM25Retriever.from_documents(load_documents(), k=CANDIDATES, preprocess_func=tokenize)
    return EnsembleRetriever(retrievers=[vector, keyword], weights=[0.5, 0.5], c=RRF_K, id_key="chunk_id")


def retrieve(question: str, k: int = 5) -> list[dict]:
    """Top-k chunks as plain dicts (same shape as src/retrieve.py), best first."""
    return [{"score": None, **d.metadata} for d in load_retriever().invoke(question)[:k]]


def ranking(question: str) -> list[int]:
    """Chunk positions in ensemble order (only the merged top-50 lists; deeper ranks are not available)."""
    position = {d.metadata["chunk_id"]: i for i, d in enumerate(load_documents())}
    return [position[d.metadata["chunk_id"]] for d in load_retriever().invoke(question)]


def answer(question: str, k: int = 5) -> dict:
    """Full RAG with LangChain: retrieve -> prompt | Claude -> guardrails. Same result shape as src/generate.py."""
    # Imported here so that retrieval-only use (the free search test) never loads the API key
    from langchain_anthropic import ChatAnthropic
    from generate import EFFORT, SYSTEM_PROMPT, format_sources

    chunks = retrieve(question, k=k)
    personal = is_personal(question)  # input guardrail
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "<sources>\n{sources}\n</sources>\n\nQuestion: {question}{note}"),
    ])
    llm = ChatAnthropic(model=MODEL, max_tokens=16000, output_config={"effort": EFFORT},
                        betas=["server-side-fallback-2026-07-01"], model_kwargs={"fallbacks": "default"})
    chain = prompt | llm
    message = chain.invoke({"sources": format_sources(chunks), "question": question,
                            "note": f"\n\n{PERSONAL_NOTE}" if personal else ""})

    stop_reason = message.response_metadata.get("stop_reason")
    text = ("[The model declined to answer this question.]" if stop_reason == "refusal" else message.text)
    guard = check_answer(question, text)  # output guardrail
    usage = message.usage_metadata or {}
    tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return {
        "question": question,
        "answer": guard["answer"],
        "original_answer": text if guard["action"] == "replaced" else None,
        "guardrail": {"personal": guard["personal"], "doses_found": guard["doses_found"], "action": guard["action"]},
        "sources": chunks,
        "model": message.response_metadata.get("model", MODEL),
        "stop_reason": stop_reason,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cost_usd": round((tokens_in * PRICE_PER_MTOK["input"] + tokens_out * PRICE_PER_MTOK["output"]) / 1_000_000, 5),
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What drugs are used to manage cytokine release syndrome after CAR-T?"
    result = answer(q)
    print(f"\nQ: {q}\n\n{result['answer']}\n\nSources:")
    for n, c in enumerate(result["sources"], 1):
        print(f"  [{n}] {c['first_author']} et al. ({c['year']}) https://pubmed.ncbi.nlm.nih.gov/{c['pmid']}/")
    print(f"\n(LangChain · {result['model']} · ~${result['cost_usd']:.4f})")
