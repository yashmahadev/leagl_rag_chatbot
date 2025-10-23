# hybrid_retriever_optimized.py
import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from concurrent.futures import ThreadPoolExecutor

# =====================================================
# CONFIGURATION
# =====================================================
CHROMA_PATH = "chroma_db"
DOCS_PATH = "data/docs"

# =====================================================
# MODEL INITIALIZATION (done once)
# =====================================================
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Persistent vector store
chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

# CrossEncoder for reranking (semantic precision)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

# =====================================================
# LOAD DOCUMENTS FOR BM25
# =====================================================
def load_bm25_corpus():
    """Load and cache all text documents for BM25 search."""
    documents, sources = [], []
    for root, _, files in os.walk(DOCS_PATH):
        for f in files:
            if f.endswith(".txt"):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as doc:
                        text = doc.read().strip()
                        if text:
                            documents.append(text)
                            sources.append(path)
                except Exception as e:
                    print(f"⚠️ Error reading {path}: {e}")
    return documents, sources

print("🔍 Initializing BM25 corpus...")
corpus, corpus_paths = load_bm25_corpus()
bm25 = BM25Okapi([d.split() for d in corpus]) if corpus else None
print(f"✅ BM25 initialized with {len(corpus)} documents.")

# =====================================================
# BM25 SEARCH
# =====================================================
def bm25_search(query, top_k=5):
    """Lexical BM25 retrieval."""
    if not bm25:
        return []
    scores = bm25.get_scores(query.split())
    ranked = np.argsort(scores)[::-1][:top_k]
    return [{"text": corpus[i], "score": float(scores[i]), "source": corpus_paths[i]} for i in ranked]

# =====================================================
# VECTOR (CHROMA) SEARCH
# =====================================================
def chroma_search(query, top_k=5):
    """Semantic vector search using Chroma."""
    try:
        results = chroma_db.similarity_search_with_score(query, k=top_k)
        return [
            {"text": doc.page_content, "score": float(score), "source": doc.metadata.get("source", "")}
            for doc, score in results
        ]
    except Exception as e:
        print(f"⚠️ Chroma search failed: {e}")
        return []

# =====================================================
# HYBRID RETRIEVAL
# =====================================================
def retrieve(query, top_k=5):
    """Hybrid retrieval combining BM25 + Vector search + Reranker."""
    if not query.strip():
        return []

    # Parallelize lexical + vector search
    with ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(bm25_search, query, 10)
        vector_future = executor.submit(chroma_search, query, 10)
        bm25_results, vector_results = bm25_future.result(), vector_future.result()

    combined = bm25_results + vector_results
    if not combined:
        return []

    # Deduplicate
    unique = {r["text"]: r for r in combined}.values()

    # Normalize scores for fairness
    bm25_scores = np.array([r["score"] for r in unique])
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-8)

    # Rerank with semantic model
    pairs = [(query, r["text"]) for r in unique]
    semantic_scores = reranker.predict(pairs)

    # Combine lexical + semantic weighting
    combined_scores = 0.3 * bm25_scores + 0.7 * semantic_scores
    reranked = sorted(zip(unique, combined_scores), key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, _ in reranked[:top_k]]
    print(f"📘 Retrieved {len(top_docs)} top documents for: '{query}'")
    return top_docs

# =====================================================
# TEST RUN
# =====================================================
if __name__ == "__main__":
    queries = [
        "What is the punishment for murder under IPC?",
        "Explain Section 420 IPC.",
        "What is the procedure for arrest under CrPC?",
        "Explain powers of NIA under NIA Act."
    ]

    for q in queries:
        print(f"\n🔎 Query: {q}")
        docs = retrieve(q)
        for i, d in enumerate(docs, start=1):
            print(f"\nResult {i}: {d['text'][:400]}...\nSource: {d['source']}")
