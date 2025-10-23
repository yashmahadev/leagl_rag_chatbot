import os
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------
#  CONFIGURATION
# ---------------------------
CHROMA_PATH = "chroma_db"
DOCS_PATH = "data/docs"

# Initialize embedding model (strong semantic model)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize vector database (persistent ChromaDB)
chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

# Initialize CrossEncoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------
#  LOAD DOCUMENTS FOR BM25
# ---------------------------
def load_documents_for_bm25():
    """Loads all documents from disk to use in BM25 lexical search."""
    documents = []
    file_paths = []

    for root, _, files in os.walk(DOCS_PATH):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if len(text) > 0:
                        documents.append(text)
                        file_paths.append(path)

    return documents, file_paths


print("🔍 Loading BM25 corpus...")
corpus, corpus_paths = load_documents_for_bm25()
bm25 = BM25Okapi([doc.split() for doc in corpus]) if corpus else None
print(f"✅ Loaded {len(corpus)} documents into BM25.")


# ---------------------------
#  SEARCH METHODS
# ---------------------------
def bm25_search(query, top_k=5):
    """Performs lexical BM25 search."""
    if not bm25:
        return []

    scores = bm25.get_scores(query.split())
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = [
        {"text": corpus[i], "score": scores[i], "source": corpus_paths[i]}
        for i in ranked_indices
    ]
    return results


def chroma_search(query, top_k=5):
    """Performs semantic vector search from ChromaDB."""
    results = chroma_db.similarity_search_with_score(query, k=top_k)
    formatted = [
        {"text": doc.page_content, "score": float(score), "source": doc.metadata.get("source", "")}
        for doc, score in results
    ]
    return formatted


# ---------------------------
#  HYBRID RETRIEVAL + RERANK
# ---------------------------
def retrieve(query, top_k=5):
    """Performs hybrid retrieval (BM25 + Vector + CrossEncoder reranking)."""
    bm25_results = bm25_search(query, top_k=10)
    vector_results = chroma_search(query, top_k=10)
    combined = bm25_results + vector_results

    if not combined:
        return []

    # Deduplicate results
    unique_docs = {r["text"]: r for r in combined}.values()

    # Rerank using semantic CrossEncoder
    pairs = [(query, r["text"]) for r in unique_docs]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in reranked[:top_k]]

    print(f"📘 Retrieved {len(top_docs)} documents for query: '{query}'")
    return top_docs


# ---------------------------
#  TEST RUN
# ---------------------------
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
