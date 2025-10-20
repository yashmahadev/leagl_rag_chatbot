import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import chromadb
from act_classifier import classify_act

# ========== STEP 1: Load all Act datasets ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "jsons")  # folder containing ipc.json, crpc.json, nia.json

def load_dataset(filename: str, act_name: str):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Missing file: {file_path}")
    
    df = pd.read_json(file_path)
    df["act_name"] = act_name
    # Combine chapter, section, title, and description into content_text
    df["content_text"] = df.apply(
        lambda r: f"{r.get('chapter', '')} {r.get('section', '')} {r.get('section_title', '')} {r.get('section_desc', '')}".strip(),
        axis=1
    )
    df["content_text"] = df["content_text"].fillna("").astype(str)
    
    # Return useful columns
    return df[["act_name", "chapter", "section", "section_title", "section_desc", "content_text"]]

# Load datasets
ipc_df = load_dataset("ipc.json", "IPC")
crpc_df = load_dataset("crpc.json", "CrPC")
nia_df = load_dataset("nia.json", "NIA")

df = pd.concat([ipc_df, crpc_df, nia_df], ignore_index=True)
print(f"✅ Loaded {len(df)} total sections from IPC, CrPC, and NIA.\n")

# ========== STEP 2: Initialize Embedding Model ==========
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# ========== STEP 3: Prepare ChromaDB Vector Store ==========
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_legal_db")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection_name = "legal_acts"

# Check if collection already exists
existing_collections = [c.name for c in client.list_collections()]
if collection_name in existing_collections:
    print(f"✅ ChromaDB collection '{collection_name}' exists. Skipping embedding...")
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

    # Embedding and storing in Chroma
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding sections"):
        text = row["content_text"]
        if not text.strip():
            continue
        emb = embedding_model.embed_query(text)
        collection.add(
            ids=[str(i)],
            embeddings=[emb],
            metadatas=[{
                "act_name": row["act_name"],
                "section": row["section"],
                "title": row["section_title"]
            }],
            documents=[text]
        )
    print("✅ Embeddings stored in ChromaDB.\n")

# ========== STEP 4: Prepare BM25 Retriever ==========
bm25_corpus = [doc.split() for doc in df["content_text"].tolist()]
bm25 = BM25Okapi(bm25_corpus)
print("✅ BM25 retriever ready.\n")

# ========== STEP 5: Hybrid Search Function ==========
def hybrid_search(query, top_k=5):
    # Step 1: Act Classification
    act, conf = classify_act(query)
    print(f"🎯 Predicted Act: {act} (confidence={conf:.2f})")

    # Filter dataset for that act
    act_df = df[df["act_name"].str.lower() == act.lower()].reset_index(drop=True)
    if act_df.empty:
        print("⚠️ No matching sections found for this Act.")
        return []

    # Step 2: BM25 retrieval
    bm25_docs = [d.split() for d in act_df["content_text"].tolist()]
    bm25_local = BM25Okapi(bm25_docs)
    bm25_scores = bm25_local.get_scores(query.split())
    top_bm25_idx = np.argsort(bm25_scores)[-top_k:][::-1]
    bm25_results = act_df.iloc[top_bm25_idx]

    # Step 3: Chroma vector similarity
    q_vec = embedding_model.embed_query(query)
    chroma_results = collection.query(query_embeddings=[q_vec], n_results=top_k)
    chroma_docs = chroma_results["documents"][0]

    # Combine BM25 + Chroma results and remove duplicates
    combined_texts = list(bm25_results["content_text"].values) + chroma_docs
    combined_texts = list(dict.fromkeys(combined_texts))

    return combined_texts[:top_k]

# ========== STEP 6: Test Queries ==========
if __name__ == "__main__":
    queries = [
        "punishment for murder",
        "procedure for arrest",
        "terrorism investigation powers",
        "criminal appeal process",
        "theft and its punishment"
    ]

    for q in queries:
        print(f"\n🔍 Query: {q}")
        results = hybrid_search(q)
        for i, res in enumerate(results, 1):
            print(f"{i}. {res[:250]}...\n")
