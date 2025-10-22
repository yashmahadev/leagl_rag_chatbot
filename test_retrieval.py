# test_retrieval_fixed.py
import os
import sys
import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from rank_bm25 import BM25Okapi
from pprint import pprint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_legal_db")
COLLECTION_NAME = "legal_acts"
DATA_DIR = os.path.join(BASE_DIR, "jsons")   # contains ipc.json, crpc.json, nia.json

# --- Helper: load merged dataframe used for BM25 fallback ---
def load_merged_dataframe():
    expected = [("ipc.json","IPC"), ("crpc.json","CrPC"), ("nia.json","NIA")]
    frames = []
    for fname, act_name in expected:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist. Continuing without it.")
            continue
        df = pd.read_json(path)
        df["act_name"] = act_name
        # create content_text consistent with ingestion code
        df["content_text"] = df.apply(lambda r: " ".join([
            str(r.get("chapter","")).strip(),
            str(r.get("section","")).strip(),
            str(r.get("section_title","")).strip(),
            str(r.get("section_desc","")).strip()
        ]).strip(), axis=1)
        frames.append(df[["act_name","chapter","section","section_title","section_desc","content_text"]])
    if not frames:
        raise FileNotFoundError("No JSON files found in ./jsons. Make sure ipc.json/crpc.json/nia.json exist.")
    merged = pd.concat(frames, ignore_index=True)
    merged["content_text"] = merged["content_text"].fillna("").astype(str)
    return merged

# --- Step A: Connect to Chroma and inspect collection ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
collections = [c.name for c in client.list_collections()]
print("Chroma collections found:", collections)
if COLLECTION_NAME not in collections:
    print(f"Error: collection '{COLLECTION_NAME}' not found in ChromaDB at {CHROMA_PATH}.")
    print("Make sure you ran the embedding script that created this collection.")
    # still attempt BM25 fallback
else:
    collection = client.get_collection(COLLECTION_NAME)
    try:
        count = collection.count()
    except Exception:
        # some chroma versions expose .count() differently; try fetching some metadata
        results = collection.get(include=["metadatas"])
        count = len(results.get("metadatas", []))
    print(f"✅ Documents in '{COLLECTION_NAME}': {count}")

    # Show 3 sample docs + metadata
    try:
        got = collection.get(ids=[str(i) for i in range(min(5, count))], include=["documents","metadatas"])
        docs = got.get("documents", [])
        metas = got.get("metadatas", [])
        print("\nSample documents and metadata (if any):")
        for i, (d, m) in enumerate(zip(docs, metas)):
            print(f"\n--- SAMPLE {i+1} ---")
            print("doc (first 240 chars):", (d[:240] if d else "<empty>"))
            print("metadata:", m)
    except Exception:
        # fallback: try a small query to get any documents
        try:
            r = collection.query(query_texts=["test"], n_results=1)
            print("sample query result exists")
        except Exception as e:
            print("Could not read sample docs:", e)

# --- Step B: Prepare embedding model (must match the one used to build embeddings) ---
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("\n✅ Embedding model ready.")

# --- Step C: Load dataframe for BM25 fallback and build index ---
df = load_merged_dataframe()
print(f"✅ Loaded merged dataframe with {len(df)} rows for BM25 fallback.")
bm25_corpus = [doc.split() for doc in df["content_text"].tolist()]
bm25 = BM25Okapi(bm25_corpus)
print("✅ BM25 index built.")

# --- Test queries to evaluate retrieval ---
queries = [
    "What is the punishment for murder under IPC?",
    "What is the procedure for arrest under CrPC?",
    "What are the powers of NIA under NIA Act?",
    "What is Section 420 IPC?",
    "Explain the concept of bail in CrPC."
]

print("\n\n=== Retrieval Tests (semantic via Chroma + BM25 fallback) ===\n")
for q in queries:
    print(f"\n🔍 Query: {q}\n")

    # Semantic search via Chroma (if collection exists)
    if COLLECTION_NAME in collections:
        q_vec = embed_model.embed_query(q)
        try:
            sem = collection.query(query_embeddings=[q_vec], n_results=5)
            sem_docs = sem.get("documents", [[]])[0]
            sem_meta = sem.get("metadatas", [[]])[0]
            sem_dist = sem.get("distances", [[]])[0]  # distance (depends on chroma config)
        except Exception as e:
            print("Semantic query error:", e)
            sem_docs, sem_meta, sem_dist = [], [], []

        if sem_docs:
            print("Top semantic (Chroma) results:")
            for i, (doc, meta, dist) in enumerate(zip(sem_docs, sem_meta, sem_dist), 1):
                # show similarity-ish measure: if it's a distance, smaller is closer; show both
                print(f"{i}. meta: {meta} | distance: {dist}")
                print("   ->", doc.replace("\n"," ") + "...")
        else:
            print("No semantic results from Chroma.")

    # BM25 fallback (show top 5)
    print("\nTop BM25 results (fallback):")
    bm25_scores = bm25.get_scores(q.split())
    idxs = np.argsort(bm25_scores)[-5:][::-1]
    for rank, idx in enumerate(idxs, start=1):
        score = bm25_scores[idx]
        row = df.iloc[idx]
        snippet = row["content_text"].replace("\n"," ")
        print(f"{rank}. [{row['act_name']} | section: {row['section']}] score={score:.3f}")
        print("   ->", snippet + "...")
    print("\n" + ("-"*90))

# --- Diagnostics & tips ---
print("\n\nDiagnostics complete. If semantic results are missing or irrelevant:")
print("- Verify that the Chroma collection 'legal_acts' was created by your embedding script.")
print("- Ensure the embedding model used for queries matches the one used to build the index (BAAI/bge-large-en-v1.5).")
print("- Ensure 'content_text' actually contains the section text + title (no empty strings).")
print("- If collection is missing, re-run your embedding script to populate ChromaDB.")
