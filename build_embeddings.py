"""
build_embeddings.py
-------------------
Builds and stores semantic embeddings for all legal acts (IPC, CrPC, NIA)
using ChromaDB + HuggingFace embeddings for fast hybrid retrieval.

Run once before using the chatbot:
    python build_embeddings.py
"""

import os
import pandas as pd
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "jsons")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_legal_db")
COLLECTION_NAME = "legal_acts"

ACT_FILES = [
    ("ipc.json", "IPC"),
    ("crpc.json", "CrPC"),
    ("nia.json", "NIA"),
]

# ------------------------------------------------
# INITIALIZE MODELS & CLIENT
# ------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

# Drop existing collection if exists (safe reset)
if any(c.name == COLLECTION_NAME for c in client.list_collections()):
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(name=COLLECTION_NAME)

# ------------------------------------------------
# DATA LOADER
# ------------------------------------------------
def load_dataset(filename: str, act_name: str) -> pd.DataFrame:
    """Load a single act JSON and prepare text for embedding."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {filename}")

    df = pd.read_json(path, dtype={"section": str})
    df["act_name"] = act_name
    df["content_text"] = (
        "Act: " + act_name + ". Section " +
        df["section"].astype(str) + ": " +
        df["section_title"].astype(str) + ". " +
        df["section_desc"].astype(str)
    ).str.strip()
    return df[["act_name", "section", "section_title", "section_desc", "content_text"]]

# ------------------------------------------------
# MAIN BUILD LOGIC
# ------------------------------------------------
def build_embeddings():
    all_dfs = []
    for file_name, act_name in ACT_FILES:
        try:
            all_dfs.append(load_dataset(file_name, act_name))
        except FileNotFoundError as e:
            print(f"⚠️ Skipping missing file: {file_name}")
            continue

    if not all_dfs:
        raise RuntimeError("❌ No valid legal act JSON files found in ./jsons directory.")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Loaded {len(df)} total sections across acts.")

    # Embed & store
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding legal sections"):
        text = row["content_text"]
        if not text.strip():
            continue

        emb = embedding_model.embed_documents([text])[0]
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

    print("✅ All embeddings stored successfully in ChromaDB.")
    print(f"📂 Path: {CHROMA_PATH}")

# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------
if __name__ == "__main__":
    build_embeddings()
