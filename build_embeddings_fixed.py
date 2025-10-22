# build_embeddings_fixed.py
import os
import pandas as pd
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "jsons")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_legal_db")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection_name = "legal_acts"

try:
    client.delete_collection(collection_name)
except Exception:
    pass

collection = client.create_collection(name=collection_name)

def load_dataset(filename, act_name):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_json(path)
    df["act_name"] = act_name
    df["content_text"] = df.apply(
        lambda r: f"Act: {act_name}. Section {r.get('section','')}: {r.get('section_title','')}. Description: {r.get('section_desc','')}".strip(),
        axis=1
    )
    return df[["act_name", "section", "section_title", "section_desc", "content_text"]]

# Load all Acts
dfs = []
for fn, name in [("ipc.json","IPC"),("crpc.json","CrPC"),("nia.json","NIA")]:
    if os.path.exists(os.path.join(DATA_DIR, fn)):
        dfs.append(load_dataset(fn, name))
df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(df)} total sections.")

# Add embeddings
for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding legal sections"):
    text = row["content_text"]
    if not text.strip(): continue
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

print("✅ All embeddings stored successfully.")
