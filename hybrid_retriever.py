from chromadb import Client
from chromadb.config import Settings
from whoosh import index, fields
import os

# Load ChromaDB
client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_legal_db"
))
collection = client.get_collection("legal_sections")

# Simple retrieval function
def retrieve_context(query, act=None, top_k=5):
    if act:
        filter_meta = {"act": act}
    else:
        filter_meta = {}
    docs = collection.query(query_texts=[query], n_results=top_k, where=filter_meta)
    return [doc for doc in docs['documents'][0]]

# Test
top_docs = retrieve_context("punishment of murder", act="IPC")
for i, doc in enumerate(top_docs):
    print(f"Top-{i+1}: {doc[:200]}...")
