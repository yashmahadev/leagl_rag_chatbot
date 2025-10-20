from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from chromadb import Client
from chromadb.config import Settings
import sys

# Load preprocessed data
df = pd.read_csv("legal_data_cleaned.csv")

# Initialize embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Initialize ChromaDB
client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_legal_db"
))
collection = client.get_or_create_collection(name="legal_sections")

# Insert documents into Chroma
for idx, row in df.iterrows():
    content_text = f"[{row['act']}] {row['title']}: {row['content']}"
    collection.add(
        ids=[f"{row['act']}_{row['section_no']}"],
        metadatas=[{
            "act": row["act"],
            "chapter": row["chapter"],
            "section_no": row["section_no"],
            "title": row["title"]
        }],
        documents=[content_text],
        embeddings=[embedding_model.embed_query(content_text)]
    )

# Persist database
client.persist()
print("Embeddings created and stored in ChromaDB successfully!")
