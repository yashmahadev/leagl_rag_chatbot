from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import chromadb
from tqdm import tqdm

# Load preprocessed data
df = pd.read_csv("legal_data_cleaned.csv")

# Initialize embeddings model (modern import)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Initialize ChromaDB (new API)
client = chromadb.PersistentClient(path="./chroma_legal_db")
collection = client.get_or_create_collection(name="legal_sections")

# Insert documents into Chroma
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding sections"):
    content_text = f"[{row['act']}] {row['title']}: {row['content']}"
    embedding = embedding_model.embed_query(content_text)

    collection.add(
        ids=[f"{row['act']}_{row['section_no']}"],
        metadatas=[{
            "act": row["act"],
            "chapter": row["chapter"],
            "section_no": row["section_no"],
            "title": row["title"]
        }],
        documents=[content_text],
        embeddings=[embedding]
    )

print("✅ Embeddings created and stored in ChromaDB successfully!")
