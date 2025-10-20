from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Define the available Acts
ACTS = {
    "Indian Penal Code": ["murder", "theft", "punishment", "offence", "assault", "kidnapping", "cheating", "criminal"],
    "Criminal Procedure Code": ["procedure", "arrest", "trial", "bail", "court", "appeal", "investigation", "warrant"],
    "National Investigation Agency Act": ["terrorism", "nia", "agency", "investigate", "uapa", "national security"]
}

# Initialize embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Precompute embeddings for each Act's keywords
ACT_EMBEDDINGS = {}
for act_name, keywords in ACTS.items():
    ACT_EMBEDDINGS[act_name] = [
        embedding_model.embed_query(kw) for kw in keywords
    ]

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def classify_act(query: str):
    """
    Predict which Act the query belongs to based on embedding similarity and rule-based hints
    Returns: (act_name, confidence)
    """
    query_embedding = embedding_model.embed_query(query.lower())

    scores = {}

    # Hybrid approach: rule-based + semantic
    for act_name, kw_embeddings in ACT_EMBEDDINGS.items():
        # Semantic score
        sem_score = max([cosine_similarity(query_embedding, emb) for emb in kw_embeddings])
        # Rule-based keyword boost
        rule_boost = sum([1 for kw in ACTS[act_name] if kw in query.lower()]) * 0.05
        scores[act_name] = sem_score + rule_boost

    # Pick the highest-scoring act
    best_act = max(scores, key=scores.get)
    confidence = scores[best_act]

    # Normalize confidence between 0–1
    confidence = min(1.0, round(confidence, 2))

    return best_act, confidence

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "punishment for murder",
        "procedure for arrest",
        "terrorism investigation process",
        "criminal appeal procedure",
        "bail process in india"
    ]

    for q in test_queries:
        act, conf = classify_act(q)
        print(f"Query: {q}")
        print(f"→ Predicted Act: {act} (Confidence: {conf})\n")
