from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# Rule-based keywords
act_keywords = {
    "IPC": [        
        "murder", "homicide", "culpable homicide", "rape", "sexual assault", "assault",
        "attempt to murder", "kidnapping", "abduction",        
        "theft", "robbery", "dacoity", "extortion", "cheating", "criminal breach of trust",
        "misappropriation", "forgery", "receiving stolen property",        # offences against 
        "sedition", "terrorism", "unlawful assembly", "rioting", "public mischief",        
        "defamation", "insult", "criminal intimidation", "annoyance",        # special 
        "bribery", "counterfeiting", "poisoning", "dangerous weapons", "abetment", "conspiracy",
        "non-cognizable", "cognizable", "bailable", "non-bailable", "compoundable"
    ],
    "CrPC": [
        "arrest", "detention", "remand", "police custody", "judicial custody",
        "charge sheet", "FIR", "investigation", "summons", "warrant", "magistrate",
        "trial", "session court", "appeal", "revision", "bail", "anticipatory bail",
        "interim bail", "release on bail", "cognizable offence", "non-cognizable offence",
        "bailable offence", "non-bailable offence", "compoundable offence", "non-compoundable offence",
        "examination of witness", "cross-examination", "plea bargaining", "acquittal", "conviction"
    ],
    "NIA": [
        "terrorism", "scheduled offences", "national investigation agency", "NIA",
        "terrorist act", "terrorist organisation", "terror funding", "UAPA", "investigation of terrorism",
        "firearms act", "arms trafficking", "bomb attack", "insurgency", "counter-terror operations",
        "inter-state terrorism", "transnational terrorism", "cyber terrorism", "major offence", "national security"
    ]
}

# Summaries for embedding classifier
act_summaries = {
    "IPC": "Defines offences and their punishments like murder, theft, rape, etc.",
    "CrPC": "Defines criminal procedures like investigation, arrest, and bail.",
    "NIA": "Defines powers of the National Investigation Agency to investigate terrorism offences."
}

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
act_vecs = {act: embedding_model.embed_query(text) for act, text in act_summaries.items()}

# Rule-based scoring
def rule_score(query):
    q = query.lower()
    return {act: sum(k in q for k in kws) for act, kws in act_keywords.items()}

# Embedding-based scoring
def embed_score(query):
    q_vec = embedding_model.embed_query(query)
    return {act: np.dot(q_vec, vec) for act, vec in act_vecs.items()}

# Hybrid classifier
def classify_act(query, alpha=0.4):
    rule = rule_score(query)
    embed = embed_score(query)

    # Normalize
    def norm(d):
        vals = np.array(list(d.values()), dtype=float)
        if vals.sum() == 0: vals += 1e-9
        return {k: v / vals.sum() for k, v in d.items()}

    rule_n = norm(rule)
    embed_n = norm(embed)
    combined = {act: alpha*rule_n.get(act,0) + (1-alpha)*embed_n.get(act,0) for act in set(rule_n)|set(embed_n)}
    best = max(combined, key=combined.get)
    confidence = combined[best]
    return best, confidence

# Example
# test_queries = [
#     "punishment for murder",
#     "procedure for arrest",
#     "terrorism investigation process",
#     "criminal appeal procedure",
#     "bail process in india"
# ]

# for q in test_queries:
#     act, conf = classify_act(q)
#     print(f"Query: {q}")
#     print(f"→ Predicted Act: {act} (Confidence: {conf})\n")
