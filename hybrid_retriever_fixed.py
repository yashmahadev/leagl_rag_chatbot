# hybrid_retriever_fixed.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from act_classifier import classify_act

# ----------------- Config -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "jsons")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_legal_db")
COLLECTION_NAME = "legal_acts"

# How many candidates to produce before reranking
HYBRID_CANDIDATES = 50
TOP_K = 5

# Cross-encoder model for pairwise reranking (small & strong)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # public and fast

# ----------------- Utilities -----------------
section_pattern = re.compile(r"(?:section|sec|s\.?)\s*\.?\s*(\d+)", flags=re.I)

def load_and_merge_jsons():
    files = [("ipc.json","IPC"), ("crpc.json","CrPC"), ("nia.json","NIA")]
    frames = []
    for fname, act in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found — skipping.")
            continue
        df = pd.read_json(path)
        df["act_name"] = act
        # unify fields
        df["chapter"] = df.get("chapter", "")
        df["section"] = df.get("section", df.get("section_no", ""))
        # some JSONs use title/description keys differently; try common names
        df["section_title"] = df.get("section_title", df.get("title", ""))
        df["section_desc"] = df.get("section_desc", df.get("description", df.get("content", "")))
        df["content_text"] = df.apply(lambda r: f"{r['act_name']} | Section {r.get('section','')}: {r.get('section_title','')}. {r.get('section_desc','')}", axis=1)
        frames.append(df[["act_name","chapter","section","section_title","section_desc","content_text"]])
    if not frames:
        raise FileNotFoundError("No JSON files loaded from ./jsons. Ensure ipc.json, crpc.json, nia.json exist.")
    merged = pd.concat(frames, ignore_index=True)
    merged["content_text"] = merged["content_text"].fillna("").astype(str)
    return merged

def build_section_lookup(df):
    # Map (act_name.lower(), section_number_str) -> row index
    lookup = {}
    for idx, row in df.iterrows():
        act = str(row["act_name"]).strip().lower()
        section = str(row["section"]).strip()
        if section:
            lookup.setdefault((act, section), []).append(idx)
    return lookup

# ----------------- Load data, BM25, Chroma, embeddings, cross-encoder -----------------
print("Loading datasets...")
df = load_and_merge_jsons()
print(f"Loaded {len(df)} sections.")

print("Preparing BM25 index...")
bm25_tokenized = [text.split() for text in df["content_text"].tolist()]
bm25 = BM25Okapi(bm25_tokenized)

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collections = [c.name for c in client.list_collections()]
if COLLECTION_NAME not in collections:
    raise RuntimeError(f"Chroma collection '{COLLECTION_NAME}' not found in {CHROMA_PATH}. Re-run embeddings build.")
collection = client.get_collection(COLLECTION_NAME)

print("Initializing embeddings (for query vectors)...")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

print("Loading cross-encoder for reranking...")
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

# Build exact lookup for sections
section_lookup = build_section_lookup(df)

# ----------------- Core retrieval functions -----------------
def exact_section_lookup(query):
    """
    If query mentions "Section <num>" and possibly an act (ipc/crpc/nia),
    return exact matches from the dataset.
    """
    m = section_pattern.search(query)
    if not m:
        return None  # no section number in query
    section_no = m.group(1).lstrip('0')  # "321" as string
    # Try to detect act in query
    lower_q = query.lower()
    detected_act = None
    for act in ["ipc", "crpc", "nia"]:
        if act in lower_q:
            detected_act = act.upper() if act!="nia" else "NIA"
            break
    results_idx = []
    if detected_act:
        key = (detected_act.lower(), section_no)
        results_idx = section_lookup.get(key, [])
    else:
        # search across acts
        for (act_k, sec), idxs in section_lookup.items():
            if sec == section_no:
                results_idx.extend(idxs)
    if not results_idx:
        return None
    # Return list of dicts with metadata & text
    out = []
    for i in results_idx:
        row = df.iloc[i]
        out.append({
            "act": row["act_name"],
            "section": str(row["section"]),
            "title": row["section_title"],
            "text": row["content_text"]
        })
    return out

def hybrid_candidates(query, act_filter=None, top_k=HYBRID_CANDIDATES):
    """
    Return a candidate list from BM25 + Chroma (semantic). We'll re-rank them later with cross-encoder.
    """
    candidates = []
    # BM25 (apply act_filter)
    if act_filter:
        mask = df["act_name"].str.lower() == act_filter.lower()
        local_texts = df[mask]["content_text"].tolist()
        if local_texts:
            local_tokens = [t.split() for t in local_texts]
            bm25_local = BM25Okapi(local_tokens)
            bm25_scores = bm25_local.get_scores(query.split())
            # top indices relative to local_texts; map back to global df indices
            top_local_idx = np.argsort(bm25_scores)[-top_k:][::-1]
            rows = df[mask].reset_index()
            for idx in top_local_idx:
                global_idx = int(rows.loc[idx, "index"])
                candidates.append((global_idx, float(bm25_scores[idx])))
    else:
        bm25_scores = bm25.get_scores(query.split())
        top_idx = np.argsort(bm25_scores)[-top_k:][::-1]
        for idx in top_idx:
            candidates.append((int(idx), float(bm25_scores[idx])))

    # Semantic (Chroma)
    try:
        q_vec = embed_model.embed_query(query)
        chroma_res = collection.query(query_embeddings=[q_vec], n_results=top_k)
        docs = chroma_res.get("documents", [[]])[0]
        metas = chroma_res.get("metadatas", [[]])[0]
        # Chroma returns documents only; we need to map them to df indices by comparing text or metadata
        for doc, meta, dist in zip(docs, metas, chroma_res.get("distances", [[]])[0]):
            # Try to match by act + section if in metadata
            act_meta = meta.get("act_name") if isinstance(meta, dict) else None
            sec_meta = meta.get("section") if isinstance(meta, dict) else None
            found_idx = None
            if act_meta and sec_meta:
                # find first matching index
                matches = df[(df["act_name"].str.lower() == str(act_meta).strip().lower()) & (df["section"].astype(str).str.strip() == str(sec_meta).strip())]
                if not matches.empty:
                    found_idx = int(matches.index[0])
            if found_idx is None:
                # fallback: find first row whose content_text starts with doc[:100] or contains doc snippet
                possible = df[df["content_text"].str.contains(doc[:80], na=False, regex=False)]
                if not possible.empty:
                    found_idx = int(possible.index[0])
            if found_idx is not None:
                candidates.append((found_idx, float(1.0 - float(dist))))  # convert distance -> similarity-ish
    except Exception as e:
        print("Chroma semantic search failed:", e)

    # combine candidates, keep unique highest score
    merged = {}
    for idx, score in candidates:
        if idx in merged:
            merged[idx] = max(merged[idx], score)
        else:
            merged[idx] = score
    # produce list sorted by preliminary score
    sorted_idx = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    # return top N indices
    return [ (int(i), float(s)) for i,s in sorted_idx[:top_k] ]

def cross_rank(query, candidate_indices, top_k=TOP_K):
    """
    Use cross-encoder to rank query-document pairs.
    candidate_indices: list of (df_index, prelim_score)
    returns list of (df_index, cross_score) sorted desc
    """
    if not candidate_indices:
        return []
    pairs = []
    idx_map = []
    for idx, _ in candidate_indices:
        doc_text = df.loc[idx, "content_text"]
        pairs.append((query, doc_text))
        idx_map.append(idx)
    # Cross-encoder expects list of [query, doc] pairs or list of strings depending on model; for ms-marco model we pass pairs
    scores = cross_encoder.predict(pairs)  # returns float score array
    scored = list(zip(idx_map, scores.tolist()))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return scored_sorted[:top_k]

# ----------------- Public function to call -----------------
def retrieve(query, top_k=TOP_K):
    # 1) Exact section lookup
    exact = exact_section_lookup(query)
    if exact:
        # return exact lookup as top results with artificially high score
        return [{"act": e["act"], "section": e["section"], "title": e["title"], "text": e["text"], "method":"exact", "score": 999.0} for e in exact][:top_k]

    # 2) Determine act via classifier for filtering (but allow fallback)
    act_pred, conf = classify_act(query)
    # if classifier confidence low, set act_filter to None (search all)
    act_filter = act_pred if conf >= 0.5 else None

    # 3) Hybrid candidate pool
    candidates = hybrid_candidates(query, act_filter=act_filter, top_k=HYBRID_CANDIDATES)
    if not candidates:
        return []

    # 4) Cross-encoder re-rank top candidates
    reranked = cross_rank(query, candidates, top_k=top_k)
    results = []
    for idx, score in reranked:
        row = df.loc[idx]
        results.append({
            "act": row["act_name"],
            "section": str(row["section"]),
            "title": row["section_title"],
            "text": row["content_text"],
            "method": "hybrid+cross",
            "score": float(score)
        })
    return results

# ----------------- Quick test (run as script) -----------------
if __name__ == "__main__":
    tests = [
        "Section 321 in ipc",
        "Section 321 in crpc",
        "punishment for murder",
        "procedure for arrest",
        "powers of NIA to investigate",
        "what is section 420 ipc",
        "bail in crpc"
    ]
    for q in tests:
        print("\n" + "="*80)
        print("Query:", q)
        res = retrieve(q, top_k=5)
        if not res:
            print("No results.")
            continue
        for i, r in enumerate(res, 1):
            print(f"{i}. ({r['method']}) [{r['act']} | Section {r['section']}] score={r['score']:.4f}")
            print("   Title:", r['title'])
            print("   Snippet:", r['text'][:300].replace("\n"," "), "...\n")
