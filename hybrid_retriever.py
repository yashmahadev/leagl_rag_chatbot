# hybrid_retriever.py (optimized)
"""
Optimized hybrid retriever:
 - Exact alphanumeric section lookup (e.g., 52A, 120B)
 - BM25 lexical retrieval (fast)
 - Chroma semantic retrieval (vector)
 - Cross-encoder reranking of combined candidates
 - Optional act filtering via classify_act
"""

import os
import re
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from act_classifier import classify_act

# ----------------------------- CONFIG -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "jsons")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_legal_db")
COLLECTION_NAME = "legal_acts"

HYBRID_CANDIDATES = 50
TOP_K = 5
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ----------------------------- HELPERS -----------------------------
# Matches "section 52", "sec. 52A", "u/s 120B", "s. 52-A" etc.
SECTION_RE = re.compile(r"(?:section|sec|s\.|u/s)\s*[-.]?\s*([0-9]+(?:[A-Za-z\-]?[0-9A-Za-z]*)?)", flags=re.I)

def normalize_section_token(token: str) -> str:
    """Normalize section token to canonical form, e.g., '52-a' -> '52A'."""
    s = str(token).strip().upper()
    s = s.replace("-", "")  # '52-A' -> '52A'
    s = s.replace(" ", "")
    return s

# ----------------------------- LOAD & INDEX DATA -----------------------------
def load_and_prepare_df() -> pd.DataFrame:
    """Load ipc/crpc/nia JSONs and build unified dataframe."""
    files = [("ipc.json", "IPC"), ("crpc.json", "CrPC"), ("nia.json", "NIA")]
    frames = []
    for fname, act in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_json(path, dtype={"section": str})
        df["act_name"] = act
        df["chapter"] = df.get("chapter", "")
        df["section"] = df.get("section", df.get("section_no", "")).astype(str)
        df["section_title"] = df.get("section_title", df.get("title", "")).astype(str)
        df["section_desc"] = df.get("section_desc", df.get("description", df.get("content", ""))).astype(str)
        df["content_text"] = df.apply(
            lambda r: f"{r['act_name']} | Section {r.get('section','')}: {r.get('section_title','')}. {r.get('section_desc','')}",
            axis=1
        )
        frames.append(df[["act_name", "chapter", "section", "section_title", "section_desc", "content_text"]])

    if not frames:
        raise FileNotFoundError("No act JSON files found in ./jsons. Place ipc.json, crpc.json, nia.json")

    merged = pd.concat(frames, ignore_index=True)
    merged["content_text"] = merged["content_text"].fillna("").astype(str)
    # Normalize section strings (keep original in df["section"] for output)
    merged["_section_norm"] = merged["section"].apply(lambda x: normalize_section_token(x) if pd.notna(x) else "")
    return merged

# Load dataframe once (module import)
DF = load_and_prepare_df()
N_DOCS = len(DF)

# Build exact lookup: (act_lower, section_norm) -> [indices]
def build_section_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], List[int]]:
    lookup = {}
    for idx, row in df.iterrows():
        act = str(row["act_name"]).strip().lower()
        secnorm = str(row["_section_norm"]).strip()
        if secnorm:
            lookup.setdefault((act, secnorm), []).append(int(idx))
    return lookup

SECTION_LOOKUP = build_section_lookup(DF)

# ----------------------------- BM25 INDEX -----------------------------
BM25_TOKENIZED = [text.split() for text in DF["content_text"].tolist()]
BM25 = BM25Okapi(BM25_TOKENIZED) if BM25_TOKENIZED else None

# ----------------------------- CHROMA (VECTOR) -----------------------------
# Chroma client + collection
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_existing_collections = [c.name for c in _chroma_client.list_collections()]
if COLLECTION_NAME not in _existing_collections:
    raise RuntimeError(f"Chroma collection '{COLLECTION_NAME}' not found at {CHROMA_PATH}. Run build_embeddings.py first.")
CHROMA_COLLECTION = _chroma_client.get_collection(COLLECTION_NAME)

# ----------------------------- EMBEDDING (query vectors) -----------------------------
EMBED_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# ----------------------------- CROSS ENCODER -----------------------------
CROSS_ENCODER = CrossEncoder(CROSS_ENCODER_MODEL)

# ----------------------------- RETRIEVAL PRIMITIVES -----------------------------
def exact_section_lookup(query: str) -> Optional[List[Dict]]:
    """Return exact matches if query contains a section token like '52A' and optional act mention."""
    m = SECTION_RE.search(query)
    if not m:
        return None
    raw = m.group(1)
    section_norm = normalize_section_token(raw)
    lower_q = query.lower()

    detected_act = None
    for act in ("ipc", "crpc", "nia"):
        if act in lower_q:
            detected_act = act
            break

    results_idx = []
    if detected_act:
        key = (detected_act.lower(), section_norm)
        results_idx = SECTION_LOOKUP.get(key, [])
    else:
        # search across all acts for this section
        for (act_k, sec), idxs in SECTION_LOOKUP.items():
            if sec == section_norm:
                results_idx.extend(idxs)

    if not results_idx:
        return None

    out = []
    for i in results_idx:
        row = DF.iloc[i]
        out.append({
            "act": row["act_name"],
            "section": str(row["section"]),
            "title": row["section_title"],
            "text": row["content_text"],
            "index": int(i)
        })
    return out

def bm25_retrieve(query: str, top_k: int) -> List[Tuple[int, float]]:
    """Return list of (df_index, score) from BM25."""
    if BM25 is None:
        return []
    tokens = query.split()
    scores = BM25.get_scores(tokens)
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [(int(i), float(scores[i])) for i in top_idx]

def chroma_retrieve(query: str, top_k: int) -> List[Tuple[int, float]]:
    """Return list of (df_index, similarity_score) from Chroma using metadata mapping where possible."""
    try:
        q_vec = EMBED_MODEL.embed_query(query)
        res = CHROMA_COLLECTION.query(query_embeddings=[q_vec], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]  # distance measure (smaller = closer)
    except Exception:
        return []

    results = []
    for doc, meta, dist in zip(docs, metas, dists):
        found_idx = None
        if isinstance(meta, dict):
            act_meta = meta.get("act_name")
            sec_meta = meta.get("section")
            if act_meta and sec_meta is not None:
                sec_norm = normalize_section_token(sec_meta)
                matches = DF[(DF["act_name"].str.lower() == str(act_meta).strip().lower()) & (DF["_section_norm"] == sec_norm)]
                if not matches.empty:
                    found_idx = int(matches.index[0])

        if found_idx is None:
            # best-effort match via substring of doc (avoid regex for speed)
            snippet = doc[:120]
            possible = DF[DF["content_text"].str.contains(snippet, na=False, regex=False)]
            if not possible.empty:
                found_idx = int(possible.index[0])

        if found_idx is not None:
            # Convert distance -> similarity (larger is better). We do 1-dist if dist in [0,1], else invert.
            try:
                sim = 1.0 - float(dist)
            except Exception:
                sim = 1.0 / (1.0 + float(dist)) if float(dist) != 0 else 1.0
            results.append((found_idx, float(sim)))
    return results

# ----------------------------- HYBRID CANDIDATES & MERGE -----------------------------
def hybrid_candidates(query: str, act_filter: Optional[str], top_k: int = HYBRID_CANDIDATES) -> List[Tuple[int, float]]:
    """
    Combine BM25 and Chroma candidates, apply act filter if provided,
    and return up to top_k unique candidates (index, combined_score).
    """
    candidates: Dict[int, float] = {}

    # BM25 candidates
    try:
        if act_filter:
            mask = DF["act_name"].str.lower() == act_filter.lower()
            local_texts = DF[mask]["content_text"].tolist()
            if local_texts:
                local_tokens = [t.split() for t in local_texts]
                bm25_local = BM25Okapi(local_tokens)
                local_scores = bm25_local.get_scores(query.split())
                top_local = np.argsort(local_scores)[-top_k:][::-1]
                rows = DF[mask].reset_index()
                for idx in top_local:
                    global_idx = int(rows.loc[int(idx), "index"])
                    candidates[global_idx] = max(candidates.get(global_idx, 0.0), float(local_scores[int(idx)]))
        else:
            for idx, score in bm25_retrieve(query, top_k):
                candidates[idx] = max(candidates.get(idx, 0.0), float(score))
    except Exception:
        # BM25 failure shouldn't break system
        pass

    # Chroma candidates
    try:
        for idx, sim in chroma_retrieve(query, top_k):
            candidates[idx] = max(candidates.get(idx, 0.0), float(sim))
    except Exception:
        pass

    # Normalize preliminary scores (min-max) for fusion fairness
    if not candidates:
        return []
    idxs, vals = zip(*candidates.items())
    arr = np.array(vals, dtype=float)
    if math.isfinite(arr.max()) and arr.ptp() > 0:
        arr_norm = (arr - arr.min()) / (arr.ptp() + 1e-9)
    else:
        arr_norm = np.ones_like(arr)

    merged = list(zip(idxs, arr_norm))
    merged_sorted = sorted(merged, key=lambda x: x[1], reverse=True)
    return merged_sorted[:top_k]

# ----------------------------- CROSS RANK (BATCHED) -----------------------------
def cross_rank(query: str, candidate_indices: List[Tuple[int, float]], top_k: int = TOP_K) -> List[Tuple[int, float]]:
    """
    Use cross-encoder to re-rank candidates. Returns list of (index, score) sorted desc.
    """
    if not candidate_indices:
        return []

    texts = [DF.loc[idx, "content_text"] for idx, _ in candidate_indices]
    pairs = [(query, t) for t in texts]
    # batch predict (CrossEncoder handles batching internally)
    scores = CROSS_ENCODER.predict(pairs, show_progress_bar=False)
    scored = [(candidate_indices[i][0], float(scores[i])) for i in range(len(scores))]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return scored_sorted[:top_k]

# ----------------------------- PUBLIC RETRIEVE -----------------------------
def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Main retrieval function used by the app.
    1) Try exact section lookup (fast) -> return if found
    2) Use act classifier to optional filter
    3) Build hybrid candidate pool (BM25 + Chroma)
    4) Cross-encoder re-rank and return top_k results
    """
    if not query or not str(query).strip():
        return []

    # 1) exact lookup
    exact = exact_section_lookup(query)
    if exact:
        return [{"act": e["act"], "section": e["section"], "title": e["title"], "text": e["text"], "method": "exact", "score": 999.0} for e in exact][:top_k]

    # 2) act classification (may be None)
    act_pred, conf = classify_act(query)
    act_filter = act_pred if conf >= 0.5 else None

    # 3) hybrid candidates
    candidates = hybrid_candidates(query, act_filter=act_filter, top_k=HYBRID_CANDIDATES)
    if not candidates:
        return []

    # 4) cross-encoder rerank
    reranked = cross_rank(query, candidates, top_k=top_k)
    results = []
    for idx, score in reranked:
        row = DF.loc[idx]
        results.append({
            "act": row["act_name"],
            "section": str(row["section"]),
            "title": row["section_title"],
            "text": row["content_text"],
            "method": "hybrid+cross",
            "score": float(score)
        })
    return results

# ----------------------------- SCRIPT TEST -----------------------------
# if __name__ == "__main__":
#     tests = [
#         "Section 52A in IPC",
#         "Section 52 in ipc",
#         "Section 321 in ipc",
#         "punishment for murder",
#         "procedure for arrest",
#         "bail in crpc"
#     ]
#     for q in tests:
#         print("\n" + "=" * 70)
#         print("Query:", q)
#         res = retrieve(q, top_k=5)
#         if not res:
#             print("No results.")
#             continue
#         for i, r in enumerate(res, 1):
#             print(f"{i}. ({r['method']}) [{r['act']} | Section {r['section']}] score={r['score']:.4f}")
#             print("   Title:", r['title'])
#             print("   Snippet:", r['text'][:300].replace("\n", " "), "...\n")
