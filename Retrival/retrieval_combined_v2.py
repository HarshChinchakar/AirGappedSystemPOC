#!/usr/bin/env python3
"""
Hybrid retrieval: Semantic (FAISS) + Keyword (BM25) with metadata filtering and optional rerank.

This is the UPDATED retrieval script that keeps the original scoring / fusion logic
but **also** returns structured JSON exactly in the schema expected by the Streamlit app:

{
  "query": "...",
  "retrieved_at": "...",
  "semantic": { "results": [ { chunk dict }, ... ] },
  "keyword":  { "results": [ { chunk dict }, ... ] },
  "final_scores": { "chunk_id": score, ... }
}

Each chunk dict contains at minimum:
 - chunk_id
 - pdf_name
 - content
 - page
 - section_type
 - score

It still prints the human-readable console output (as before) and then prints the JSON blob
to stdout at the end so external callers (Streamlit) can parse it.

Usage:
 python3 retrieval_semantic_keyword.py --query "..." [--index_dir ./Embedding/Data_Base] [--top_k 5]
"""

import argparse, json, sys, re, math, logging
from pathlib import Path
from datetime import datetime

# dependencies
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi
except Exception as e:
    print(json.dumps({"error": f"Missing dependencies: {e}", "hint": "pip install sentence-transformers faiss-cpu rank_bm25 numpy"}))
    sys.exit(1)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("retrieval")

# ---------------- Config & helpers ----------------
DEFAULT_INDEX_DIR = Path("Embedding/Data_Base")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional reranker
SNIPPET_LEN = 350

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def read_jsonl(path):
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try:
                out.append(json.loads(ln))
            except:
                continue
    return out

def normalize_whitespace(s):
    return " ".join(str(s or "").split())

def extract_years(s):
    if not s: return set()
    yrs_full = re.findall(r"\b(?:19|20)\d{2}\b", s)
    return set(yrs_full)

def extract_company_tokens(s):
    tokens = re.findall(r"[A-Z]{2,}|[A-Za-z]{3,}", s)
    tokens = [t.strip() for t in tokens if len(t.strip()) >= 3]
    return set(tokens)

# ---------------- Load artifacts ----------------
def load_faiss(index_dir: Path):
    idx_path = index_dir / "faiss.index"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at {idx_path}")
    index = faiss.read_index(str(idx_path))
    return index

def load_id_map(index_dir: Path):
    arr = read_jsonl(index_dir / "id_map.jsonl")
    id_map = {}
    for o in arr:
        try:
            vid = int(o.get("vector_id"))
            cid = o.get("chunk_id")
            if cid is not None:
                id_map[vid] = cid
        except:
            continue
    return id_map

def load_meta(index_dir: Path):
    arr = read_jsonl(index_dir / "meta.jsonl")
    meta = {}
    for o in arr:
        cid = o.get("chunk_id")
        if cid:
            meta[cid] = o
    return meta

def load_full(index_dir: Path):
    arr = read_jsonl(index_dir / "chunks_full.jsonl")
    full = {}
    for o in arr:
        cid = o.get("chunk_id")
        if cid:
            full[cid] = o
    return full

# ---------------- Index building for BM25 ----------------
def build_fielded_corpus(meta_map, full_map):
    ids = []
    texts = []
    fielded = []
    for cid, m in meta_map.items():
        f = full_map.get(cid, {})
        content = f.get("content") or m.get("content") or m.get("snippet") or ""
        pdf_name = (m.get("pdf_name") or f.get("pdf_name") or "")
        section_type = (m.get("section_type") or f.get("section_type") or "")
        field_text = f"{pdf_name} {section_type} {content}"
        tokens = [t.lower() for t in re.findall(r"\w+", field_text)]
        ids.append(cid)
        texts.append(field_text)
        fielded.append(tokens)
    return ids, texts, fielded

# ---------------- Semantic helpers ----------------
def embed_query(model, text):
    v = model.encode([normalize_whitespace(text)], convert_to_numpy=True, show_progress_bar=False)[0].astype("float32")
    norm = np.linalg.norm(v) + 1e-9
    return v / norm

def run_faiss_search(index, q_vec, k, ef_search=None):
    try:
        if ef_search is not None and hasattr(index, 'hnsw'):
            index.hnsw.efSearch = ef_search
    except Exception:
        pass
    q = np.array([q_vec], dtype="float32")
    D, I = index.search(q, k)
    return D[0].tolist(), I[0].tolist()

# ---------------- Fusion & scoring ----------------
def normalize_scores_dict(d):
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}

def fuse_scores(semantic_scores, bm25_scores, meta_boosts, weights=(0.45,0.35,0.20)):
    sd = normalize_scores_dict(semantic_scores)
    bd = normalize_scores_dict(bm25_scores)
    fused = {}
    for cid in set(list(sd.keys()) + list(bd.keys())):
        s = sd.get(cid, 0.0)
        b = bd.get(cid, 0.0)
        meta = meta_boosts.get(cid, 0.0)
        combined = weights[0]*s + weights[1]*b + weights[2]*meta
        fused[cid] = combined
    return fused

# ---------------- Main retrieval pipeline ----------------
def retrieval_pipeline(
    query,
    index_dir=DEFAULT_INDEX_DIR,
    top_k=5,
    semantic_top_n=200,
    bm25_top_n=200,
    ef_search=200,
    use_rerank=True,
    rerank_model_name=CROSS_ENCODER_MODEL,
):
    start_time = datetime.utcnow()
    log.info(f"Query: {query}")

    # load artifacts
    meta = load_meta(index_dir)
    full = load_full(index_dir)
    id_map = load_id_map(index_dir)
    index = load_faiss(index_dir)

    # Preprocess query
    query_norm = normalize_whitespace(query)
    query_years = extract_years(query_norm)
    company_tokens = extract_company_tokens(query_norm)
    has_company_token = len(company_tokens) > 0
    log.info(f"Parsed years: {query_years} | company tokens (sample): {list(company_tokens)[:6]}")

    # Build BM25 corpus (fielded)
    ids_corpus, texts_corpus, tokenized_corpus = build_fielded_corpus(meta, full)
    bm25 = BM25Okapi(tokenized_corpus)

    # BM25 scoring
    q_tokens = [t.lower() for t in re.findall(r"\w+", query_norm)]
    bm25_scores_arr = bm25.get_scores(q_tokens)
    bm25_scores = {ids_corpus[i]: float(bm25_scores_arr[i]) for i in np.argsort(bm25_scores_arr)[::-1][:bm25_top_n] if bm25_scores_arr[i] > 0}

    # Semantic retrieval
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_vec = embed_query(model, query_norm)
    D, I = run_faiss_search(index, q_vec, semantic_top_n, ef_search=ef_search)

    semantic_map_raw = {}
    for dist, idx in zip(D, I):
        if idx < 0: continue
        cid = id_map.get(int(idx))
        if cid:
            semantic_map_raw[cid] = float(dist)

    # Convert semantic distances -> similarity-like scores
    if semantic_map_raw:
        dvals = list(semantic_map_raw.values())
        lo, hi = min(dvals), max(dvals)
        denom = (hi - lo + 1e-9)
        semantic_scores = {k: 1.0 - ((v - lo) / denom) for k, v in semantic_map_raw.items()}
    else:
        semantic_scores = {}

    # Metadata boosts
    meta_boosts = {}
    for cid in set(list(semantic_scores.keys()) + list(bm25_scores.keys())):
        m = meta.get(cid, {}) or {}
        boost = 0.0
        if query_years:
            year_match = False
            cand_fields = [
                str(m.get("last_reported","")),
                str(m.get("extracted_at_utc","")),
                str(m.get("pdf_name","")),
                str(full.get(cid, {}).get("content",""))
            ]
            for field in cand_fields:
                for y in query_years:
                    if y in field:
                        year_match = True
                        break
                if year_match: break
            if year_match:
                boost += 1.0
        pdfname = str(m.get("pdf_name","") or "")
        if has_company_token and any(tok.lower() in pdfname.lower() for tok in company_tokens):
            boost += 0.8
        sec_type = str(m.get("section_type") or full.get(cid, {}).get("section_type","") or "")
        if "table" in sec_type.lower() or m.get("table_id") or full.get(cid, {}).get("table_id"):
            boost += 0.5
        cnt = (full.get(cid, {}).get("content","") or m.get("content","") or "").lower()
        if any(k in cnt for k in ("cibil","credit","score","rank","company credit","company credit report")):
            boost += 0.7
        meta_boosts[cid] = boost

    meta_boosts = normalize_scores_dict(meta_boosts)

    # Fusion
    fused = fuse_scores(semantic_scores, {k: v for k,v in bm25_scores.items()}, meta_boosts,
                        weights=(0.45, 0.35, 0.20))

    # Select candidates for rerank
    candidates_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [cid for cid, _ in candidates_sorted[:max(30, top_k*4)]]

    # Cross-encoder rerank (optional)
    if use_rerank and top_candidates:
        try:
            cross = CrossEncoder(rerank_model_name)
            pairs = []
            cand_map = {}
            for cid in top_candidates:
                cont = (full.get(cid, {}).get("content") or meta.get(cid, {}).get("content") or "")
                pairs.append((query_norm, cont))
                cand_map[len(pairs)-1] = cid
            rerank_scores = cross.predict(pairs)
            rerank_map = {cand_map[i]: float(s) for i, s in enumerate(rerank_scores)}
            rerank_map = normalize_scores_dict(rerank_map)
            final_scores = {}
            for cid in top_candidates:
                base = fused.get(cid, 0.0)
                r = rerank_map.get(cid, 0.0)
                final_scores[cid] = 0.6 * r + 0.4 * base
        except Exception as e:
            log.warning(f"Rerank failed: {e} -- falling back to fused scores")
            final_scores = {cid: fused.get(cid, 0.0) for cid in top_candidates}
    else:
        final_scores = {cid: fused.get(cid, 0.0) for cid in top_candidates}

    # Build lists for printing & JSON
    semantic_order = sorted([(cid, semantic_scores.get(cid, 0.0)) for cid in semantic_scores.keys()],
                            key=lambda x: x[1], reverse=True)
    semantic_top = [cid for cid, _ in semantic_order if cid in final_scores][:top_k]

    keyword_order = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
    keyword_top = [cid for cid,_ in keyword_order if cid in final_scores][:top_k]

    def fmt_snippet(text):
        if not text:
            return "<no content>"
        t = text.strip()
        return (t[:SNIPPET_LEN] + ("..." if len(t) > SNIPPET_LEN else ""))

    def print_chunk_list(title, cids, score_map):
        print("\n" + "="*80)
        print(title)
        print("="*80)
        for i, cid in enumerate(cids, 1):
            m = meta.get(cid, {}) or {}
            f = full.get(cid, {}) or {}
            pdfname = m.get("pdf_name") or f.get("pdf_name") or "<unknown>"
            content = (f.get("content") or m.get("content") or "").strip()
            snippet = fmt_snippet(content)
            score = score_map.get(cid, None)
            print(f"[{i}] {cid} | {pdfname}")
            if score is not None:
                print(f"    score: {score:.5f}")
            print("    content:", snippet)
            print("-"*80)

    display_scores_sem = {cid: final_scores.get(cid, semantic_scores.get(cid,0.0)) for cid in semantic_top}
    display_scores_kw = {cid: final_scores.get(cid, bm25_scores.get(cid,0.0)) for cid in keyword_top}

    print("\n" + "#"*80)
    print(f"Query: {query}")
    print(f"Retrieved at: {now_iso()}")
    print("#"*80)

    print_chunk_list("SEMANTIC TOP RESULTS (chunk_id | pdf_name | content... )", semantic_top, display_scores_sem)
    print_chunk_list("KEYWORD (BM25) TOP RESULTS (chunk_id | pdf_name | content... )", keyword_top, display_scores_kw)

    elapsed = (datetime.utcnow() - start_time).total_seconds()
    log.info(f"Completed retrieval in {elapsed:.2f}s")

    # Build structured results (full content + metadata) for Streamlit
    def build_chunk_entry(cid, score_source_map):
        m = meta.get(cid, {}) or {}
        f = full.get(cid, {}) or {}
        entry = {
            "chunk_id": cid,
            "pdf_name": m.get("pdf_name") or f.get("pdf_name") or "<unknown>",
            "content": f.get("content") or m.get("content") or m.get("snippet") or "",
            "page": m.get("page") or f.get("page") or None,
            "section_type": m.get("section_type") or f.get("section_type") or None,
            "table_id": m.get("table_id") or f.get("table_id") or None,
            "score": float(score_source_map.get(cid, 0.0))
        }
        return entry

    semantic_results = [ build_chunk_entry(cid, final_scores if final_scores else {**{cid: semantic_scores.get(cid,0.0)}}) for cid in semantic_top ]
    keyword_results  = [ build_chunk_entry(cid, final_scores if final_scores else {**{cid: bm25_scores.get(cid,0.0)}}) for cid in keyword_top ]

    output = {
        "query": query,
        "retrieved_at": now_iso(),
        "semantic": { "results": semantic_results },
        "keyword":  { "results": keyword_results },
        "final_scores": { cid: float(s) for cid, s in final_scores.items() }
    }

    # Print the JSON blob as the last thing to stdout so external callers can parse it.
    print("\n" + "="*80)
    print("RETRIEVAL_JSON_OUTPUT:")
    print(json.dumps(output, ensure_ascii=False))
    # Also print pretty json to stderr (for readability) if you want (optional)
    # print(json.dumps(output, indent=2, ensure_ascii=False), file=sys.stderr)

    return output

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=False)
    p.add_argument("--index_dir", type=str, default=str(DEFAULT_INDEX_DIR))
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--semantic_top_n", type=int, default=200)
    p.add_argument("--bm25_top_n", type=int, default=200)
    p.add_argument("--ef_search", type=int, default=200)
    p.add_argument("--no_rerank", action="store_true", help="Disable cross-encoder rerank")
    args = p.parse_args()

    if args.query:
        query = args.query
    else:
        if sys.stdin.isatty():
            query = input("Enter query: ").strip()
        else:
            query = sys.stdin.read().strip()
    if not query:
        print("No query provided. Use --query or pipe input.")
        return

    idx_dir = Path(args.index_dir)
    if not idx_dir.exists():
        print(f"Index directory not found: {idx_dir}")
        return

    retrieval_pipeline(
        query,
        index_dir=idx_dir,
        top_k=args.top_k,
        semantic_top_n=args.semantic_top_n,
        bm25_top_n=args.bm25_top_n,
        ef_search=args.ef_search,
        use_rerank=(not args.no_rerank)
    )

if __name__ == "__main__":
    main()
