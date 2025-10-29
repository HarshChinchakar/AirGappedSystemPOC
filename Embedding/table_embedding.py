#!/usr/bin/env python3
"""
build_table_summary_embeddings.py

Create a dedicated embedding index for table summaries only.

Reads:
 - ./Data Ingestion/Extracted Data/tables.json            (full table entries)
 - ./Data Ingestion/Extracted Data/table_summaries/*.jsonl (generated summaries jsonl)

Writes artifacts to:
 - <output_dir>/embeddings.npy        (memmap)
 - <output_dir>/faiss.index
 - <output_dir>/id_map.jsonl
 - <output_dir>/meta.jsonl
 - <output_dir>/chunks_full.jsonl
 - <output_dir>/index.meta.json
 - <output_dir>/embedding_run_summary.json

Usage:
  python3 build_table_summary_embeddings.py
  (or run with --help)
"""
from pathlib import Path
import json, sys, time, logging, argparse, math, hashlib
from datetime import datetime
from typing import List, Dict, Any

# external deps
try:
    import numpy as np
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception as e:
    print("Missing dependency:", e, file=sys.stderr)
    print("Install: pip install sentence-transformers faiss-cpu numpy tqdm", file=sys.stderr)
    raise

# ---------------- Config / CLI ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--tables-json", type=str, default="Data Ingestion/Extracted Data/tables.json",
                    help="Full tables JSON (array) produced by the extractor")
parser.add_argument("--summaries", type=str, default="Data Ingestion/Extracted Data/table_summaries/tables_summaries.jsonl",
                    help="Table summaries JSONL (one JSON per line)")
parser.add_argument("--out", type=str, default="Embedding/Table Embeddings",
                    help="Output directory for table embeddings")
parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--hnsw-m", type=int, default=32)
parser.add_argument("--ef-construction", type=int, default=200)
parser.add_argument("--ef-search", type=int, default=50)
args = parser.parse_args()

TABLES_JSON = Path(args.tables_json)
SUMMARIES_FILE = Path(args.summaries)
OUTPUT_DIR = Path(args.out)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = args.model
BATCH_SIZE = int(args.batch_size)
HNSW_M = int(args.hnsw_m)
HNSW_EF_CONSTRUCTION = int(args.ef_construction)
HNSW_EF_SEARCH = int(args.ef_search)

LOG_PATH = OUTPUT_DIR / "build_table_embeddings.log"
logging.basicConfig(filename=str(LOG_PATH), level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
console = logging.getLogger("console")
console.setLevel(logging.INFO)

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def sha1_prefix(s: str, n=10):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

# ---------------- Helpers ----------------
def read_json_file(p: Path) -> List[Dict[str,Any]]:
    if not p.exists():
        logging.warning(f"{p} not found.")
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # possible single-object file - wrap
            return [data]
        return []
    except Exception as e:
        logging.exception(f"Failed to load {p}: {e}")
        return []

def read_jsonl(path: Path) -> List[Dict[str,Any]]:
    out = []
    if not path.exists():
        logging.warning(f"{path} not found.")
        return out
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                logging.warning("Skipping invalid json line in %s", path)
                continue
    return out

def make_chunk_id(prefix: str, suffix: str):
    return f"{prefix}_chunk_{sha1_prefix(suffix)}"

# ---------------- Load inputs ----------------
start_time = time.time()
console.info("Loading source files...")
tables = read_json_file(TABLES_JSON)
summaries = read_jsonl(SUMMARIES_FILE)

console.info(f"Loaded {len(tables)} table entries and {len(summaries)} summaries.")

# Build map by table_id for quick lookup
table_map = { t.get("table_id"): t for t in tables if t.get("table_id") }

# Prepare chunks from summaries: keep only those summaries that we want to embed.
prepared_chunks: List[Dict[str,Any]] = []
missing_table_count = 0
for s in summaries:
    # pick content: prefer generated_summary -> summary -> content
    content = (s.get("generated_summary") or s.get("summary") or s.get("content") or "").strip()
    if not content:
        # skip empty summaries
        continue
    table_id = s.get("table_id") or s.get("table") or None
    # chunk id: prefer existing chunk_id else table_id else hashed content
    if s.get("chunk_id"):
        cid = s.get("chunk_id")
    elif table_id:
        cid = table_id + "_summary"
    else:
        cid = make_chunk_id("table_summary", content[:200])
    # attach original_table if available
    original_table = table_map.get(table_id)
    if original_table is None:
        missing_table_count += 1
    chunk = {
        "chunk_id": cid,
        "content": content,
        "source_reference": s.get("source_reference") or s.get("source_file"),
        "pdf_name": s.get("source_file") or s.get("pdf_name"),
        "page": s.get("page"),
        "section_type": s.get("section_type") or "table_summary",
        "table_id": table_id,
        "original_table": original_table  # may be None
    }
    prepared_chunks.append(chunk)

console.info(f"Prepared {len(prepared_chunks)} summary chunks (missing original_table for {missing_table_count} summaries).")

# dedupe by chunk_id preserving order
seen = set()
deduped = []
for c in prepared_chunks:
    cid = c.get("chunk_id")
    if not cid:
        cid = make_chunk_id(c.get("pdf_name","table_summary"), (c.get("content") or "")[:200])
        c["chunk_id"] = cid
    if cid in seen:
        continue
    seen.add(cid)
    deduped.append(c)
prepared_chunks = deduped

# filter tiny
filtered = [c for c in prepared_chunks if c.get("content") and len(c.get("content").strip()) > 8]
console.info(f"Filtered tiny/empty chunks; remaining: {len(filtered)}")

if len(filtered) == 0:
    console.error("No table-summary chunks to embed. Exiting.")
    sys.exit(1)

# ---------------- Build embeddings ----------------
console.info(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
dim = model.get_sentence_embedding_dimension()
console.info(f"Model dim: {dim}")

n = len(filtered)
console.info(f"Embedding {n} table-summary chunks; batch_size={BATCH_SIZE}")

# Prepare FAISS index and memmap
index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
try:
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
except Exception:
    pass

emb_path = OUTPUT_DIR / "embeddings.npy"
# remove old artifacts if exist
for p in (OUTPUT_DIR / "faiss.index", OUTPUT_DIR / "id_map.jsonl", OUTPUT_DIR / "meta.jsonl", OUTPUT_DIR / "chunks_full.jsonl"):
    if p.exists():
        p.unlink()

# create memmap
embeddings_memmap = np.memmap(str(emb_path), dtype="float32", mode="w+", shape=(n, dim))

id_map_path = OUTPUT_DIR / "id_map.jsonl"
meta_path = OUTPUT_DIR / "meta.jsonl"
chunks_full_path = OUTPUT_DIR / "chunks_full.jsonl"

vid_counter = 0
for start in tqdm(range(0, n, BATCH_SIZE), desc="Embedding batches"):
    end = min(n, start + BATCH_SIZE)
    batch = filtered[start:end]
    texts = [c["content"] for c in batch]
    emb = model.encode(texts, batch_size=len(texts), show_progress_bar=False, convert_to_numpy=True)
    emb = emb.astype("float32")
    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    emb = emb / norms

    # write to memmap and faiss
    embeddings_memmap[start:end, :] = emb
    index.add(emb)

    # write id_map + meta + chunks_full incrementally
    with id_map_path.open("a", encoding="utf-8") as imf, meta_path.open("a", encoding="utf-8") as mf, chunks_full_path.open("a", encoding="utf-8") as cf:
        for i, c in enumerate(batch):
            vid = vid_counter
            imf.write(json.dumps({"vector_id": vid, "chunk_id": c["chunk_id"]}, ensure_ascii=False) + "\n")
            snippet = (c.get("content") or "")[:300].replace("\n", " ")
            meta_obj = {
                "chunk_id": c["chunk_id"],
                "source_reference": c.get("source_reference"),
                "pdf_name": c.get("pdf_name"),
                "page": c.get("page"),
                "section_type": c.get("section_type"),
                "snippet": snippet,
                "content": c.get("content"),
                "table_id": c.get("table_id"),
            }
            mf.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")
            # store full chunk (including original_table if present)
            cf.write(json.dumps(c, ensure_ascii=False) + "\n")
            vid_counter += 1

# flush memmap
embeddings_memmap.flush()

# finalize faiss index
try:
    index.hnsw.efSearch = HNSW_EF_SEARCH
except Exception:
    pass

faiss_index_path = OUTPUT_DIR / "faiss.index"
faiss.write_index(index, str(faiss_index_path))
console.info(f"Wrote FAISS index to {faiss_index_path}")

# write index meta / summary
index_meta = {
    "embedding_model": MODEL_NAME,
    "dim": int(dim),
    "num_vectors": int(n),
    "index_type": "IndexHNSWFlat",
    "hnsw": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION, "efSearch": HNSW_EF_SEARCH},
    "created_at": now_iso()
}
with (OUTPUT_DIR / "index.meta.json").open("w", encoding="utf-8") as f:
    json.dump(index_meta, f, indent=2, ensure_ascii=False)

elapsed = time.time() - start_time
artifacts = {
    "embeddings": str((OUTPUT_DIR / "embeddings.npy").resolve()),
    "faiss_index": str((OUTPUT_DIR / "faiss.index").resolve()),
    "id_map": str((OUTPUT_DIR / "id_map.jsonl").resolve()),
    "meta": str((OUTPUT_DIR / "meta.jsonl").resolve()),
    "chunks_full": str((OUTPUT_DIR / "chunks_full.jsonl").resolve()),
    "index_meta": str((OUTPUT_DIR / "index.meta.json").resolve()),
    "num_vectors": n
}
run_summary = {
    "timestamp": now_iso(),
    "tables_json": str(TABLES_JSON.resolve()),
    "summaries_file": str(SUMMARIES_FILE.resolve()),
    "num_chunks": n,
    "artifacts": artifacts,
    "time_taken_s": round(elapsed, 2)
}
with (OUTPUT_DIR / "embedding_run_summary.json").open("w", encoding="utf-8") as f:
    json.dump(run_summary, f, indent=2, ensure_ascii=False)

console.info("Embedding run completed.")
print("Done. Artifacts:")
for k,v in artifacts.items():
    print(f"{k}: {v}")
