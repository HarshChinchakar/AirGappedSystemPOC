# #!/usr/bin/env python3
# """
# build_embeddings_from_extracted.py

# Reads:
#   ./Data Ingestion/Extracted Data/text_chunks.json
#   ./Data Ingestion/Extracted Data/tables.json

# Creates embeddings with sentence-transformers/all-MiniLM-L6-v2
# and writes artifacts to:
#   ./Embedding/Data_Base/
#     - embeddings.npy (memmap)
#     - faiss.index
#     - id_map.jsonl
#     - meta.jsonl  (contains full content)
#     - chunks_full.jsonl (backup of chunks)
#     - index.meta.json
#     - embedding_run_summary.json

# Behavior:
#  - Splits large text/table chunks using TOKEN_LIMIT (uses tiktoken if installed).
#  - Deduplicates chunk_ids.
#  - Streams embeddings in batches; writes to memmap and adds to FAISS incrementally.
#  - Includes robust logging.
# """
# from pathlib import Path
# import json, os, sys, hashlib, time, math
# from datetime import datetime
# from typing import List, Dict, Any
# import logging

# try:
#     import numpy as np
#     from tqdm import tqdm
#     from sentence_transformers import SentenceTransformer
#     import faiss
# except Exception as e:
#     print("Missing required libraries. Please install: sentence-transformers faiss-cpu numpy tqdm", file=sys.stderr)
#     raise

# # optional tokenizer
# try:
#     import tiktoken
# except Exception:
#     tiktoken = None

# # ---------------- Config ----------------
# TEXT_CHUNKS_FILE = Path("./Data Ingestion/Extracted Data/text_chunks.json")
# TABLES_FILE = Path("./Data Ingestion/Extracted Data/tables.json")
# OUTPUT_DIR = Path("./Embedding/Data_Base")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# DEFAULT_BATCH_SIZE = 128
# TOKEN_LIMIT = 900  # keep consistent with your pipelines
# TABLE_SUMMARY_ROWS = 5
# HNSW_M = 32
# HNSW_EF_CONSTRUCTION = 200
# HNSW_EF_SEARCH = 50

# LOG_FILE = OUTPUT_DIR / "build_embeddings.log"
# logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO,
#                     format="%(asctime)s | %(levelname)s | %(message)s")
# console_log = logging.getLogger("console")
# console_log.setLevel(logging.INFO)

# # ---------------- Helpers ----------------
# def now_iso():
#     return datetime.utcnow().isoformat() + "Z"

# def sha1_prefix(s: str, n=10):
#     return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

# def normalize_whitespace(s: str) -> str:
#     return " ".join(s.strip().split())

# # tokenizer / token counting
# def get_token_encoder():
#     if tiktoken is None:
#         return None
#     try:
#         return tiktoken.get_encoding("cl100k_base")
#     except Exception:
#         try:
#             return tiktoken.encoding_for_model("gpt-4o-mini")
#         except Exception:
#             return None

# ENC = get_token_encoder()

# def count_tokens_for_text(text: str) -> int:
#     if not text:
#         return 0
#     if ENC is None:
#         return max(1, int(len(text) / 4))
#     try:
#         return len(ENC.encode(text))
#     except Exception:
#         return max(1, int(len(text) / 4))

# # ---------------- Splitting helpers ----------------
# def split_text_by_token_limit(text: str, token_limit: int):
#     """Token-aware (if possible) splitting on word boundaries."""
#     if count_tokens_for_text(text) <= token_limit:
#         return [text]
#     words = text.split()
#     parts = []
#     cur = []
#     for w in words:
#         cur.append(w)
#         cand = " ".join(cur)
#         if count_tokens_for_text(cand) >= token_limit:
#             if len(cur) > 1:
#                 parts.append(" ".join(cur[:-1]))
#                 cur = [cur[-1]]
#             else:
#                 parts.append(cand)
#                 cur = []
#     if cur:
#         parts.append(" ".join(cur))
#     parts = [p for p in parts if p.strip()]
#     return parts if parts else [text[:2000]]

# def token_estimate_for_rows(rows: List[List[Any]]):
#     try:
#         s = json.dumps(rows, ensure_ascii=False)
#     except Exception:
#         s = "\n".join(["\t".join([str(c) if c is not None else "" for c in r]) for r in rows])
#     return count_tokens_for_text(s), len(s)

# def split_table_rows_by_token_limit(header: List[str], rows: List[List[Any]], token_limit: int):
#     """rows: list-of-rows (data rows). Returns list of parts: each part = [header] + data rows"""
#     parts = []
#     header_row = [str(h) for h in header] if header else []
#     cur = [header_row]
#     for r in rows:
#         if not any(str(x).strip() for x in r):
#             continue
#         tentative = cur + [r]
#         tok, _ = token_estimate_for_rows(tentative)
#         if tok <= token_limit:
#             cur.append(r)
#         else:
#             if len(cur) == 1:
#                 parts.append(tentative)
#                 cur = [header_row]
#             else:
#                 parts.append(list(cur))
#                 cur = [header_row, r]
#     if len(cur) > 1:
#         parts.append(list(cur))
#     return parts

# def coerce_cell(x):
#     if x is None:
#         return ""
#     return str(x)

# def make_chunk_id(prefix: str, suffix: str):
#     return f"{prefix}_chunk_{sha1_prefix(suffix)}"

# # ---------------- Read & Prepare ----------------
# def load_json_file(path: Path):
#     if not path.exists():
#         logging.warning(f"{path} not found.")
#         return []
#     try:
#         return json.loads(path.read_text(encoding="utf-8"))
#     except Exception as e:
#         logging.exception(f"Failed to load {path}: {e}")
#         return []

# def prepare_chunks_from_texts(text_chunks: List[Dict[str,Any]], token_limit: int):
#     out = []
#     for c in text_chunks:
#         content = c.get("content") or c.get("text") or ""
#         content = normalize_whitespace(content)
#         if not content:
#             continue
#         base_id = c.get("chunk_id") or make_chunk_id(c.get("source_file","text"), content[:200])
#         if count_tokens_for_text(content) <= token_limit:
#             out.append({
#                 "chunk_id": base_id,
#                 "content": content,
#                 "source_reference": c.get("source_reference"),
#                 "pdf_name": c.get("source_file") or c.get("pdf_name"),
#                 "page": c.get("page"),
#                 "section_type": c.get("section_type") or "text",
#                 "bbox": c.get("bbox"),
#                 "avg_confidence": c.get("avg_confidence")
#             })
#         else:
#             parts = split_text_by_token_limit(content, token_limit)
#             for i, p in enumerate(parts, start=1):
#                 cid = f"{base_id}_part{i}"
#                 out.append({
#                     "chunk_id": cid,
#                     "content": p,
#                     "source_reference": c.get("source_reference"),
#                     "pdf_name": c.get("source_file") or c.get("pdf_name"),
#                     "page": c.get("page"),
#                     "section_type": c.get("section_type") or "text",
#                     "bbox": c.get("bbox"),
#                     "avg_confidence": c.get("avg_confidence")
#                 })
#     return out

# def prepare_chunks_from_tables(tables: List[Dict[str,Any]], token_limit: int, summary_rows=TABLE_SUMMARY_ROWS):
#     out = []
#     for t in tables:
#         src_file = t.get("source_file")
#         page = t.get("page")
#         table_id = t.get("table_id") or f"{Path(src_file).stem}_table"
#         headers = t.get("headers") or []
#         rows = t.get("rows") or []
#         if not rows:
#             continue
#         norm_rows = [[coerce_cell(c) for c in r] for r in rows]
#         # detect if first row equals headers
#         first_row = norm_rows[0] if norm_rows else []
#         data_rows = norm_rows[1:] if len(norm_rows) > 1 and headers and all(str(a).strip() == str(b).strip() for a,b in zip(first_row, headers[:len(first_row)])) else norm_rows
#         # summary chunk
#         sample_rows = data_rows[:summary_rows]
#         summary_parts = []
#         if headers:
#             summary_parts.append(" | ".join([str(h).strip() for h in headers if str(h).strip() != ""]))
#         for r in sample_rows:
#             summary_parts.append(" | ".join([str(c).strip() for c in r]))
#         summary_text = "\n".join(summary_parts).strip()
#         if summary_text:
#             sid = f"{table_id}_summary"
#             out.append({
#                 "chunk_id": sid,
#                 "content": summary_text,
#                 "source_reference": t.get("source_reference"),
#                 "pdf_name": src_file,
#                 "page": page,
#                 "section_type": "table_summary",
#                 "table_id": table_id,
#                 "original_table": t
#             })
#         # full table parts
#         rows_for_parts = data_rows if data_rows else []
#         if not rows_for_parts:
#             rows_for_parts = [headers] if headers else []
#         parts = split_table_rows_by_token_limit(headers, rows_for_parts, token_limit)
#         for idx, part_rows in enumerate(parts, start=1):
#             serialized_lines = []
#             for r in part_rows:
#                 serialized_lines.append("\t".join([str(c) for c in r]))
#             serialized_text = "\n".join(serialized_lines)
#             cid = f"{table_id}_part{idx}"
#             out.append({
#                 "chunk_id": cid,
#                 "content": serialized_text,
#                 "source_reference": t.get("source_reference"),
#                 "pdf_name": src_file,
#                 "page": page,
#                 "section_type": "table",
#                 "table_id": table_id,
#                 "table_part_index": idx,
#                 "original_table": t
#             })
#     return out

# def dedupe_chunks(chunks: List[Dict[str,Any]]):
#     seen = set()
#     out = []
#     for c in chunks:
#         cid = c.get("chunk_id")
#         if not cid:
#             cid = make_chunk_id(c.get("pdf_name","chunk"), (c.get("content") or "")[:200])
#             c["chunk_id"] = cid
#         if cid in seen:
#             continue
#         seen.add(cid)
#         out.append(c)
#     return out

# # ---------------- Build and persist ----------------
# def build_embeddings_and_index(prepared_chunks: List[Dict[str,Any]], embedding_model: str, batch_size: int):
#     n = len(prepared_chunks)
#     if n == 0:
#         raise RuntimeError("No chunks to embed")

#     # init model
#     device = "cpu"
#     try:
#         import torch
#         if torch.cuda.is_available():
#             device = "cuda"
#     except Exception:
#         pass
#     model = SentenceTransformer(embedding_model, device=device)
#     dim = model.get_sentence_embedding_dimension()

#     # auto-tune batch size
#     if device == "cuda":
#         batch_size = max(batch_size, 256)
#     logging.info(f"Embedding model {embedding_model} on {device}; dim={dim}; batch_size={batch_size}; chunks={n}")

#     # Prepare FAISS index and memmap for embeddings
#     index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
#     try:
#         index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
#     except Exception:
#         pass

#     emb_path = OUTPUT_DIR / "embeddings.npy"
#     # create memmap
#     embeddings_memmap = np.memmap(str(emb_path), dtype="float32", mode="w+", shape=(n, dim))

#     id_map_path = OUTPUT_DIR / "id_map.jsonl"
#     meta_path = OUTPUT_DIR / "meta.jsonl"
#     chunks_full_path = OUTPUT_DIR / "chunks_full.jsonl"

#     # if files exist, overwrite (we are creating new run)
#     for p in (id_map_path, meta_path, chunks_full_path):
#         if p.exists():
#             p.unlink()

#     vid_counter = 0
#     # Process batches
#     for start in tqdm(range(0, n, batch_size), desc="Embedding batches"):
#         end = min(n, start + batch_size)
#         batch_chunks = prepared_chunks[start:end]
#         texts = [c["content"] for c in batch_chunks]
#         # compute embeddings
#         emb = model.encode(texts, batch_size=len(texts), show_progress_bar=False, convert_to_numpy=True)
#         emb = emb.astype("float32")
#         # normalize rows
#         norms = np.linalg.norm(emb, axis=1, keepdims=True)
#         norms[norms == 0.0] = 1.0
#         emb = emb / norms

#         # write to memmap
#         embeddings_memmap[start:end, :] = emb

#         # add to FAISS
#         index.add(emb)

#         # write id_map + meta + chunks_full incrementally
#         with id_map_path.open("a", encoding="utf-8") as imf, meta_path.open("a", encoding="utf-8") as mf, chunks_full_path.open("a", encoding="utf-8") as cf:
#             for i, c in enumerate(batch_chunks):
#                 vid = vid_counter
#                 imf.write(json.dumps({"vector_id": vid, "chunk_id": c["chunk_id"]}, ensure_ascii=False) + "\n")
#                 snippet = (c.get("content") or "")[:300].replace("\n", " ")
#                 meta_obj = {
#                     "chunk_id": c["chunk_id"],
#                     "source_reference": c.get("source_reference"),
#                     "pdf_name": c.get("pdf_name"),
#                     "page": c.get("page"),
#                     "section_type": c.get("section_type"),
#                     "snippet": snippet,
#                     # store full content too (fix previous pipeline bug)
#                     "content": c.get("content"),
#                     "table_id": c.get("table_id"),
#                     "table_part_index": c.get("table_part_index"),
#                 }
#                 mf.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")
#                 # backup full chunk record for debugging
#                 cf.write(json.dumps(c, ensure_ascii=False) + "\n")
#                 vid_counter += 1

#     # flush memmap to disk
#     embeddings_memmap.flush()

#     # finalize faiss index
#     try:
#         index.hnsw.efSearch = HNSW_EF_SEARCH
#     except Exception:
#         pass

#     faiss_index_path = OUTPUT_DIR / "faiss.index"
#     faiss.write_index(index, str(faiss_index_path))
#     logging.info(f"Wrote FAISS index to {faiss_index_path}")

#     # write index meta
#     index_meta = {
#         "embedding_model": embedding_model,
#         "dim": int(dim),
#         "num_vectors": int(n),
#         "index_type": "IndexHNSWFlat",
#         "hnsw": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION, "efSearch": HNSW_EF_SEARCH},
#         "created_at": now_iso()
#     }
#     with (OUTPUT_DIR / "index.meta.json").open("w", encoding="utf-8") as f:
#         json.dump(index_meta, f, indent=2, ensure_ascii=False)

#     return {
#         "embeddings": str(emb_path.resolve()),
#         "faiss_index": str(faiss_index_path.resolve()),
#         "id_map": str(id_map_path.resolve()),
#         "meta": str(meta_path.resolve()),
#         "chunks_full": str(chunks_full_path.resolve()),
#         "index_meta": str((OUTPUT_DIR / "index.meta.json").resolve()),
#         "num_vectors": int(n)
#     }

# # ---------------- Main ----------------
# def main():
#     start_time = time.time()
#     logging.info("Embedding run started")
#     # load source files
#     text_json = load_json_file(TEXT_CHUNKS_FILE)
#     table_json = load_json_file(TABLES_FILE)

#     text_entries = text_json if isinstance(text_json, list) else (text_json.get("chunks") if isinstance(text_json, dict) else [])
#     table_entries = table_json if isinstance(table_json, list) else (table_json.get("tables") if isinstance(table_json, dict) else [])

#     logging.info(f"Loaded text_chunks: {len(text_entries)} entries; tables: {len(table_entries)} entries")

#     # prepare
#     prepared_text_chunks = prepare_chunks_from_texts(text_entries, TOKEN_LIMIT)
#     prepared_table_chunks = prepare_chunks_from_tables(table_entries, TOKEN_LIMIT, summary_rows=TABLE_SUMMARY_ROWS)

#     combined = prepared_text_chunks + prepared_table_chunks
#     combined = dedupe_chunks(combined)
#     logging.info(f"Prepared total chunks for embedding (deduped): {len(combined)}")

#     # filter out tiny/empty chunks
#     filtered = [c for c in combined if (c.get("content") and len(c.get("content").strip()) > 8)]
#     logging.info(f"Filtered tiny chunks; remaining: {len(filtered)}")

#     # determine batch size
#     batch_size = DEFAULT_BATCH_SIZE
#     try:
#         import torch
#         if torch.cuda.is_available():
#             batch_size = max(batch_size, 256)
#     except Exception:
#         pass

#     artifacts = build_embeddings_and_index(filtered, EMBEDDING_MODEL, batch_size)

#     elapsed = time.time() - start_time
#     run_summary = {
#         "timestamp": now_iso(),
#         "input_text_chunks": str(TEXT_CHUNKS_FILE.resolve()),
#         "input_tables": str(TABLES_FILE.resolve()),
#         "num_chunks": len(filtered),
#         "artifacts": artifacts,
#         "time_taken_s": round(elapsed, 2)
#     }
#     with (OUTPUT_DIR / "embedding_run_summary.json").open("w", encoding="utf-8") as f:
#         json.dump(run_summary, f, indent=2, ensure_ascii=False)

#     logging.info(f"Embedding run completed: {run_summary}")
#     print("Done. Artifacts:")
#     for k,v in artifacts.items():
#         print(f"{k}: {v}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
build_embeddings_from_extracted.py

Reads:
  ./Data Ingestion/Extracted Data/backups/text_chunks.json
  ./Data Ingestion/Extracted Data/backups/tables_summaries.jsonl

Creates embeddings with sentence-transformers/all-MiniLM-L6-v2
and writes artifacts to:
  ./Embedding/Data_Base/
    - embeddings.npy (memmap)
    - faiss.index
    - id_map.jsonl
    - meta.jsonl  (contains full content)
    - chunks_full.jsonl (backup of chunks)
    - index.meta.json
    - embedding_run_summary.json

Behavior:
 - Splits large text/table chunks using TOKEN_LIMIT (uses tiktoken if installed).
 - Deduplicates chunk_ids.
 - Streams embeddings in batches; writes to memmap and adds to FAISS incrementally.
 - Includes robust logging.
"""
from pathlib import Path
import json, os, sys, hashlib, time, math
from datetime import datetime
from typing import List, Dict, Any
import logging

try:
    import numpy as np
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception as e:
    print("Missing required libraries. Please install: sentence-transformers faiss-cpu numpy tqdm", file=sys.stderr)
    raise

# optional tokenizer
try:
    import tiktoken
except Exception:
    tiktoken = None

# ---------------- Config ----------------
# using backup paths provided by user
TEXT_CHUNKS_FILE = Path("./Data Ingestion/Extracted Data/backups/text_chunks.json")
TABLES_FILE = Path("./Data Ingestion/Extracted Data/backups/tables_summaries.jsonl")
OUTPUT_DIR = Path("./Embedding/Data_Base")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 128
TOKEN_LIMIT = 900  # keep consistent with your pipelines
TABLE_SUMMARY_ROWS = 5
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 50

LOG_FILE = OUTPUT_DIR / "build_embeddings.log"
logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
console_log = logging.getLogger("console")
console_log.setLevel(logging.INFO)

# ---------------- Helpers ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def sha1_prefix(s: str, n=10):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def normalize_whitespace(s: str) -> str:
    return " ".join(s.strip().split())

# tokenizer / token counting
def get_token_encoder():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            return tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            return None

ENC = get_token_encoder()

def count_tokens_for_text(text: str) -> int:
    if not text:
        return 0
    if ENC is None:
        return max(1, int(len(text) / 4))
    try:
        return len(ENC.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))

# ---------------- Splitting helpers ----------------
def split_text_by_token_limit(text: str, token_limit: int):
    """Token-aware (if possible) splitting on word boundaries."""
    if count_tokens_for_text(text) <= token_limit:
        return [text]
    words = text.split()
    parts = []
    cur = []
    for w in words:
        cur.append(w)
        cand = " ".join(cur)
        if count_tokens_for_text(cand) >= token_limit:
            if len(cur) > 1:
                parts.append(" ".join(cur[:-1]))
                cur = [cur[-1]]
            else:
                parts.append(cand)
                cur = []
    if cur:
        parts.append(" ".join(cur))
    parts = [p for p in parts if p.strip()]
    return parts if parts else [text[:2000]]

def token_estimate_for_rows(rows: List[List[Any]]):
    try:
        s = json.dumps(rows, ensure_ascii=False)
    except Exception:
        s = "\n".join(["\t".join([str(c) if c is not None else "" for c in r]) for r in rows])
    return count_tokens_for_text(s), len(s)

def split_table_rows_by_token_limit(header: List[str], rows: List[List[Any]], token_limit: int):
    """rows: list-of-rows (data rows). Returns list of parts: each part = [header] + data rows"""
    parts = []
    header_row = [str(h) for h in header] if header else []
    cur = [header_row]
    for r in rows:
        if not any(str(x).strip() for x in r):
            continue
        tentative = cur + [r]
        tok, _ = token_estimate_for_rows(tentative)
        if tok <= token_limit:
            cur.append(r)
        else:
            if len(cur) == 1:
                parts.append(tentative)
                cur = [header_row]
            else:
                parts.append(list(cur))
                cur = [header_row, r]
    if len(cur) > 1:
        parts.append(list(cur))
    return parts

def coerce_cell(x):
    if x is None:
        return ""
    return str(x)

def make_chunk_id(prefix: str, suffix: str):
    return f"{prefix}_chunk_{sha1_prefix(suffix)}"

# ---------------- Read & Prepare ----------------
def load_json_file(path: Path):
    """
    Loads either JSON (list/dict) or JSONL (one JSON per line). Returns list for both cases.
    If file missing, returns [].
    """
    if not path.exists():
        logging.warning(f"{path} not found.")
        return []
    try:
        if path.suffix.lower() == ".jsonl" or path.suffix.lower().endswith("jsonl"):
            out = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        # try to be tolerant
                        continue
            return out
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # if dict contains top-level keys like "chunks" or "tables"
                if "chunks" in data and isinstance(data["chunks"], list):
                    return data["chunks"]
                if "tables" in data and isinstance(data["tables"], list):
                    return data["tables"]
                # fall back to returning the dict as single element
                return [data]
    except Exception as e:
        logging.exception(f"Failed to load {path}: {e}")
        return []
    return []

def prepare_chunks_from_texts(text_chunks: List[Dict[str,Any]], token_limit: int):
    out = []
    for c in text_chunks:
        content = c.get("content") or c.get("text") or ""
        content = normalize_whitespace(content)
        if not content:
            continue
        base_id = c.get("chunk_id") or make_chunk_id(c.get("source_file","text"), content[:200])
        if count_tokens_for_text(content) <= token_limit:
            out.append({
                "chunk_id": base_id,
                "content": content,
                "source_reference": c.get("source_reference"),
                "pdf_name": c.get("source_file") or c.get("pdf_name"),
                "page": c.get("page"),
                "section_type": c.get("section_type") or "text",
                "bbox": c.get("bbox"),
                "avg_confidence": c.get("avg_confidence")
            })
        else:
            parts = split_text_by_token_limit(content, token_limit)
            for i, p in enumerate(parts, start=1):
                cid = f"{base_id}_part{i}"
                out.append({
                    "chunk_id": cid,
                    "content": p,
                    "source_reference": c.get("source_reference"),
                    "pdf_name": c.get("source_file") or c.get("pdf_name"),
                    "page": c.get("page"),
                    "section_type": c.get("section_type") or "text",
                    "bbox": c.get("bbox"),
                    "avg_confidence": c.get("avg_confidence")
                })
    return out

def prepare_chunks_from_tables(tables: List[Dict[str,Any]], token_limit: int, summary_rows=TABLE_SUMMARY_ROWS):
    """
    Original table preparation kept (splitting full tables into parts + producing a summary chunk).
    If the input 'tables' are actually 'summaries' JSON objects (contain 'generated_summary' or 'generated_summary'),
    this function will also handle that by treating them as one-chunk summaries.
    """
    out = []
    # detect whether these are table-summaries (already pre-summarized)
    # heuristics: presence of 'generated_summary' or 'generated_summary' or 'generated_summary' keys
    is_summary_list = all(any(k in t for k in ("generated_summary", "generated_summary", "generated_summary_text", "summary", "content")) for t in tables) and len(tables) > 0

    if is_summary_list:
        logging.info("Detected table summaries input - preparing chunks directly from summaries.")
        for s in tables:
            # pick content
            content = s.get("generated_summary") or s.get("summary") or s.get("content") or s.get("generated_summary_text") or ""
            content = normalize_whitespace(content)
            if not content:
                continue
            # use chunk_id if present, else use table_id or table_id-derived
            cid = s.get("chunk_id") or s.get("table_id") or s.get("table") or make_chunk_id("table_summary", (s.get("table_id") or content)[:200])
            out.append({
                "chunk_id": cid,
                "content": content,
                "source_reference": s.get("source_reference") or s.get("source_file") or s.get("pdf_name"),
                "pdf_name": s.get("source_file") or s.get("pdf_name"),
                "page": s.get("page"),
                "section_type": s.get("section_type") or "table_summary",
                "table_id": s.get("table_id") or s.get("table"),
                "original_table": s.get("original_table") or None
            })
        return out

    # otherwise assume full tables structure and follow original logic
    for t in tables:
        src_file = t.get("source_file")
        page = t.get("page")
        table_id = t.get("table_id") or f"{Path(src_file).stem}_table"
        headers = t.get("headers") or []
        rows = t.get("rows") or []
        if not rows:
            continue
        norm_rows = [[coerce_cell(c) for c in r] for r in rows]
        # detect if first row equals headers
        first_row = norm_rows[0] if norm_rows else []
        data_rows = norm_rows[1:] if len(norm_rows) > 1 and headers and all(str(a).strip() == str(b).strip() for a,b in zip(first_row, headers[:len(first_row)])) else norm_rows
        # summary chunk
        sample_rows = data_rows[:summary_rows]
        summary_parts = []
        if headers:
            summary_parts.append(" | ".join([str(h).strip() for h in headers if str(h).strip() != ""]))
        for r in sample_rows:
            summary_parts.append(" | ".join([str(c).strip() for c in r]))
        summary_text = "\n".join(summary_parts).strip()
        if summary_text:
            sid = f"{table_id}_summary"
            out.append({
                "chunk_id": sid,
                "content": summary_text,
                "source_reference": t.get("source_reference"),
                "pdf_name": src_file,
                "page": page,
                "section_type": "table_summary",
                "table_id": table_id,
                "original_table": t
            })
        # full table parts
        rows_for_parts = data_rows if data_rows else []
        if not rows_for_parts:
            rows_for_parts = [headers] if headers else []
        parts = split_table_rows_by_token_limit(headers, rows_for_parts, token_limit)
        for idx, part_rows in enumerate(parts, start=1):
            serialized_lines = []
            for r in part_rows:
                serialized_lines.append("\t".join([str(c) for c in r]))
            serialized_text = "\n".join(serialized_lines)
            cid = f"{table_id}_part{idx}"
            out.append({
                "chunk_id": cid,
                "content": serialized_text,
                "source_reference": t.get("source_reference"),
                "pdf_name": src_file,
                "page": page,
                "section_type": "table",
                "table_id": table_id,
                "table_part_index": idx,
                "original_table": t
            })
    return out

def dedupe_chunks(chunks: List[Dict[str,Any]]):
    seen = set()
    out = []
    for c in chunks:
        cid = c.get("chunk_id")
        if not cid:
            cid = make_chunk_id(c.get("pdf_name","chunk"), (c.get("content") or "")[:200])
            c["chunk_id"] = cid
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out

# ---------------- Build and persist ----------------
def build_embeddings_and_index(prepared_chunks: List[Dict[str,Any]], embedding_model: str, batch_size: int):
    n = len(prepared_chunks)
    if n == 0:
        raise RuntimeError("No chunks to embed")

    # init model
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    model = SentenceTransformer(embedding_model, device=device)
    dim = model.get_sentence_embedding_dimension()

    # auto-tune batch size
    if device == "cuda":
        batch_size = max(batch_size, 256)
    logging.info(f"Embedding model {embedding_model} on {device}; dim={dim}; batch_size={batch_size}; chunks={n}")

    # Prepare FAISS index and memmap for embeddings
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    try:
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    except Exception:
        pass

    emb_path = OUTPUT_DIR / "embeddings.npy"
    # create memmap
    embeddings_memmap = np.memmap(str(emb_path), dtype="float32", mode="w+", shape=(n, dim))

    id_map_path = OUTPUT_DIR / "id_map.jsonl"
    meta_path = OUTPUT_DIR / "meta.jsonl"
    chunks_full_path = OUTPUT_DIR / "chunks_full.jsonl"

    # if files exist, overwrite (we are creating new run)
    for p in (id_map_path, meta_path, chunks_full_path):
        if p.exists():
            p.unlink()

    vid_counter = 0
    # Process batches
    for start in tqdm(range(0, n, batch_size), desc="Embedding batches"):
        end = min(n, start + batch_size)
        batch_chunks = prepared_chunks[start:end]
        texts = [c["content"] for c in batch_chunks]
        # compute embeddings
        emb = model.encode(texts, batch_size=len(texts), show_progress_bar=False, convert_to_numpy=True)
        emb = emb.astype("float32")
        # normalize rows
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        emb = emb / norms

        # write to memmap
        embeddings_memmap[start:end, :] = emb

        # add to FAISS
        index.add(emb)

        # write id_map + meta + chunks_full incrementally
        with id_map_path.open("a", encoding="utf-8") as imf, meta_path.open("a", encoding="utf-8") as mf, chunks_full_path.open("a", encoding="utf-8") as cf:
            for i, c in enumerate(batch_chunks):
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
                    # store full content too (fix previous pipeline bug)
                    "content": c.get("content"),
                    "table_id": c.get("table_id"),
                    "table_part_index": c.get("table_part_index"),
                }
                mf.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")
                # backup full chunk record for debugging
                cf.write(json.dumps(c, ensure_ascii=False) + "\n")
                vid_counter += 1

    # flush memmap to disk
    embeddings_memmap.flush()

    # finalize faiss index
    try:
        index.hnsw.efSearch = HNSW_EF_SEARCH
    except Exception:
        pass

    faiss_index_path = OUTPUT_DIR / "faiss.index"
    faiss.write_index(index, str(faiss_index_path))
    logging.info(f"Wrote FAISS index to {faiss_index_path}")

    # write index meta
    index_meta = {
        "embedding_model": embedding_model,
        "dim": int(dim),
        "num_vectors": int(n),
        "index_type": "IndexHNSWFlat",
        "hnsw": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION, "efSearch": HNSW_EF_SEARCH},
        "created_at": now_iso()
    }
    with (OUTPUT_DIR / "index.meta.json").open("w", encoding="utf-8") as f:
        json.dump(index_meta, f, indent=2, ensure_ascii=False)

    return {
        "embeddings": str(emb_path.resolve()),
        "faiss_index": str(faiss_index_path.resolve()),
        "id_map": str(id_map_path.resolve()),
        "meta": str(meta_path.resolve()),
        "chunks_full": str(chunks_full_path.resolve()),
        "index_meta": str((OUTPUT_DIR / "index.meta.json").resolve()),
        "num_vectors": int(n)
    }

# ---------------- Main ----------------
def main():
    start_time = time.time()
    logging.info("Embedding run started")
    # load source files
    text_json = load_json_file(TEXT_CHUNKS_FILE)
    table_json = load_json_file(TABLES_FILE)

    text_entries = text_json if isinstance(text_json, list) else (text_json.get("chunks") if isinstance(text_json, dict) else [])
    table_entries = table_json if isinstance(table_json, list) else (table_json.get("tables") if isinstance(table_json, dict) else [])

    logging.info(f"Loaded text_chunks: {len(text_entries)} entries; tables/summaries: {len(table_entries)} entries")

    # prepare
    prepared_text_chunks = prepare_chunks_from_texts(text_entries, TOKEN_LIMIT)

    # For table entries we support both:
    # - raw tables (with headers + rows) -> prepare_chunks_from_tables
    # - table summaries JSONL (with generated_summary/content fields) -> handled in same function
    prepared_table_chunks = prepare_chunks_from_tables(table_entries, TOKEN_LIMIT, summary_rows=TABLE_SUMMARY_ROWS)

    combined = prepared_text_chunks + prepared_table_chunks
    combined = dedupe_chunks(combined)
    logging.info(f"Prepared total chunks for embedding (deduped): {len(combined)}")

    # filter out tiny/empty chunks
    filtered = [c for c in combined if (c.get("content") and len(c.get("content").strip()) > 8)]
    logging.info(f"Filtered tiny chunks; remaining: {len(filtered)}")

    # determine batch size
    batch_size = DEFAULT_BATCH_SIZE
    try:
        import torch
        if torch.cuda.is_available():
            batch_size = max(batch_size, 256)
    except Exception:
        pass

    artifacts = build_embeddings_and_index(filtered, EMBEDDING_MODEL, batch_size)

    elapsed = time.time() - start_time
    run_summary = {
        "timestamp": now_iso(),
        "input_text_chunks": str(TEXT_CHUNKS_FILE.resolve()),
        "input_tables_or_summaries": str(TABLES_FILE.resolve()),
        "num_chunks": len(filtered),
        "artifacts": artifacts,
        "time_taken_s": round(elapsed, 2)
    }
    with (OUTPUT_DIR / "embedding_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    logging.info(f"Embedding run completed: {run_summary}")
    print("Done. Artifacts:")
    for k,v in artifacts.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
