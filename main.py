

# #!/usr/bin/env python3
# """
# Streamlit RAG Interface (Tables / Text + Tables selector)
# Final Version — Creator: Harsh Chinchakar
# """

# import os, sys, json, re, shlex, subprocess
# from pathlib import Path
# from datetime import datetime
# import streamlit as st
# from typing import List, Dict, Any
# from dotenv import load_dotenv

# # Load from your .env file
# load_dotenv(".env")
# BASE_DIR = Path(__file__).resolve().parent
# # # ---------------- CONFIG ----------------
# # DEFAULT_RETRIEVAL_SCRIPT = "Retrival/retrieval_combined_v2.py"
# # TABLES_ONLY_SCRIPT = "Retrival/retrival_tables.py"
# DEFAULT_RETRIEVAL_SCRIPT = str(BASE_DIR / "Retrival" / "retrieval_combined_v2.py")
# TABLES_ONLY_SCRIPT      = str(BASE_DIR / "Retrival" / "retrival_tables.py") 
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
# MODEL = "gpt-4o-mini"
# SEMANTIC_TOP = 12
# KEYWORD_TOP = 12
# CHUNK_CHAR_LIMIT = 2500
# TIMEOUT_SECONDS = 120

# # ---------------- Helper Functions ----------------
# def now_iso():
#     return datetime.utcnow().isoformat() + "Z"

# # def run_retrieval(query: str, retrieval_script: str):
# #     """Executes the retrieval script via subprocess and returns parsed JSON output."""
# #     cmd = f"python3 {shlex.quote(retrieval_script)} --query {shlex.quote(query)} --top_k {max(SEMANTIC_TOP, KEYWORD_TOP)}"
# #     proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
# #     if proc.returncode != 0:
# #         raise RuntimeError(f"Retrieval script failed: {proc.stderr}")

# #     match = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
# #     if not match:
# #         raise RuntimeError("No valid JSON found in retrieval output.")
# #     return json.loads(match.group(1))

# import shutil

# def run_retrieval(query: str, retrieval_script: str):
#     """
#     Run retrieval script via subprocess using the same Python interpreter / venv.
#     Returns parsed dict if successful or if the subprocess printed valid retrieval JSON
#     (even when the process exit code != 0). Raises RuntimeError only when no usable
#     JSON/chunks are found.

#     This is defensive to handle cases where the retrieval subprocess prints dependency
#     warnings/errors AFTER producing usable chunk JSON (or prints an 'error' JSON).
#     """
#     retrieval_script = str(retrieval_script)
#     if not os.path.isfile(retrieval_script):
#         raise RuntimeError(f"Retrieval script not found: {retrieval_script!r}")

#     python_exec = sys.executable or shutil.which("python3") or "python3"
#     top_k = max(SEMANTIC_TOP, KEYWORD_TOP)
#     cmd_list = [python_exec, retrieval_script, "--query", query, "--top_k", str(top_k)]

#     # Ensure subprocess uses same env and can import local modules
#     env = os.environ.copy()
#     existing_pp = env.get("PYTHONPATH", "")
#     env["PYTHONPATH"] = str(BASE_DIR) + (":" + existing_pp if existing_pp else "")

#     proc = subprocess.run(
#         cmd_list,
#         capture_output=True,
#         text=True,
#         timeout=TIMEOUT_SECONDS,
#         cwd=str(BASE_DIR),
#         env=env,
#         shell=False,
#     )

#     stdout = proc.stdout or ""
#     stderr = proc.stderr or ""

#     # Try to extract JSON (primary marker then fallback)
#     parsed = None
#     try:
#         m = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", stdout)
#         if not m:
#             m = re.search(r"(\{[\s\S]+\})", stdout)
#         if m:
#             parsed = json.loads(m.group(1))
#     except Exception as e:
#         # JSON exists but parse failed — treat as fatal, include stdout for debug
#         raise RuntimeError(f"Failed to parse JSON from retrieval stdout: {e}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

#     # If we got parsed JSON, check if it contains usable retrieval chunks
#     if isinstance(parsed, dict):
#         # If parsed has semantic/keyword results, return it even if rc != 0
#         has_semantic = bool(parsed.get("semantic", {}).get("results"))
#         has_keyword = bool(parsed.get("keyword", {}).get("results"))
#         if has_semantic or has_keyword:
#             return parsed

#         # If parsed contains an 'error' payload only, surface it as an error
#         if parsed.get("error") and not (has_semantic or has_keyword):
#             hint = ""
#             err_text = str(parsed.get("error") or "")
#             if "huggingface_hub" in err_text or "cached_download" in err_text:
#                 hint = "\nHint: the retrieval script reports a huggingface_hub version mismatch. Try upgrading: `pip install --upgrade huggingface_hub sentence-transformers numpy` in the venv used by Streamlit."
#             raise RuntimeError(f"Retrieval script returned an error payload: {err_text}{hint}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

#         # parsed JSON present but no semantic/keyword -> return parsed (caller can decide)
#         return parsed

#     # No JSON parsed
#     # If subprocess exited cleanly but printed no JSON -> fatal
#     if proc.returncode == 0:
#         raise RuntimeError(f"Retrieval subprocess completed (rc=0) but produced no JSON.\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

#     # proc.returncode != 0 and no JSON -> include hints for common dependency issues
#     # inspect stdout/stderr for known dependency messages
#     combined = stdout + "\n\n" + stderr
#     if "No module named" in combined or "cannot import name" in combined or "Missing dependencies" in combined:
#         suggestion = (
#             "\nSuggested fix: ensure the retrieval subprocess uses the same Python venv as Streamlit. "
#             "On the host, run `which python3` and `pip install sentence-transformers faiss-cpu rank_bm25 numpy huggingface_hub` "
#             "inside that environment. If you use virtualenv/venv, make sure sys.executable points to it."
#         )
#     else:
#         suggestion = ""

#     raise RuntimeError(
#         f"Retrieval script failed (rc={proc.returncode}) and no JSON could be parsed.\n\nCMD: {' '.join(cmd_list)}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n{suggestion}"
#     )


# def is_table_chunk(item: Dict[str, Any]) -> bool:
#     st = (item.get("section_type") or "").lower()
#     if "table" in st or "table_summary" in st:
#         return True
#     if item.get("table_id") or item.get("table_part_index"):
#         return True
#     pdf = (item.get("pdf_name") or "").lower()
#     if pdf.endswith(".xlsx") or pdf.endswith(".xls"):
#         return True
#     content = (item.get("content") or "")[:500]
#     if "\t" in content or " | " in content or content.strip().startswith("|"):
#         return True
#     return False

# def dedupe_and_merge(semantic, keyword):
#     combined, seen = [], set()
#     for src in (semantic or [])[:SEMANTIC_TOP] + (keyword or [])[:KEYWORD_TOP]:
#         cid = src.get("chunk_id") or f"{src.get('pdf_name')}::p{src.get('page')}"
#         if cid in seen:
#             continue
#         seen.add(cid)
#         content = (src.get("content") or "")[:CHUNK_CHAR_LIMIT]
#         combined.append({
#             "chunk_id": cid,
#             "pdf_name": src.get("pdf_name") or "<unknown>",
#             "page": src.get("page"),
#             "section_type": src.get("section_type"),
#             "score": src.get("score", None),
#             "content": content
#         })
#     return combined

# # ---------------- Core RAG Logic ----------------
# def run_rag_pipeline(query, scope="tables"):
#     """Main RAG logic — unchanged except for robust OpenAI usage + structured chunk context."""
#     if scope == "tables":
#         retrieval_script = TABLES_ONLY_SCRIPT
#         apply_filter = False
#     else:
#         retrieval_script = DEFAULT_RETRIEVAL_SCRIPT
#         apply_filter = True

#     retrieval = run_retrieval(query, retrieval_script)
#     sem_results = retrieval.get("semantic", {}).get("results", [])
#     kw_results = retrieval.get("keyword", {}).get("results", [])

#     def filter_scope(items):
#         if not apply_filter:
#             return items
#         if scope == "text":
#             return [i for i in items if not is_table_chunk(i)]
#         elif scope == "tables":
#             return [i for i in items if is_table_chunk(i)]
#         return items

#     sem_filtered, kw_filtered = filter_scope(sem_results), filter_scope(kw_results)
#     merged_chunks = dedupe_and_merge(sem_filtered, kw_filtered)

#     if not merged_chunks:
#         return {"answer": "No relevant chunks found for this query."}

#     # ---------------- Prepare structured context and system prompt ----------------
#     system_prompt = (
#         "Rules (obey strictly):\n"
#         "1) Use ONLY the information present in the provided retrieved chunks to answer. Prefer direct evidence but if indirect evidence is present use to or process it towards maximum bound and infer required facts without Generating facts\n"
#         "2) If the exact fact is not directly stated, you MAY synthesize or infer an answer based on INDIRECT or PARTIAL evidence found across chunks. "
#         "When you do so, explicitly mark the statement as 'Inferred' and cite the supporting chunks for that inference.\n"
#         "3) Only respond with: \"Information not found in retrieved dataset.\" when there is absolutely no relevant information or inference possible from the provided chunks.\n"
#         "4) Cite each factual statement with the chunk citation format: [chunk_id | pdf_name]. If multiple chunks support it, cite all.\n"
#         "5) Produce a concise, formal natural-language answer; follow with a 'Sources' list (chunk citations). "
#         "Also include a short 'Limitations' note if any inferred conclusions were used.\n"
#         "When the required data isnt completely present then focus on the present data in the retrieved chunks and frame a short answer towards that"
#         "(7) When numbers are not present in the chunks -DO NOT ANSWER WITH NUMERIC ANSWERS - State that information is not present but this is what we found"
#     )

#     # Send the chunks as structured JSON array
#     chunk_context = json.dumps(merged_chunks, ensure_ascii=False)

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {
#             "role": "user",
#             "content": (
#                 f"User query: {query}\n\n"
#                 "Context: Structured JSON array of retrieved chunks follows. Each chunk is a dict: "
#                 " {chunk_id, pdf_name, page, section_type, score, content}.\n\n"
#                 f"{chunk_context}"
#             ),
#         },
#     ]

#     # ---------------- Robust OpenAI invocation ----------------
#     resp = None
#     answer = None
#     try:
#         # Try to use modern OpenAI client if present
#         try:
#             from openai import OpenAI  # modern 1.x client
#             client = OpenAI(api_key=OPENAI_API_KEY)
#             resp = client.chat.completions.create(
#                 model=MODEL,
#                 messages=messages,
#                 temperature=0.1,
#                 max_tokens=700
#             )
#             # Try extraction in several possible shapes
#             try:
#                 answer = resp["choices"][0]["message"]["content"]
#             except Exception:
#                 try:
#                     answer = resp.choices[0].message.content
#                 except Exception:
#                     answer = None

#         except Exception as client_exc:
#             # If the modern client failed (e.g. proxies/unexpected arg), fall back to direct HTTP call
#             # or the legacy openai API depending on what's available.
#             # Try legacy openai module if present and supports ChatCompletion
#             try:
#                 import openai as openai_legacy
#                 ver = getattr(openai_legacy, "__version__", "")
#                 # If legacy (version < 1.0) and exposes ChatCompletion, use it
#                 if ver and int(ver.split(".")[0]) < 1 and hasattr(openai_legacy, "ChatCompletion"):
#                     openai_legacy.api_key = OPENAI_API_KEY
#                     resp = openai_legacy.ChatCompletion.create(
#                         model=MODEL, messages=messages, temperature=0.1, max_tokens=700
#                     )
#                     try:
#                         answer = resp["choices"][0]["message"]["content"]
#                     except Exception:
#                         answer = getattr(resp.choices[0].message, "content", None)
#                 else:
#                     # Final fallback: call the REST API directly using requests (no openai client)
#                     import requests
#                     url = "https://api.openai.com/v1/chat/completions"
#                     headers = {
#                         "Authorization": f"Bearer {OPENAI_API_KEY}",
#                         "Content-Type": "application/json",
#                     }
#                     payload = {
#                         "model": MODEL,
#                         "messages": messages,
#                         "temperature": 0.1,
#                         "max_tokens": 700,
#                     }
#                     r = requests.post(url, headers=headers, json=payload, timeout=120)
#                     try:
#                         rj = r.json()
#                     except Exception as rexc:
#                         raise RuntimeError(f"OpenAI HTTP call failed to return JSON: {rexc}\nStatus:{r.status_code}\nText:{r.text}")
#                     # store resp-like object for debugging later
#                     resp = rj
#                     # extract
#                     try:
#                         answer = rj["choices"][0]["message"]["content"]
#                     except Exception:
#                         answer = None
#             except Exception as fallback_exc:
#                 # bubble up an informative error containing both attempts
#                 raise RuntimeError(f"OpenAI client failure: {client_exc}\nFallback attempt failed: {fallback_exc}")

#     except Exception as e:
#         # surface as runtime error with details so UI shows it
#         raise RuntimeError(f"OpenAI invocation failed: {e}")

#     # Validate we have an answer
#     if not answer:
#         raise RuntimeError(f"OpenAI returned unexpected response shape or empty answer. Raw response: {resp}")

#     # ---------------- Add source citation (file names without extensions) ----------------
#     pdf_names = sorted(set([
#         Path(c["pdf_name"]).stem for c in merged_chunks if c.get("pdf_name")
#     ]))
#     if pdf_names:
#         citations = "\n\n**Sources Referenced:** " + ", ".join(pdf_names)
#         answer = answer.strip() + citations

#     return answer


# # ---------------- Streamlit UI ----------------
# st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

# tabs = st.tabs(["📂 Processed Files", "💬 Query Interface"])

# # --- Tab 1: Processed Files ---
# with tabs[0]:
#     st.header("📂 Processed Files (Available for Querying)")
#     st.markdown("""
#     These are the **processed files** currently available for querying through the RAG system.
#     > ⚙️ *Due to hardware constraints, only these datasets are currently pre-embedded.*
#     """)

#     st.subheader("📘 Source Files (Text Content)")
#     st.markdown("""
#     - APR & MAY  
#     - APR & MAY__dup1  
#     - APR & MAY__dup2  
#     - APR & MAY__dup3  
#     - APR TO AUG  
#     - APR TO AUG__dup1  
#     - APR TO AUG__dup2  
#     - APR TO JAN__dup1  
#     - Apr to June 20  
#     - Apr-20 to OCT-20  
#     - April 2020 To March 2021  
#     - BD 26 POSHS April To March-2021  
#     - BD 26 POSHS METAL INDUSTRIES PVT LTD April To Feb 2021  
#     - bd 26 apr to dec  
#     - Jan to Feb 21  
#     - July to Sept 20  
#     - Nov to Dec 20  
#     - Oct 20  
#     - POSHS BD 26 April To Jan 2021  
#     - POSHS METAL INDUSTRIES PVT LTD April To Feb 2021
#     """)

#     st.subheader("📊 Source Files (Table Summaries)")
#     st.markdown("""
#     - Axis Bank Statement-April-20 to Oct-20  
#     - CHEMBUR FY 20-21  
#     - CIBIL Score_PMIPL_As on Nov 2020  
#     - Poshs Tax Audit 2015-16_p32  
#     - Final rejection Summary sheet SPC VSM  
#     - MIS-Poshs Metal-Mar-21
#     """)

#     st.markdown("---")
#     st.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

# # --- Tab 2: Main Query Interface ---
# with tabs[1]:
#     st.header(" RAG Query Interface")
#     st.sidebar.title("Guidelines")

#     st.sidebar.subheader("[*] Do’s")
#     st.sidebar.markdown("""
#     - Keep queries **specific**.  
#     - Choose correct retrieval scope.  
#     - Verify results using sources.
#     """)

#     st.sidebar.subheader("[*] Don’ts")
#     st.sidebar.markdown("""
#     - Don’t query unprocessed files.  
#     - Don’t expect live data or updates.  
#     - Avoid vague questions.
#     """)

#     st.sidebar.markdown("---")
#     st.sidebar.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

#     # --- Main layout for query input ---
#     col1, col2 = st.columns([4, 2])
#     with col1:
#         query = st.text_area("Enter your query:", placeholder="e.g. Provide a summary of outstanding loans and their repayment schedules.")
#     with col2:
#         scope_option = st.selectbox("Retrieval Scope", ["Text + Tables", "Tables Only"], index=0)
#         run_button = st.button("Run Query")

#     if run_button and query.strip():
#         with st.spinner("Running RAG pipeline..."):
#             scope = "tables" if scope_option == "Tables Only" else "all"
#             try:
#                 answer = run_rag_pipeline(query.strip(), scope=scope)
#                 st.success("✅ Query processed successfully.")
#                 st.markdown("### 🧠 Response")
#                 st.write(answer)
#             except Exception as e:
#                 st.error(f"❌ Error: {e}")

#!/usr/bin/env python3
"""
Streamlit RAG Interface (Tables / Text + Tables selector)
Final Version — Creator: Harsh Chinchakar
"""

import os, sys, json, re, shlex, subprocess, shutil
from pathlib import Path
from datetime import datetime
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv

# ---------------- Load Config ----------------
load_dotenv(".env")
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_RETRIEVAL_SCRIPT = str(BASE_DIR / "Retrival" / "retrieval_combined_v2.py")
TABLES_ONLY_SCRIPT = str(BASE_DIR / "Retrival" / "retrival_tables.py")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"
SEMANTIC_TOP = 12
KEYWORD_TOP = 12
CHUNK_CHAR_LIMIT = 2500
TIMEOUT_SECONDS = 120

# ---------------- Helper Functions ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def run_retrieval(query: str, retrieval_script: str):
    retrieval_script = str(retrieval_script)
    if not os.path.isfile(retrieval_script):
        raise RuntimeError(f"Retrieval script not found: {retrieval_script!r}")

    python_exec = sys.executable or shutil.which("python3") or "python3"
    top_k = max(SEMANTIC_TOP, KEYWORD_TOP)
    cmd_list = [python_exec, retrieval_script, "--query", query, "--top_k", str(top_k)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR) + (":" + env.get("PYTHONPATH", ""))

    proc = subprocess.run(
        cmd_list,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        cwd=str(BASE_DIR),
        env=env,
        shell=False,
    )

    stdout, stderr = proc.stdout or "", proc.stderr or ""

    parsed = None
    try:
        m = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", stdout)
        if not m:
            m = re.search(r"(\{[\s\S]+\})", stdout)
        if m:
            parsed = json.loads(m.group(1))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from retrieval stdout: {e}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

    if isinstance(parsed, dict):
        has_semantic = bool(parsed.get("semantic", {}).get("results"))
        has_keyword = bool(parsed.get("keyword", {}).get("results"))
        if has_semantic or has_keyword:
            return parsed
        if parsed.get("error") and not (has_semantic or has_keyword):
            raise RuntimeError(f"Retrieval script returned an error payload: {parsed.get('error')}")
        return parsed

    raise RuntimeError(f"Retrieval script failed.\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

def is_table_chunk(item: Dict[str, Any]) -> bool:
    stype = (item.get("section_type") or "").lower()
    if "table" in stype or "table_summary" in stype:
        return True
    if item.get("table_id") or item.get("table_part_index"):
        return True
    pdf = (item.get("pdf_name") or "").lower()
    if pdf.endswith(".xlsx") or pdf.endswith(".xls"):
        return True
    content = (item.get("content") or "")[:500]
    if "\t" in content or " | " in content or content.strip().startswith("|"):
        return True
    return False

def dedupe_and_merge(semantic, keyword):
    combined, seen = [], set()
    for src in (semantic or [])[:SEMANTIC_TOP] + (keyword or [])[:KEYWORD_TOP]:
        cid = src.get("chunk_id") or f"{src.get('pdf_name')}::p{src.get('page')}"
        if cid in seen:
            continue
        seen.add(cid)
        combined.append({
            "chunk_id": cid,
            "pdf_name": src.get("pdf_name") or "<unknown>",
            "page": src.get("page"),
            "section_type": src.get("section_type"),
            "score": src.get("score", None),
            "content": (src.get("content") or "")[:CHUNK_CHAR_LIMIT],
        })
    return combined

# ---------------- Core RAG Logic ----------------
def run_rag_pipeline(query, scope="tables"):
    """Main RAG logic (retrieval + OpenAI structured query)."""
    if scope == "tables":
        retrieval_script = TABLES_ONLY_SCRIPT
        apply_filter = False
    else:
        retrieval_script = DEFAULT_RETRIEVAL_SCRIPT
        apply_filter = True

    retrieval = run_retrieval(query, retrieval_script)
    sem_results, kw_results = retrieval.get("semantic", {}).get("results", []), retrieval.get("keyword", {}).get("results", [])

    def filter_scope(items):
        if not apply_filter:
            return items
        if scope == "text":
            return [i for i in items if not is_table_chunk(i)]
        elif scope == "tables":
            return [i for i in items if is_table_chunk(i)]
        return items

    merged_chunks = dedupe_and_merge(filter_scope(sem_results), filter_scope(kw_results))
    if not merged_chunks:
        return {"answer": "No relevant chunks found for this query."}

    system_prompt = (
        "Rules (obey strictly):\n"
        "1) Use ONLY retrieved chunks. Prefer direct evidence but infer carefully.\n"
        "2) Cite each statement with [chunk_id | pdf_name].\n"
        "3) Output concise, formal answers with a 'Sources' list.\n"
    )

    chunk_context = json.dumps(merged_chunks, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\nContext (JSON chunks):\n{chunk_context}"},
    ]

    # ---- Robust OpenAI call ----
    import requests
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages, "temperature": 0.1, "max_tokens": 700}
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    rj = r.json()
    answer = rj.get("choices", [{}])[0].get("message", {}).get("content", "")

    pdf_names = sorted(set([Path(c["pdf_name"]).stem for c in merged_chunks if c.get("pdf_name")]))
    if pdf_names:
        answer += "\n\n**Sources Referenced:** " + ", ".join(pdf_names)
    return answer


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

tabs = st.tabs(["📘 Documentation", "🧩 Expected Questions & Files", "💬 Query Interface"])

# --- Tab 1: Documentation ---
with tabs[0]:
    st.header("📘 Air-Gapped RAG System — Documentation")
    st.markdown("""
    ### 🔐 Overview
    This Proof of Concept (POC) demonstrates a **locally processed RAG (Retrieval-Augmented Generation)** system —
    built to mimic **air-gapped environments** where external network access is restricted.

    **Key Highlights**
    - ⚙️ **Offline-first processing**: All document parsing, OCR, and embedding generation are done locally.
    - 🧠 **Structured retrieval**: Supports hybrid retrieval (semantic + keyword) for both text and tabular data.
    - 🔒 **Data isolation**: The system can run in an air-gapped setup with zero external API calls for retrieval.
    - 📄 **Tabular intelligence**: Advanced parsing and structured citation of tables to enhance factual accuracy.
    - 🧩 **Context precision**: Combines FAISS + BM25-based ranking for maximum retrieval accuracy.
    - 🧠 **Inference + citation-aware LLM responses** for auditable outputs.

    ---
    ### 🧱 Architectural Abstract
    ```
    [Document Store] → [Text + Table Extraction] → [Chunking + Embedding] 
       → [Hybrid Retrieval Engine] → [RAG Pipeline] → [Cited, Verified Response]
    ```
    The architecture emphasizes **traceable information flow** and **highly contextual answer synthesis**.

    ---
    ### ⚡ Why This Approach?
    - ✅ Suitable for **enterprise-grade isolated data systems**
    - ✅ **Zero dependency** on live cloud retrieval
    - ✅ Enables **structured tabular reasoning** within natural language responses
    - ✅ Extensible to multimodal document understanding
    """)

# --- Tab 2: Expected Questions & Files Processed ---
with tabs[1]:
    st.header("🧩 Expected Questions & Files Processed")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Expected / Recommended Questions")
        st.markdown("""
        - What are the total outstanding liabilities as of Q3 FY20-21?  
        - Provide a summarized financial overview from April to October 2020.  
        - Extract and summarize all loan-related entries.  
        - What are the monthly expense trends across departments?  
        - Are there any notable CIBIL score changes across entities?  
        - Generate a short summary of audit notes from FY 2020-21.  
        - Identify key vendors from financial tables and summaries.
        """)

    with col2:
        st.subheader("📊 Files Processed (with Limitations)")
        st.markdown("""
        Due to **air-gapped simulation constraints**, only a **limited corpus** is embedded and searchable.
        This ensures **faster, localized retrieval** without external API dependency.

        **Processed Files Include:**
        - APR & MAY series  
        - APR TO AUG and JAN to FEB documents  
        - POSHS and BD 26 series (2020–21)  
        - CIBIL Score PMIPL (Nov 2020)  
        - Audit & MIS summaries  
        - Axis Bank Statement (Apr–Oct 2020)  
        - CHEMBUR FY 20-21 and related summaries  
        """)

        st.info("🔗 For detailed chunk-level view, visit: [Chunk Visualizer App](https://air-gapped-system-poc-qdvatkypzyappztfxsaypgq.streamlit.app)")

# --- Tab 3: Query Interface ---
with tabs[2]:
    st.header("💬 RAG Query Interface")
    st.sidebar.title("Guidelines")
    st.sidebar.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

    st.sidebar.subheader("[*] Do’s")
    st.sidebar.markdown("- Keep queries **specific**.\n- Choose correct retrieval scope.\n- Verify results using sources.")

    st.sidebar.subheader("[*] Don’ts")
    st.sidebar.markdown("- Don’t query unprocessed files.\n- Don’t expect live updates.\n- Avoid vague questions.")

    col1, col2 = st.columns([4, 2])
    with col1:
        query = st.text_area("Enter your query:", placeholder="e.g. Provide a summary of outstanding loans and repayment schedules.")
    with col2:
        scope_option = st.selectbox("Retrieval Scope", ["Text + Tables", "Tables Only"], index=0)
        run_button = st.button("Run Query")

    if run_button and query.strip():
        with st.spinner("Running RAG pipeline..."):
            try:
                scope = "tables" if scope_option == "Tables Only" else "all"
                answer = run_rag_pipeline(query.strip(), scope=scope)
                st.success("✅ Query processed successfully.")
                st.markdown("### 🧠 Response")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")
    elif run_button:
        st.warning("Please enter a query before running.")

# # --- Tab 4: Processed Files (Original) ---
# with tabs[3]:
#     st.header("📂 Processed Files (Available for Querying)")
#     st.markdown("""
#     These are the **processed files** currently available for querying through the RAG system.
#     > ⚙️ *Due to hardware constraints, only these datasets are currently pre-embedded.*
#     """)
#     st.subheader("📘 Text Content Files")
#     st.markdown("""
#     - APR & MAY  
#     - APR TO AUG  
#     - APR TO JAN__dup1  
#     - April 2020 To March 2021  
#     - POSHS METAL INDUSTRIES PVT LTD April To Feb 2021  
#     - BD 26 POSHS METAL INDUSTRIES PVT LTD April To Feb 2021  
#     """)
#     st.subheader("📊 Table Summary Files")
#     st.markdown("""
#     - Axis Bank Statement (Apr–Oct 2020)  
#     - CHEMBUR FY 20-21  
#     - CIBIL Score_PMIPL_As on Nov 2020  
#     - Final rejection Summary sheet SPC VSM  
#     - MIS-Poshs Metal-Mar-21  
#     """)
#     st.markdown("---")
#     st.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

#     elif run_button:
#         st.warning("Please enter a query before running.")


