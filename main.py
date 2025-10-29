# #!/usr/bin/env python3
# """
# Streamlit RAG Interface (Tables / Text + Tables selector)
# Author: Harsh Chinchakar

# Pages:
# 1. 📂 Processed Files — shows available files (placeholder)
# 2. 💬 Query Interface — run full RAG pipeline with scope options and display response

# All original retrieval and RAG pipeline logic remains intact.
# """

# import os, sys, json, re, shlex, subprocess
# from pathlib import Path
# from datetime import datetime
# import streamlit as st
# from typing import List, Dict, Any

# # ---------------- CONFIG ----------------
# DEFAULT_RETRIEVAL_SCRIPT = "./Retrival/retrieval_combined_v2.py"
# TABLES_ONLY_SCRIPT = "./Retrival/retrival_tables.py"
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# MODEL = "gpt-4o-mini"
# SEMANTIC_TOP = 8
# KEYWORD_TOP = 8
# CHUNK_CHAR_LIMIT = 2500
# TIMEOUT_SECONDS = 120

# # ---------------- Helper Functions ----------------
# def now_iso():
#     return datetime.utcnow().isoformat() + "Z"

# def run_retrieval(query: str, retrieval_script: str):
#     """Executes the retrieval script via subprocess and returns parsed JSON output."""
#     cmd = f"python3 {shlex.quote(retrieval_script)} --query {shlex.quote(query)} --top_k {max(SEMANTIC_TOP, KEYWORD_TOP)}"
#     proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
#     if proc.returncode != 0:
#         raise RuntimeError(f"Retrieval script failed: {proc.stderr}")

#     match = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
#     if not match:
#         raise RuntimeError("No valid JSON found in retrieval output.")
#     return json.loads(match.group(1))

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
#     """Main RAG logic — unchanged from the CLI pipeline."""
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

#     # Prepare OpenAI call
#     from openai import OpenAI
#     client = OpenAI(api_key=OPENAI_API_KEY)

#     system_prompt = (
#         "You are a formal company assistant answering from retrieved document chunks.\n"
#         "Use only provided data. If inferred, mark as 'Inferred'. Cite chunks as [chunk_id | pdf_name]."
#     )
#     chunk_context = "\n".join([
#         json.dumps({
#             "chunk_id": c["chunk_id"],
#             "pdf_name": c["pdf_name"],
#             "page": c["page"],
#             "content": c["content"]
#         }, ensure_ascii=False)
#         for c in merged_chunks
#     ])

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": f"User query: {query}\n\nContext:\n{chunk_context}"}
#     ]

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=messages,
#         temperature=0.1,
#         max_tokens=700
#     )
#     return response.choices[0].message.content

# # ---------------- Streamlit UI ----------------
# st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

# tabs = st.tabs(["📂 Processed Files", "💬 Query Interface"])

# # --- Tab 1: Processed Files ---
# with tabs[0]:
#     st.header("📂 Processed Files (Available for Querying)")
#     st.markdown("""
#     These are the **processed files** currently available for querying through the RAG system.
#     > ⚙️ *Due to hardware constraints, only these datasets are currently pre-embedded.*

#     **Available Files**
#     - `Final rejection Summary sheet SPC  VSM.xlsx`
#     - `CIBIL Score_PMIPL_As on Nov 2020.pdf`
#     - `MIS Report_Q2 FY21.pdf`
#     - *(more files can be added later)*
#     """)
#     st.markdown("---")
#     st.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

# # --- Tab 2: Main Query Interface ---
# with tabs[1]:
#     st.header("💬 RAG Query Interface")
#     st.sidebar.title("Settings & Guidelines")

#     st.sidebar.subheader("✅ Do’s")
#     st.sidebar.markdown("""
#     - Keep queries **clear and specific**.  
#     - Select appropriate **retrieval scope** (Tables / Text + Tables).  
#     - Review cited sources for verification.
#     """)

#     st.sidebar.subheader("❌ Don’ts")
#     st.sidebar.markdown("""
#     - Don’t ask for data outside the processed documents.  
#     - Don’t expect real-time web or database access.  
#     - Avoid vague queries (“give everything” etc.).
#     """)

#     st.sidebar.markdown("---")
#     scope_option = st.sidebar.radio(
#         "Select Retrieval Scope:",
#         ["Tables Only", "Text + Tables"],
#         index=0
#     )

#     st.sidebar.markdown("---")
#     st.sidebar.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

#     query = st.text_area("🔍 Enter your query:", placeholder="e.g., Compare budgeted vs. actual MIS figures for Q2 FY21")
#     run_button = st.button("Run Query")

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

#     elif run_button:
#         st.warning("Please enter a query before running.")

#!/usr/bin/env python3
"""
Streamlit RAG Interface (Tables / Text + Tables selector)
Final Version — Creator: Harsh Chinchakar
"""

import os, sys, json, re, shlex, subprocess
from pathlib import Path
from datetime import datetime
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load from your .env file
load_dotenv(".env")
BASE_DIR = Path(__file__).resolve().parent
# # ---------------- CONFIG ----------------
# DEFAULT_RETRIEVAL_SCRIPT = "Retrival/retrieval_combined_v2.py"
# TABLES_ONLY_SCRIPT = "Retrival/retrival_tables.py"
DEFAULT_RETRIEVAL_SCRIPT = str(BASE_DIR / "Retrival" / "retrieval_combined_v2.py")
TABLES_ONLY_SCRIPT      = str(BASE_DIR / "Retrival" / "retrival_tables.py") 
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"
SEMANTIC_TOP = 12
KEYWORD_TOP = 12
CHUNK_CHAR_LIMIT = 2500
TIMEOUT_SECONDS = 120

# ---------------- Helper Functions ----------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

# def run_retrieval(query: str, retrieval_script: str):
#     """Executes the retrieval script via subprocess and returns parsed JSON output."""
#     cmd = f"python3 {shlex.quote(retrieval_script)} --query {shlex.quote(query)} --top_k {max(SEMANTIC_TOP, KEYWORD_TOP)}"
#     proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
#     if proc.returncode != 0:
#         raise RuntimeError(f"Retrieval script failed: {proc.stderr}")

#     match = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", proc.stdout)
#     if not match:
#         raise RuntimeError("No valid JSON found in retrieval output.")
#     return json.loads(match.group(1))

import shutil

def run_retrieval(query: str, retrieval_script: str):
    """Runs the retrieval script via subprocess and returns parsed JSON output.
    Returns a dict on success, raises RuntimeError with helpful message on failure.
    """
    # Ensure we have a string path
    retrieval_script = str(retrieval_script)

    # Quick existence check (helps on hosted platforms)
    if not os.path.isfile(retrieval_script):
        raise RuntimeError(f"Retrieval script not found: {retrieval_script!r}")

    # Build safe command
    python_exec = shutil.which("python3") or sys.executable or "python3"
    cmd = f"{python_exec} {shlex.quote(retrieval_script)} --query {shlex.quote(query)} --top_k {max(SEMANTIC_TOP, KEYWORD_TOP)}"

    # Run subprocess
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, cwd=str(BASE_DIR))

    # Always include stdout/stderr in the error path for quick debugging
    if proc.returncode != 0:
        raise RuntimeError(
            f"Retrieval script failed (rc={proc.returncode}).\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    stdout = proc.stdout or ""
    # Safe check before using re on stdout
    if not isinstance(stdout, (str, bytes)):
        raise RuntimeError("Retrieval subprocess produced non-text stdout")

    # Try primary marker first
    m = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", stdout)
    if not m:
        # fallback: first JSON object in output
        m2 = re.search(r"(\{[\s\S]+\})", stdout)
        m = m2

    if not m:
        # include stdout to help debugging
        raise RuntimeError(f"No JSON found in retrieval stdout. Full stdout:\n{stdout}\n\nStderr:\n{proc.stderr}")

    # parse json (this can raise JSONDecodeError which will bubble up)
    parsed = json.loads(m.group(1))
    return parsed


def is_table_chunk(item: Dict[str, Any]) -> bool:
    st = (item.get("section_type") or "").lower()
    if "table" in st or "table_summary" in st:
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
        content = (src.get("content") or "")[:CHUNK_CHAR_LIMIT]
        combined.append({
            "chunk_id": cid,
            "pdf_name": src.get("pdf_name") or "<unknown>",
            "page": src.get("page"),
            "section_type": src.get("section_type"),
            "score": src.get("score", None),
            "content": content
        })
    return combined

# ---------------- Core RAG Logic ----------------
def run_rag_pipeline(query, scope="tables"):
    """Main RAG logic — unchanged from the CLI pipeline."""
    if scope == "tables":
        retrieval_script = TABLES_ONLY_SCRIPT
        apply_filter = False
    else:
        retrieval_script = DEFAULT_RETRIEVAL_SCRIPT
        apply_filter = True

    retrieval = run_retrieval(query, retrieval_script)
    sem_results = retrieval.get("semantic", {}).get("results", [])
    kw_results = retrieval.get("keyword", {}).get("results", [])

    def filter_scope(items):
        if not apply_filter:
            return items
        if scope == "text":
            return [i for i in items if not is_table_chunk(i)]
        elif scope == "tables":
            return [i for i in items if is_table_chunk(i)]
        return items

    sem_filtered, kw_filtered = filter_scope(sem_results), filter_scope(kw_results)
    merged_chunks = dedupe_and_merge(sem_filtered, kw_filtered)

    if not merged_chunks:
        return {"answer": "No relevant chunks found for this query."}

    # Prepare OpenAI call
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "Rules (obey strictly):\n"
        "1) Use ONLY the information present in the provided retrieved chunks to answer. Prefer direct evidence but if indirect evidence is present use to or process it towards maximum bound and infer required facts without Generating facts\n"
        "2) If the exact fact is not directly stated, you MAY synthesize or infer an answer based on INDIRECT or PARTIAL evidence found across chunks. "
        "When you do so, explicitly mark the statement as 'Inferred' and cite the supporting chunks for that inference.\n"
        "3) Only respond with: \"Information not found in retrieved dataset.\" when there is absolutely no relevant information or inference possible from the provided chunks.\n"
        "4) Cite each factual statement with the chunk citation format: [chunk_id | pdf_name]. If multiple chunks support it, cite all.\n"
        "5) Produce a concise, formal natural-language answer; follow with a 'Sources' list (chunk citations). "
        "Also include a short 'Limitations' note if any inferred conclusions were used.\n"
        "When the required data isnt completely present then focus on the present data in the retrieved chunks and frame a short answer towards that"
        "(7) When numbers are not present in the chunks -DO NOT ANSWER WITH NUMERIC ANSWERS - State that information is not present but this is what we found"
    )
    chunk_context = "\n".join([
        json.dumps({
            "chunk_id": c["chunk_id"],
            "pdf_name": c["pdf_name"],
            "page": c["page"],
            "content": c["content"]
        }, ensure_ascii=False)
        for c in merged_chunks
    ])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User query: {query}\n\nContext:\n{chunk_context}"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=700
    )
    answer = response.choices[0].message.content

    # 🔹 Add source citation (file names without extensions)
    pdf_names = sorted(set([
        Path(c["pdf_name"]).stem for c in merged_chunks if c.get("pdf_name")
    ]))
    if pdf_names:
        citations = "\n\n**Sources Referenced:** " + ", ".join(pdf_names)
        answer = answer.strip() + citations

    return answer

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

tabs = st.tabs(["📂 Processed Files", "💬 Query Interface"])

# --- Tab 1: Processed Files ---
with tabs[0]:
    st.header("📂 Processed Files (Available for Querying)")
    st.markdown("""
    These are the **processed files** currently available for querying through the RAG system.
    > ⚙️ *Due to hardware constraints, only these datasets are currently pre-embedded.*
    """)

    st.subheader("📘 Source Files (Text Content)")
    st.markdown("""
    - APR & MAY  
    - APR & MAY__dup1  
    - APR & MAY__dup2  
    - APR & MAY__dup3  
    - APR TO AUG  
    - APR TO AUG__dup1  
    - APR TO AUG__dup2  
    - APR TO JAN__dup1  
    - Apr to June 20  
    - Apr-20 to OCT-20  
    - April 2020 To March 2021  
    - BD 26 POSHS April To March-2021  
    - BD 26 POSHS METAL INDUSTRIES PVT LTD April To Feb 2021  
    - bd 26 apr to dec  
    - Jan to Feb 21  
    - July to Sept 20  
    - Nov to Dec 20  
    - Oct 20  
    - POSHS BD 26 April To Jan 2021  
    - POSHS METAL INDUSTRIES PVT LTD April To Feb 2021
    """)

    st.subheader("📊 Source Files (Table Summaries)")
    st.markdown("""
    - Axis Bank Statement-April-20 to Oct-20  
    - CHEMBUR FY 20-21  
    - CIBIL Score_PMIPL_As on Nov 2020  
    - Poshs Tax Audit 2015-16_p32  
    - Final rejection Summary sheet SPC VSM  
    - MIS-Poshs Metal-Mar-21
    """)

    st.markdown("---")
    st.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

# --- Tab 2: Main Query Interface ---
with tabs[1]:
    st.header(" RAG Query Interface")
    st.sidebar.title("Guidelines")

    st.sidebar.subheader("[*] Do’s")
    st.sidebar.markdown("""
    - Keep queries **specific**.  
    - Choose correct retrieval scope.  
    - Verify results using sources.
    """)

    st.sidebar.subheader("[*] Don’ts")
    st.sidebar.markdown("""
    - Don’t query unprocessed files.  
    - Don’t expect live data or updates.  
    - Avoid vague questions.
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 **Creator — Harsh Chinchakar**")

    # --- Main layout for query input ---
    col1, col2 = st.columns([4, 2])
    with col1:
        query = st.text_area("Enter your query:", placeholder="e.g. Provide a summary of outstanding loans and their repayment schedules.")
    with col2:
        scope_option = st.selectbox("Retrieval Scope", ["Text + Tables", "Tables Only"], index=0)
        run_button = st.button("Run Query")

    if run_button and query.strip():
        with st.spinner("Running RAG pipeline..."):
            scope = "tables" if scope_option == "Tables Only" else "all"
            try:
                answer = run_rag_pipeline(query.strip(), scope=scope)
                st.success("✅ Query processed successfully.")
                st.markdown("### 🧠 Response")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif run_button:
        st.warning("Please enter a query before running.")
