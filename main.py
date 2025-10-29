

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
    """
    Run retrieval script via subprocess using the same Python interpreter / venv.
    Returns parsed dict if successful or if the subprocess printed valid retrieval JSON
    (even when the process exit code != 0). Raises RuntimeError only when no usable
    JSON/chunks are found.

    This is defensive to handle cases where the retrieval subprocess prints dependency
    warnings/errors AFTER producing usable chunk JSON (or prints an 'error' JSON).
    """
    retrieval_script = str(retrieval_script)
    if not os.path.isfile(retrieval_script):
        raise RuntimeError(f"Retrieval script not found: {retrieval_script!r}")

    python_exec = sys.executable or shutil.which("python3") or "python3"
    top_k = max(SEMANTIC_TOP, KEYWORD_TOP)
    cmd_list = [python_exec, retrieval_script, "--query", query, "--top_k", str(top_k)]

    # Ensure subprocess uses same env and can import local modules
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(BASE_DIR) + (":" + existing_pp if existing_pp else "")

    proc = subprocess.run(
        cmd_list,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        cwd=str(BASE_DIR),
        env=env,
        shell=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Try to extract JSON (primary marker then fallback)
    parsed = None
    try:
        m = re.search(r"RETRIEVAL_JSON_OUTPUT:\s*(\{[\s\S]+\})", stdout)
        if not m:
            m = re.search(r"(\{[\s\S]+\})", stdout)
        if m:
            parsed = json.loads(m.group(1))
    except Exception as e:
        # JSON exists but parse failed — treat as fatal, include stdout for debug
        raise RuntimeError(f"Failed to parse JSON from retrieval stdout: {e}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

    # If we got parsed JSON, check if it contains usable retrieval chunks
    if isinstance(parsed, dict):
        # If parsed has semantic/keyword results, return it even if rc != 0
        has_semantic = bool(parsed.get("semantic", {}).get("results"))
        has_keyword = bool(parsed.get("keyword", {}).get("results"))
        if has_semantic or has_keyword:
            return parsed

        # If parsed contains an 'error' payload only, surface it as an error
        if parsed.get("error") and not (has_semantic or has_keyword):
            hint = ""
            err_text = str(parsed.get("error") or "")
            if "huggingface_hub" in err_text or "cached_download" in err_text:
                hint = "\nHint: the retrieval script reports a huggingface_hub version mismatch. Try upgrading: `pip install --upgrade huggingface_hub sentence-transformers numpy` in the venv used by Streamlit."
            raise RuntimeError(f"Retrieval script returned an error payload: {err_text}{hint}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

        # parsed JSON present but no semantic/keyword -> return parsed (caller can decide)
        return parsed

    # No JSON parsed
    # If subprocess exited cleanly but printed no JSON -> fatal
    if proc.returncode == 0:
        raise RuntimeError(f"Retrieval subprocess completed (rc=0) but produced no JSON.\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

    # proc.returncode != 0 and no JSON -> include hints for common dependency issues
    # inspect stdout/stderr for known dependency messages
    combined = stdout + "\n\n" + stderr
    if "No module named" in combined or "cannot import name" in combined or "Missing dependencies" in combined:
        suggestion = (
            "\nSuggested fix: ensure the retrieval subprocess uses the same Python venv as Streamlit. "
            "On the host, run `which python3` and `pip install sentence-transformers faiss-cpu rank_bm25 numpy huggingface_hub` "
            "inside that environment. If you use virtualenv/venv, make sure sys.executable points to it."
        )
    else:
        suggestion = ""

    raise RuntimeError(
        f"Retrieval script failed (rc={proc.returncode}) and no JSON could be parsed.\n\nCMD: {' '.join(cmd_list)}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n{suggestion}"
    )


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
    """Main RAG logic — unchanged from the CLI pipeline (only OpenAI call + structured context changed)."""
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

    # ---------------- OpenAI call (stable usage + structured chunk context) ----------------
    import openai

    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY in st.secrets or your environment.")

    # Use module-level API (compatible across versions and respects HTTP(S)_PROXY env vars)
    openai.api_key = OPENAI_API_KEY

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

    # Send the chunks as a structured JSON array (list of dicts) so the model receives structured context
    # Each chunk already has: chunk_id, pdf_name, page, section_type, score, content (truncated)
    chunk_context = json.dumps(merged_chunks, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User query: {query}\n\n"
                "Context: Structured JSON array of retrieved chunks follows. Each chunk is a dict: "
                " {chunk_id, pdf_name, page, section_type, score, content}.\n\n"
                f"{chunk_context}"
            ),
        },
    ]

    # Classic ChatCompletion call — widely compatible
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=700
    )

    # Extract content robustly
    try:
        answer = resp["choices"][0]["message"]["content"]
    except Exception:
        answer = getattr(resp.choices[0].message, "content", None)
        if answer is None:
            raise RuntimeError(f"Unexpected OpenAI response shape: {resp}")

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


