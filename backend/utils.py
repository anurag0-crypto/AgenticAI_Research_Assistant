"""
utils.py — small, boring helper functions shared by graph.py, tools.py and main.py.

Keeping these in one place means graph.py can stay focused on *agent logic*
instead of string plumbing.
"""

import io
import json
import re


# --------------------------------------------------------- LLM JSON parsing --

def safe_json_extract(text: str) -> dict:
    """Best-effort parse of an LLM's "respond with ONLY a JSON object" reply.

    Strips markdown code fences if the model added them anyway, and falls
    back to grabbing the first {...} block if the whole string isn't valid
    JSON on its own. Returns {} if nothing usable is found, so callers should
    always provide sensible .get() defaults rather than assuming keys exist.
    """
    if not text:
        return {}
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
    return {}


# ------------------------------------------------- live "field log" wording --

_ICONS = {
    "web_search": "🔎",
    "arxiv_search": "📄",
    "calculator": "🧮",
    "knowledge_base_search": "📚",
    "save_memory": "📌",
    "recall_memory": "🗂️",
    "generate_pdf_report": "🖨️",
}


def describe_call(call: dict) -> str:
    """Human-readable line shown in the UI's live log the moment a tool is called."""
    name = call.get("name", "")
    args = call.get("args", {}) or {}
    icon = _ICONS.get(name, "⚙️")
    if name == "web_search":
        return f"{icon} Searching the web for \u201c{args.get('query', '')}\u201d"
    if name == "arxiv_search":
        return f"{icon} Checking arXiv for \u201c{args.get('query', '')}\u201d"
    if name == "calculator":
        return f"{icon} Calculating: {args.get('expression', '')}"
    if name == "knowledge_base_search":
        return f"{icon} Searching your uploaded documents for \u201c{args.get('query', '')}\u201d"
    if name == "save_memory":
        return f"{icon} Filing a note: \u201c{str(args.get('fact', ''))[:80]}\u201d"
    if name == "recall_memory":
        return f"{icon} Recalling past notes about \u201c{args.get('query', '')}\u201d"
    if name == "generate_pdf_report":
        return f"{icon} Drafting a PDF report: \u201c{args.get('title', '')}\u201d"
    return f"{icon} Calling {name}"


def summarize_result(name: str, result) -> str:
    """Human-readable line shown right after a tool call returns."""
    text = str(result)
    icon = _ICONS.get(name, "✅")
    if name in ("web_search", "arxiv_search", "knowledge_base_search"):
        n = len([ln for ln in text.splitlines() if " :: " in ln])
        return f"{icon} Found {n} result(s)" if n else f"{icon} No results found"
    if name == "calculator":
        return f"{icon} {text}"
    if name == "save_memory":
        return f"{icon} Saved to long-term memory"
    if name == "recall_memory":
        n = 0 if "No relevant" in text else len([ln for ln in text.splitlines() if ln.strip()])
        return f"{icon} Recalled {n} note(s)" if n else f"{icon} Nothing relevant saved yet"
    if name == "generate_pdf_report":
        return f"{icon} Report ready"
    return f"{icon} Done"


def extract_sources(text) -> list:
    """Pull structured {title, url} entries out of a tool's 'TITLE :: URL :: snippet' lines."""
    out = []
    for line in str(text).splitlines():
        parts = line.split(" :: ")
        if len(parts) >= 2 and parts[1].strip().startswith("http"):
            out.append({"title": parts[0].strip(), "url": parts[1].strip()})
    return out


# --------------------------------------------------------- document ingest --

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list:
    """Split text into overlapping chunks for the RAG tool to search over."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i : i + size]
        if chunk.strip():
            chunks.append(chunk)
        i += max(size - overlap, 1)
    return chunks


def extract_text_from_upload(filename: str, content: bytes) -> str:
    """Pull plain text out of an uploaded file (.pdf, .txt, .md, or anything text-ish)."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="ignore")
