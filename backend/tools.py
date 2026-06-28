"""
tools.py — everything the agent can *do*, beyond talking.

Each tool is a plain LangChain @tool function. Three are global (web_search,
arxiv_search, calculator); four are session-scoped (they need to know which
inquiry they're attached to) and are produced by build_tools(session_id),
which closes over the session id so the LLM never has to pass it manually.

Why each tool exists (also explained in the README):
  - web_search             the LLM's knowledge is frozen at training time;
                            this gets it current, real information.
  - arxiv_search           general web search is noisy for academic claims;
                            this gets real paper titles/authors/dates instead
                            of the model guessing citations from memory.
  - calculator             LLMs are unreliable at exact arithmetic; this
                            offloads any real computation to actual code.
  - knowledge_base_search  grounds answers in documents *you* uploaded,
                            rather than the model's general knowledge.
  - save_memory/recall_memory   give the agent a persistent memory across
                            sessions, not just within one conversation.
  - generate_pdf_report    turns a chat answer into a tangible deliverable.
"""

import ast
import datetime
import math
import operator
import os
import re
from pathlib import Path

import requests
from langchain_core.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import db

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - fallback for older installs
    from duckduckgo_search import DDGS

import arxiv
from fpdf import FPDF

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(Path(__file__).resolve().parent / "reports")))


# ------------------------------------------------------------- web_search --

@tool
def web_search(query: str) -> str:
    """Search the public web for current information: news, facts, prices, statistics,
    or anything published after the model's training cutoff or too niche to be memorized.
    Returns up to 5 results, one per line, formatted as 'TITLE :: URL :: snippet'."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    try:
        if tavily_key:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": 5},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            lines = []
            for r in data.get("results", [])[:5]:
                snippet = (r.get("content") or "")[:220].replace("\n", " ")
                lines.append(f"{r.get('title', 'Untitled')} :: {r.get('url', '')} :: {snippet}")
            return "\n".join(lines) if lines else "No results found."
        else:
            results = DDGS().text(query, max_results=5)
            lines = []
            for r in results:
                snippet = (r.get("body") or "")[:220].replace("\n", " ")
                lines.append(f"{r.get('title', 'Untitled')} :: {r.get('href', '')} :: {snippet}")
            return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Web search error: {e}. Try rephrasing the query or rely on another tool."


# ------------------------------------------------------------ arxiv_search --

@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv.org for academic papers relevant to the query. Use this for scientific,
    technical, or research-heavy sub-questions where a peer-reviewed or preprint source
    matters more than a general web page. Returns up to 5 papers, one per line, formatted
    as 'TITLE :: URL :: authors (year): abstract snippet'."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
        lines = []
        for r in client.results(search):
            authors = ", ".join(a.name for a in r.authors[:3])
            if len(r.authors) > 3:
                authors += " et al."
            year = r.published.year if r.published else "n.d."
            snippet = r.summary[:220].replace("\n", " ")
            lines.append(f"{r.title} :: {r.entry_id} :: {authors} ({year}): {snippet}")
        return "\n".join(lines) if lines else "No papers found on arXiv for this query."
    except Exception as e:
        return f"arXiv search error: {e}"


# -------------------------------------------------------------- calculator --

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "pow": pow,
    "factorial": math.factorial,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _ALLOWED_NAMES[node.id]
    raise ValueError(f"Unsupported expression near: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely and exactly. Supports +, -, *, /, **, %,
    parentheses, the functions sqrt/log/log10/exp/sin/cos/tan/abs/round/factorial, and the
    constants pi and e. Always use this for arithmetic, percentages, growth rates, or any
    other precise number rather than estimating it yourself — language models are unreliable
    at exact arithmetic."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return f"RESULT :: {result}"
    except Exception as e:
        return f"Calculator error: could not evaluate '{expression}' ({e})"


# ------------------------------------------------------------------ PDF tool --

_UNICODE_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'",      # ' '
    "\u201c": '"', "\u201d": '"',      # " "
    "\u2013": "-", "\u2014": "-",      # en dash, em dash
    "\u2026": "...",                    # ellipsis
    "\u2022": "-", "\u25cf": "-", "\u25aa": "-",  # bullet variants
    "\u2192": "->", "\u2190": "<-",   # arrows
    "\u21bb": "(loop)", "\u2194": "<->",
    "\u00a0": " ",                       # non-breaking space
    "\u2212": "-",                       # minus sign
}


def _pdf_safe(text: str, max_token_len: int = 70) -> str:
    """Make a line of (possibly LLM-generated) markdown-ish text safe to hand
    to fpdf2's multi_cell. Two independent failure modes get neutralized here:

    1. A single unbroken "word" wider than the page (a raw URL or markdown
       link) — fpdf2 raises 'not enough horizontal space' for these.
    2. A character outside the basic Helvetica core font's encoding (smart
       quotes, em dashes, emoji, non-Latin scripts) — fpdf2 raises
       FPDFException for these. LLMs use "smart typography" constantly, so
       this is actually the more common of the two in practice.
    """
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    for src, dst in _UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.encode("latin-1", errors="ignore").decode("latin-1")

    def _break_long(match):
        token = match.group(0)
        return " ".join(token[i : i + max_token_len] for i in range(0, len(token), max_token_len))

    text = re.sub(rf"\S{{{max_token_len + 1},}}", _break_long, text)
    return text


def _render_pdf_rich(title: str, markdown_content: str, path) -> None:
    """The nice, formatted renderer: headings, bullets, bold stripped to
    plain text. Can still theoretically choke on some fpdf2 edge case we
    haven't seen yet — that's what _render_pdf_plain below is for."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    safe_title = _pdf_safe(title, max_token_len=35).strip() or "Research Report"
    pdf.multi_cell(0, 10, safe_title)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    stamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    pdf.cell(0, 6, f"Generated by Marginalia - {stamp}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for raw_line in markdown_content.splitlines():
        line = raw_line.rstrip()
        if not line:
            pdf.ln(3)
            continue
        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        clean = _pdf_safe(clean)

        if line.startswith("## "):
            body = clean[3:].strip()
            if not body:
                continue
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, body)
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("# "):
            body = clean[2:].strip()
            if not body:
                continue
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 8, body)
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("- ") or line.startswith("* "):
            body = clean[2:].strip()
            if not body:
                continue
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, f"   -  {body}")
        else:
            body = clean.strip()
            if not body:
                continue
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, body)

    pdf.output(str(path))


def _render_pdf_plain(title: str, markdown_content: str, path) -> None:
    """Absolute last-resort renderer. No markdown parsing, no headings, no
    bullets — just guaranteed-safe, fixed-width, never-empty chunks of plain
    text. Every chunk is short, non-empty, and pure latin-1, so this cannot
    trigger either of fpdf2's known failure modes. This is what fires if
    _render_pdf_rich fails for any reason at all, expected or not."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    def safe(s: str) -> str:
        return (s or "").encode("latin-1", errors="ignore").decode("latin-1")

    def chunked_cell(text: str, chunk_len: int) -> None:
        text = text.strip()
        if not text:
            return
        for i in range(0, len(text), chunk_len):
            chunk = text[i : i + chunk_len].strip()
            if chunk:
                pdf.multi_cell(0, 6, chunk)

    pdf.set_font("Helvetica", "B", 16)
    safe_title = safe(title).strip()[:120] or "Research Report"
    chunked_cell(safe_title, 30)
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    body = safe(markdown_content).replace("\r", "")
    for paragraph in body.split("\n"):
        text = re.sub(r"[#*_`>]", "", paragraph).strip()
        if not text:
            pdf.ln(4)
            continue
        chunked_cell(text, 45)

    pdf.output(str(path))


def _build_pdf(title: str, markdown_content: str, session_id: str) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{session_id}_{int(datetime.datetime.now().timestamp())}.pdf"
    path = REPORTS_DIR / fname
    try:
        _render_pdf_rich(title, markdown_content, path)
    except Exception:
        # The nice renderer hit something unforeseen — fall back to the
        # bulletproof plain renderer so a report is *always* produced.
        _render_pdf_plain(title, markdown_content, path)
    return f"/reports/{fname}"


# --------------------------------------------------- session-scoped tools --

def build_tools(session_id: str):
    """Build the session-scoped tools (RAG, memory, PDF export) for one inquiry."""

    @tool
    def knowledge_base_search(query: str) -> str:
        """Search the documents the user has uploaded for this session (PDF/TXT/MD notes)
        and return the most relevant excerpts. Use this whenever the question might be
        answered by the user's own material rather than general knowledge or the open web.
        Returns matches formatted as 'FILENAME (chunk N) :: doc-id :: excerpt'."""
        chunks = db.get_document_chunks(session_id)
        if not chunks:
            return "No documents have been uploaded for this session. Use web_search or arxiv_search instead."
        texts = [c["content"] for c in chunks]
        vectorizer = TfidfVectorizer(stop_words="english").fit(texts + [query])
        matrix = vectorizer.transform(texts)
        qvec = vectorizer.transform([query])
        sims = cosine_similarity(qvec, matrix)[0]
        ranked = sorted(zip(sims, chunks), key=lambda x: x[0], reverse=True)[:4]
        lines = []
        for score, c in ranked:
            if score <= 0:
                continue
            snippet = c["content"][:400].replace("\n", " ")
            lines.append(
                f"{c['filename']} (chunk {c['chunk_index']}) :: "
                f"doc://{c['filename']}#{c['chunk_index']} :: {snippet}"
            )
        return "\n".join(lines) if lines else "No relevant passages found in the uploaded documents."

    @tool
    def save_memory(fact: str, tag: str = "fact") -> str:
        """Permanently save a short, important fact, preference, or goal about the user or
        the ongoing project so it can be recalled in *future* sessions, not just this one.
        Use sparingly, only for durable information (e.g. 'prefers concise answers',
        'is writing a thesis on battery chemistry') — not transient details from one
        question. tag should be one of: fact, preference, goal."""
        db.save_memory(session_id, fact, tag)
        return f"Saved to long-term memory: {fact}"

    @tool
    def recall_memory(query: str) -> str:
        """Recall previously saved long-term facts, preferences, or goals relevant to the
        query — across ALL past sessions, not just this one."""
        facts = db.search_memory(query)
        if not facts:
            return "No relevant saved memory found."
        return "\n".join(f"[{f['tag']}] {f['fact']}" for f in facts)

    @tool
    def generate_pdf_report(title: str, markdown_content: str) -> str:
        """Generate a polished, downloadable PDF report from markdown-style content
        (use '# '/'## ' headings and '- ' bullets). Call this once, after writing the
        final research summary, only for substantive research worth a formal write-up —
        not for quick factual answers or casual chat. Returns 'Report generated :: /reports/<file>.pdf'."""
        url = _build_pdf(title, markdown_content, session_id)
        return f"Report generated :: {url}"

    return [
        web_search,
        arxiv_search,
        calculator,
        knowledge_base_search,
        save_memory,
        recall_memory,
        generate_pdf_report,
    ]