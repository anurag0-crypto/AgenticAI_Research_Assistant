"""
db.py — the desk's filing cabinet.

Everything Marginalia needs to remember lives here, in a single SQLite file:
  - sessions            one row per "inquiry" (a chat thread)
  - messages            short-term memory: the back-and-forth within a session
  - long_term_memory    long-term memory: durable facts/preferences/goals the
                         agent chose to save, readable across ALL sessions
  - documents           chunked text from files the user uploaded, used for RAG

No ORM, no external service — just sqlite3 from the standard library, so the
whole memory system is easy to read top to bottom in one sitting.
"""

import sqlite3
import os
import datetime
from pathlib import Path
from contextlib import contextmanager

DB_PATH = os.getenv("APP_DB_PATH", str(Path(__file__).resolve().parent / "data" / "app.db"))
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS long_term_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    fact TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT 'fact',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.executescript(SCHEMA)


def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


# ---------------------------------------------------------------- sessions --

def create_session(session_id: str, title: str = "Untitled inquiry"):
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (session_id, title, _now(), _now()),
        )


def get_session(session_id: str):
    with _conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None


def list_sessions():
    with _conn() as c:
        rows = c.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]


def update_session_title(session_id: str, title: str):
    with _conn() as c:
        c.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))


def touch_session(session_id: str):
    with _conn() as c:
        c.execute("UPDATE sessions SET updated_at=? WHERE id=?", (_now(), session_id))


def delete_session(session_id: str):
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM documents WHERE session_id=?", (session_id,))


# ---------------------------------------------------------------- messages --

def add_message(session_id: str, role: str, content: str):
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session_id, role, content, _now()),
        )


def get_recent_messages(session_id: str, limit: int = 12):
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


# --------------------------------------------------------- long-term memory --

def save_memory(session_id: str, fact: str, tag: str = "fact"):
    with _conn() as c:
        c.execute(
            "INSERT INTO long_term_memory (session_id, fact, tag, created_at) VALUES (?,?,?,?)",
            (session_id, fact, tag, _now()),
        )


def list_memory(limit: int = 200):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM long_term_memory ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_memory(memory_id: int):
    with _conn() as c:
        c.execute("DELETE FROM long_term_memory WHERE id=?", (memory_id,))


def search_memory(query: str, limit: int = 5):
    """Tiny keyword-overlap search across ALL sessions' saved memory.

    This is intentionally simple (no embeddings) so it's transparent and free
    to run on every turn. Falls back to "most recent" if nothing matches, so
    recall_memory rarely comes back completely empty when memory exists.
    """
    words = {w.lower() for w in query.split() if len(w) > 2}
    rows = list_memory(limit=500)
    if not rows:
        return []
    if not words:
        return rows[:limit]
    scored = []
    for r in rows:
        fact_words = set(r["fact"].lower().split())
        scored.append((len(words & fact_words), r))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [r for score, r in scored if score > 0][:limit]
    if not top:
        top = rows[: min(limit, 3)]
    return top


# -------------------------------------------------------- documents (RAG) --

def add_document_chunks(session_id: str, filename: str, chunks: list):
    with _conn() as c:
        for i, chunk in enumerate(chunks):
            c.execute(
                "INSERT INTO documents (session_id, filename, chunk_index, content, created_at) "
                "VALUES (?,?,?,?,?)",
                (session_id, filename, i, chunk, _now()),
            )


def get_document_chunks(session_id: str):
    with _conn() as c:
        rows = c.execute(
            "SELECT filename, chunk_index, content FROM documents WHERE session_id=? "
            "ORDER BY filename, chunk_index",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_documents(session_id: str):
    with _conn() as c:
        rows = c.execute(
            "SELECT filename, COUNT(*) as chunks FROM documents WHERE session_id=? "
            "GROUP BY filename",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
