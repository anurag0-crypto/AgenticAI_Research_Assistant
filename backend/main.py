"""
main.py — the front desk: HTTP/WebSocket layer that wires the UI to the agent.

Run with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 — the frontend is served from the same
process, so there's nothing else to start.
"""

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import db
from graph import GRAPH
from utils import chunk_text, extract_text_from_upload

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "reports")))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

db.init_db()

app = FastAPI(title="Marginalia Research Desk")


# --------------------------------------------------------------------- API --

@app.get("/api/health")
async def health():
    return {"status": "ok", "provider": os.getenv("LLM_PROVIDER", "anthropic")}


@app.get("/api/sessions")
async def api_list_sessions():
    return db.list_sessions()


@app.post("/api/sessions")
async def api_create_session():
    sid = str(uuid.uuid4())
    db.create_session(sid)
    return db.get_session(sid)


@app.get("/api/sessions/{session_id}/messages")
async def api_session_messages(session_id: str):
    if not db.get_session(session_id):
        raise HTTPException(404, "Session not found")
    return db.get_recent_messages(session_id, limit=200)


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    db.delete_session(session_id)
    return {"deleted": session_id}


@app.get("/api/sessions/{session_id}/documents")
async def api_session_documents(session_id: str):
    return db.list_documents(session_id)


@app.post("/api/sessions/{session_id}/upload")
async def api_upload(session_id: str, file: UploadFile = File(...)):
    if not db.get_session(session_id):
        db.create_session(session_id)
    content = await file.read()
    try:
        text = extract_text_from_upload(file.filename, content)
    except Exception as e:
        raise HTTPException(400, f"Could not read {file.filename}: {e}")
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, f"No readable text found in {file.filename}")
    db.add_document_chunks(session_id, file.filename, chunks)
    return {"filename": file.filename, "chunks": len(chunks)}


@app.get("/api/memory")
async def api_list_memory():
    return db.list_memory()


@app.delete("/api/memory/{memory_id}")
async def api_delete_memory(memory_id: int):
    db.delete_memory(memory_id)
    return {"deleted": memory_id}


# ------------------------------------------------------------- agent turns --

def _derive_title(text: str) -> str:
    words = text.strip().split()
    title = " ".join(words[:8])
    if len(words) > 8 or len(title) >= 60:
        title = title[:60] + "\u2026"
    return title or "Untitled inquiry"


async def run_turn(session_id: str, user_text: str, ws: WebSocket):
    async def emit(event: dict):
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            pass

    db.add_message(session_id, "user", user_text)
    history = db.get_recent_messages(session_id, limit=12)
    memory_hits = db.search_memory(user_text, limit=5)

    state = {
        "session_id": session_id,
        "run_id": str(uuid.uuid4()),
        "user_query": user_text,
        "chat_history": history,
        "relevant_memory": [f"[{m['tag']}] {m['fact']}" for m in memory_hits],
        "iterations": 0,
        "pending_tasks": [],
        "completed_tasks": [],
        "findings": "",
        "sources": [],
        "saved_memories": [],
        "report_url": None,
        "emit": emit,
    }

    await emit({"type": "run_start"})
    try:
        result = await GRAPH.ainvoke(state)
    except Exception as e:
        await emit({"type": "error", "message": str(e)})
        return

    final_text = result.get("final_answer") or (
        "I couldn't put together an answer this time \u2014 try rephrasing the question."
    )
    db.add_message(session_id, "assistant", final_text)
    db.touch_session(session_id)

    seen, sources = set(), []
    for s in result.get("sources", []):
        if s["url"] not in seen:
            seen.add(s["url"])
            sources.append(s)

    await emit(
        {
            "type": "final",
            "content": final_text,
            "report_url": result.get("report_url"),
            "sources": sources,
            "route": result.get("route"),
            "plan": result.get("plan"),
            "saved_memories": result.get("saved_memories", []),
        }
    )


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if not db.get_session(session_id):
        db.create_session(session_id)
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if data.get("type") == "query":
                text = (data.get("content") or "").strip()
                if not text:
                    continue
                session = db.get_session(session_id)
                if session and session["title"] == "Untitled inquiry":
                    db.update_session_title(session_id, _derive_title(text))
                await run_turn(session_id, text, websocket)
    except WebSocketDisconnect:
        pass


# --------------------------------------------------------- static frontend --
# Registered last on purpose: a "/" StaticFiles mount is a catch-all, so it
# must come after every /api and /ws route or it would shadow them.

app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
