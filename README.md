# Marginalia — an AI research desk

Marginalia is a multi-agent research assistant. You ask it something; it
**plans** a short research strategy, **researches** each part with real
tools (web search, arXiv, your own uploaded documents), **checks its own
work** and decides whether it needs another pass, then **writes up** a
final answer and **remembers** durable
facts about you for next time. A live "corkboard" in the UI shows which
agent is working at every moment.

It's built for the Agentic AI assignment (Task 5): an LLM + tools + memory +
multi-step autonomous workflow, not a single prompt-in/response-out chatbot.

---

## 1. What's actually in here

```
marginalia/
├── backend/
│   ├── main.py          FastAPI app: REST endpoints, WebSocket, serves the frontend
│   ├── graph.py          the LangGraph multi-agent workflow (the heart of the project)
│   ├── tools.py          every tool the agent can call, and why
│   ├── llm.py             swap-any-provider LLM factory (Claude / OpenAI / Groq / Gemini)
│   ├── db.py               SQLite memory layer (sessions, chat history, long-term memory, docs)
│   ├── utils.py            small shared helpers (JSON parsing, chunking, logging text)
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── index.html        the research-desk UI
    ├── style.css           "corkboard pipeline" design system
    └── app.js                WebSocket client + rendering, no build step
```

One process, one port: `uvicorn` serves the API, the WebSocket, *and* the
static frontend together, so there's nothing else to spin up.

---

## 2. Quick start

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# open .env and set LLM_PROVIDER + the matching API key (just one provider is enough)

uvicorn main:app --reload --port 8000
```

Opening **http://localhost:8000** creates a first "inquiry"
automatically.

 One LLM API key required:

| Provider  | Get a key at                              | Free tier? |
|-----------|--------------------------------------------|------------|
| Anthropic | console.anthropic.com/settings/keys        | trial credit |
| OpenAI    | platform.openai.com/api-keys                | trial credit |
| Groq      | console.groq.com/keys                       | **yes, generous** |
| Gemini    | aistudio.google.com/apikey                   | **yes** |

Web search works out of the box with no key (free DuckDuckGo backend). If
a free [Tavily](https://tavily.com) key is added , the agent automatically
upgrades to it for higher quality, more reliable results.

> Model names move fast. If a default model in `.env.example` is no longer
> live by the time you read this, swap in the provider's current model id —
> nothing else in the code needs to change, since `llm.py` is the single
> place models are named.

---

## 3. The agent workflow

This isn't `prompt → LLM → response`. A single user message can trigger up
to **six** specialized LLM calls across **six roles**, wired together with
[LangGraph](https://langchain-ai.github.io/langgraph/):

```
                                                 ┌───────────┐
                                  ┌──research──▶ │  CURATOR  │  (Planner)
                                  │              │  plans 2-4 │
                                  │              │  sub-tasks │
                                  │              └─────┬─────┘
┌─────────────┐                  │                     ▼
│  DESK CLERK │──────────────────┤              ┌─────────────┐
│  (Router)   │                  │              │ FIELD AGENT │  (Researcher)
│  classifies │                  │              │ tools per   │◀─┐
│  the message│                  │              │ sub-task    │  │ loop back for
└──────┬──────┘                  │              └──────┬──────┘  │ one more pass
       │                         │                     ▼          │ if a real gap
     chat                        │              ┌─────────────┐  │ is found
       ▼                         │              │ FACT CHECKER│──┘
┌──────────────┐                 │              │ (Analyst)   │
│ QUICK REPLY  │                 │              │ verifies,   │
│ (simple_     │                 │              │ decides if  │
│  responder)  │                 │              │ enough      │
└──────┬───────┘                 │              └──────┬──────┘
       │                         │                  sufficient
       ▼                         │                     ▼
      END                       (research)      ┌─────────────┐
                                                  │   EDITOR    │  (Writer)
                                                  │ writes the  │
                                                  │ final answer│
                                                  │
                                                     + memory│
                                                  └──────┬──────┘
                                                          ▼
                                                         END
```

Each box is **one LangGraph node** in `graph.py`. Concretely, this single
graph demonstrates four of the patterns from Anthropic's
["Building Effective Agents"](https://www.anthropic.com/engineering/building-effective-agents):

- **Routing** — the Desk Clerk classifies every message as `research` or
  `chat` before doing any real work, so a "thanks!" doesn't trigger a full
  research pipeline.
- **Prompt chaining** — Curator → Field Agent → Fact Checker → Editor is a
  straight pipeline where each step's output feeds the next.
- **Orchestrator/worker** — the Field Agent doesn't follow a fixed script;
  for each sub-task it *decides* which tool(s) to call (web search? arXiv?
  the user's own documents? more than one?) in a small ReAct loop.
- **Evaluator-optimizer** — the Fact Checker can reject the research as
  insufficient and send the Field Agent back for one more targeted pass
  (capped at one retry, so cost and latency stay bounded).

The UI's "corkboard" (`frontend/style.css` / `app.js`) mirrors this exactly:
every node emits `status: active` / `status: done` events over the
WebSocket, so you watch the pin drop on each card in real time, watch the
Fact Checker's loop-back happen live, and see the branch the Quick Reply
path takes when the Router decides full research isn't needed.

---

## 4. Tools — and why each one is necessary

| Tool | Used by | Why an LLM alone can't do this |
|---|---|---|
| `web_search` | Field Agent | The model's knowledge is frozen at training time; this gets current facts, news, and prices. |
| `arxiv_search` | Field Agent | General web search is noisy for academic claims; this returns real paper titles/authors/dates instead of the model guessing or inventing citations. |
| `calculator` | Fact Checker | LLMs are unreliable at exact arithmetic. Any number worth checking (percentages, growth rates, sums) goes through a real, sandboxed evaluator instead of being "eyeballed." |
| `knowledge_base_search` | Field Agent | Grounds answers in **your** uploaded documents (TF-IDF retrieval over chunked text) — for material that isn't on the public web and isn't in the model's training data. |
| `save_memory` / `recall_memory` | Editor, Quick Reply | Give the agent a memory that survives *after* the chat ends — an LLM call by itself has zero persistent state. |

Every tool call is logged to the UI's "Field log" the moment it happens —
that's not cosmetic, it's literally streaming the same `tool_calls` LangGraph
captured from the LLM, so you can see *exactly* which tool was chosen, with
what arguments, and what came back.

The calculator is deliberately **not** `eval()` — it's a hand-written,
whitelisted AST evaluator (`tools.py`) that only allows arithmetic
operators, a few safe math functions, and numeric constants, since handing
a raw `eval()` to anything an LLM controls is a real security risk.

---

## 5. Memory — two kinds, on purpose

**Short-term (per session)** — `messages` table in SQLite. Every turn in an
inquiry is stored and replayed back into the Router/Planner/Quick-Reply
prompts as conversation context, so follow-up questions work naturally.

**Long-term (across sessions)** — `long_term_memory` table. This is memory
in the sense the assignment means it: it **outlives the conversation**. The
Editor and Quick Reply roles can call `save_memory` to file away a durable
fact, preference, or goal ("prefers concise answers," "is comparing solar
vs. wind for a town council report"). On every *future* turn — in any
session — the backend pre-fetches relevant saved facts
(`db.search_memory`) and feeds them to the Curator/Planner before it even
starts planning, and the agent can also explicitly call `recall_memory`
mid-conversation. The sidebar's "Index" panel lists everything saved, with
a one click "forget" button, so memory is inspectable and correctable, not
a black box.

**Document memory (RAG)** — uploaded files are chunked and stored in
`documents`, scoped to one inquiry, and retrieved by TF‑IDF cosine
similarity in `knowledge_base_search`. This is a third, deliberately
separate kind of memory: per-document, not per-fact.

---

## 6. Mapping back to the assignment

- ✅ Accepts user input — chat composer, streamed over WebSocket
- ✅ Uses an LLM API — any of Claude / OpenAI / Groq / Gemini, switchable in `.env`
- ✅ Uses **7** tools across 2+ categories (web search, academic API, sandboxed
  calculator, file-based RAG, a small SQLite-backed memory store)
- ✅ Tool necessity is explained above and visible live in the UI log
- ✅ Memory persists across sessions (long-term facts) *and* within a session
  (chat history) — not just a rolling chat buffer
- ✅ Multi-step workflow — up to 6 chained/branching LLM calls per message,
  not one call
- ✅ Demonstrates planning (Curator), tool selection + use (Field Agent),
  evaluation/decision-making (Fact Checker's loop-back, Router's classification),
  and autonomous final action 
- ✅ Polished, purpose-built frontend (no UI framework, fully custom design)

**Bonus features implemented:** multi-agent architecture (LangGraph,
6 roles), RAG over uploaded documents, live workflow
visualization, custom tool creation (a hand-rolled safe calculator, a custom
TF-IDF retrieval tool), and a from-scratch design system instead of a
default template.

---

## 7. Notes, limits, and honest caveats

- The TF-IDF retrieval in `knowledge_base_search` is intentionally simple
  (no embeddings model, no vector DB) so the whole RAG path is readable in
  one file. Swapping in a proper embedding index would be a natural next step.
- `recall_memory`'s search is keyword-overlap, not semantic — good enough
  for a small number of saved facts, but it's the first thing to upgrade if
  long-term memory grows large.
- The research loop is capped at one retry (`MAX_ITERATIONS = 2` in
  `graph.py`) to keep latency and API cost predictable for a demo; raise it
  if you want the Fact Checker to be more persistent.
- SQLite access in `main.py` is synchronous; fine at demo scale, but a real
  deployment with concurrent users would want `asyncio.to_thread` around DB
  calls or a proper async driver.
- Library and model names move quickly — `requirements.txt` is pinned to a
  known-good set; if something is unavailable by the time you install it,
  bump that one line rather than the whole file.

---

