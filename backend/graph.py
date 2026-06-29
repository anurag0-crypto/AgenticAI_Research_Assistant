"""
graph.py — the multi-agent workflow itself.

Six roles, wired together with LangGraph's StateGraph:

    START
      |
      v
   ROUTER  (Desk Clerk) ---chat---> SIMPLE_RESPONDER (Quick reply) --> END
      |
   research
      v
   PLANNER (Curator)
      v
  RESEARCHER (Field Agent) <---loop back if more research is needed---+
      v                                                                |
   ANALYST (Fact Checker) ---not sufficient, budget left---------------+
      |
    sufficient
      v
   WRITER (Editor) --> END

This single graph demonstrates four of the agent patterns from Anthropic's
"Building Effective Agents": ROUTING (the entry classification), PROMPT
CHAINING (planner -> researcher -> analyst -> writer), an ORCHESTRATOR/WORKER
split (the researcher dynamically chooses tools per sub-task), and an
EVALUATOR-OPTIMIZER loop (the analyst can send the researcher back for
another pass).

Design choice: instead of one shared message thread for the whole run (the
classic single-agent ReAct pattern), each node builds its own short-lived
prompt from the parts of `state` it actually needs. That keeps four very
different "personalities" (planner/researcher/analyst/writer) from polluting
each other's context, which is what makes this a *multi-agent* system rather
than one agent with a long memory.
"""

from typing import Any, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from llm import get_llm
from tools import build_tools
from utils import describe_call, extract_sources, safe_json_extract, summarize_result

MAX_ITERATIONS = 2  # at most one "go research more" loop, to bound cost/latency


class AgentState(TypedDict, total=False):
    session_id: str
    run_id: str
    user_query: str
    chat_history: list
    relevant_memory: list

    route: str  # "research" | "chat"

    plan: dict
    pending_tasks: list
    completed_tasks: list
    findings: str
    sources: list
    iterations: int
    analysis: dict

    final_answer: str
    report_url: Optional[str]
    saved_memories: list

    emit: Any  # async callable: emit(event: dict) -> None, used to drive the live UI


# ------------------------------------------------------------- prompt utils --

def _history_block(history: list) -> str:
    if not history:
        return "(no earlier messages in this session)"
    lines = []
    for m in history[-8:]:
        role = "Visitor" if m["role"] == "user" else "Desk"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _memory_block(memory: list) -> str:
    if not memory:
        return "(nothing relevant saved yet)"
    return "\n".join(f"- {m}" for m in memory)


# ------------------------------------------------------- the ReAct sub-loop --

async def _tool_loop(llm, tools, system_prompt, user_prompt, emit, node_name, max_rounds=4):
    """Run a small reason -> act -> observe loop for one node.

    The LLM is offered `tools`; if it calls any, we execute them, log a
    friendly line to the UI for each, and feed the results back. This repeats
    until the model stops calling tools (or we hit max_rounds, in which case
    we force a final answer). Returns (final_text, tool_log, sources).
    """
    tools_by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools) if tools else llm
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    tool_log, sources = [], []

    for _ in range(max_rounds):
        ai_msg: Optional[AIMessage] = None
        last_error = None

        # Some hosted models (Groq's open-weight tool calling especially)
        # occasionally return a malformed/unparseable function call that the
        # provider's API rejects outright before any content comes back.
        # This is non-deterministic provider-side flakiness, not something
        # our prompt controls, so a same-input retry often just succeeds.
        for attempt in range(2):
            try:
                ai_msg = await bound.ainvoke(messages)
                break
            except Exception as e:
                last_error = e

        if ai_msg is None:
            # Both attempts failed: drop tools and force a plain-text answer
            # from what we have so far, rather than failing the whole turn.
            await emit({
                "type": "log",
                "node": node_name,
                "message": f"⚠️ Tool-calling hiccup ({type(last_error).__name__}) — answering without further tools.",
            })
            fallback = await llm.ainvoke(
                messages
                + [
                    HumanMessage(
                        content="Tool calling isn't available right now. Answer as best you can "
                        "from the information already available in this conversation, in plain text."
                    )
                ]
            )
            return fallback.content, tool_log, sources

        messages.append(ai_msg)
        calls = getattr(ai_msg, "tool_calls", None) or []
        if not calls:
            return ai_msg.content, tool_log, sources

        for call in calls:
            await emit({"type": "log", "node": node_name, "message": describe_call(call)})
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = f"Unknown tool: {call['name']}"
            else:
                try:
                    result = await tool.ainvoke(call["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            tool_log.append({"tool": call["name"], "args": call["args"], "result": str(result)[:4000]})
            sources.extend(extract_sources(result))
            await emit({"type": "log", "node": node_name, "message": summarize_result(call["name"], result)})
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    closing = await llm.ainvoke(
        messages + [HumanMessage(content="Stop calling tools now and give your final answer in plain text.")]
    )
    return closing.content, tool_log, sources


# ----------------------------------------------------------------- ROUTER --

async def router_node(state: AgentState) -> dict:
    await state["emit"]({"type": "status", "node": "router", "status": "active"})
    llm = get_llm(temperature=0)
    system = (
        "You are the Desk Clerk at a research desk. Decide how to route the visitor's message.\n"
        'Reply with ONLY a JSON object: {"route": "research" | "chat", "reason": "<one short sentence>"}.\n'
        'Choose "research" when answering well requires looking things up, gathering evidence from '
        "multiple sources, doing arithmetic, or producing a structured write-up.\n"
        'Choose "chat" for greetings, thanks, small talk, meta-questions about this conversation, or '
        "anything answerable purely from memory of what's already been discussed or saved."
    )
    user = f"Recent conversation:\n{_history_block(state['chat_history'])}\n\nNew message: {state['user_query']}"
    resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    data = safe_json_extract(resp.content)
    route = data.get("route") if data.get("route") in ("research", "chat") else "research"

    await state["emit"]({"type": "route", "path": route, "reason": data.get("reason", "")})
    await state["emit"]({"type": "status", "node": "router", "status": "done"})
    return {"route": route}


# ---------------------------------------------------------------- PLANNER --

async def planner_node(state: AgentState) -> dict:
    await state["emit"]({"type": "status", "node": "planner", "status": "active"})
    llm = get_llm(temperature=0.2)
    system = (
        "You are the Curator at a research desk: you turn a visitor's question into a short, "
        "concrete research plan. Reply with ONLY a JSON object:\n"
        '{"intent_summary": "<one sentence on what the visitor really wants>", '
        '"sub_tasks": ["<focused, search-friendly question>", ...]}\n'
        "Produce 2 to 4 sub_tasks. Each must be specific enough to type directly into a search "
        "engine or paper database. Prefer fewer, sharper sub-tasks over many vague ones."
    )
    user = (
        f"Relevant things remembered about this visitor:\n{_memory_block(state['relevant_memory'])}\n\n"
        f"Recent conversation:\n{_history_block(state['chat_history'])}\n\n"
        f"Question to plan for: {state['user_query']}"
    )
    resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    data = safe_json_extract(resp.content)
    sub_tasks = data.get("sub_tasks") or [state["user_query"]]
    plan = {"intent_summary": data.get("intent_summary", state["user_query"]), "sub_tasks": sub_tasks}

    await state["emit"]({"type": "plan", "plan": plan})
    await state["emit"]({"type": "status", "node": "planner", "status": "done"})
    return {"plan": plan, "pending_tasks": sub_tasks}


# -------------------------------------------------------------- RESEARCHER --

async def researcher_node(state: AgentState) -> dict:
    await state["emit"]({"type": "status", "node": "researcher", "status": "active"})
    llm = get_llm(temperature=0.2)
    by_name = {t.name: t for t in build_tools(state["session_id"])}
    tools = [by_name["web_search"], by_name["arxiv_search"], by_name["knowledge_base_search"]]

    findings = state.get("findings", "")
    completed = list(state.get("completed_tasks", []))
    sources = list(state.get("sources", []))

    for task in state.get("pending_tasks", []):
        system = (
            "You are the Field Agent at a research desk. Investigate exactly ONE sub-question using "
            "the tools available (web search, arXiv, the visitor's uploaded documents). Use 1-3 tool "
            "calls — enough to ground your answer, not more than needed. When you have enough, reply "
            "in plain text: a tight 3-6 sentence summary of what you found, mentioning sources by name."
        )
        user = f"Original question: {state['user_query']}\nSub-question to investigate: {task}"
        summary, _, new_sources = await _tool_loop(llm, tools, system, user, state["emit"], "researcher")
        findings += f"\n\n### {task}\n{summary}"
        completed.append({"task": task, "summary": summary})
        sources += new_sources

    await state["emit"]({"type": "status", "node": "researcher", "status": "done"})
    return {"findings": findings, "completed_tasks": completed, "sources": sources, "pending_tasks": []}


# ----------------------------------------------------------------- ANALYST --

async def analyst_node(state: AgentState) -> dict:
    await state["emit"]({"type": "status", "node": "analyst", "status": "active"})
    llm = get_llm(temperature=0)
    by_name = {t.name: t for t in build_tools(state["session_id"])}
    tools = [by_name["calculator"]]

    system = (
        "You are the Fact Checker at a research desk. Review the research notes against the "
        "original question. If any figures, percentages, or statistics in the notes deserve "
        "double-checking, use the calculator tool. When you're done, reply with ONLY a JSON object:\n"
        '{"sufficient": true|false, "reasoning": "<one or two sentences>", '
        '"follow_up_tasks": ["<question>", ...]}\n'
        "follow_up_tasks should be empty if sufficient is true. Mark sufficient as false only if "
        "there is a real, important gap — don't ask for more research just for completeness's sake."
    )
    user = f"Original question: {state['user_query']}\n\nResearch notes so far:{state['findings']}"
    raw, _, _ = await _tool_loop(llm, tools, system, user, state["emit"], "analyst", max_rounds=3)
    data = safe_json_extract(raw)
    analysis = {
        "sufficient": data.get("sufficient", True),
        "reasoning": data.get("reasoning", ""),
        "follow_up_tasks": data.get("follow_up_tasks") or [],
    }
    iterations = state.get("iterations", 0) + 1
    pending = analysis["follow_up_tasks"] if (not analysis["sufficient"] and iterations < MAX_ITERATIONS) else []

    await state["emit"]({"type": "analysis", "analysis": analysis})
    await state["emit"]({"type": "status", "node": "analyst", "status": "done"})
    return {"analysis": analysis, "iterations": iterations, "pending_tasks": pending}


# ------------------------------------------------------------------ WRITER --

async def writer_node(state: AgentState) -> dict:
    await state["emit"]({"type": "status", "node": "writer", "status": "active"})
    llm = get_llm(temperature=0.4)
    by_name = {t.name: t for t in build_tools(state["session_id"])}
    tools = [by_name["save_memory"]]

    system = (
        "You are the Editor at a research desk. Write the final answer for the visitor using the "
        "research notes below. Use clear markdown: a short lead sentence, then '## ' headings and "
        "'- ' bullets if it helps, citing sources by name (not raw URLs). Be honest about open "
        "questions or conflicting evidence.\n\n"
        "If a durable fact, preference, or goal about the visitor emerged during this inquiry, call "
        "save_memory (at most once) to file it away for future sessions. Then reply with your final "
        "answer in plain markdown text and stop."
    )
    user = (
        f"Original question: {state['user_query']}\n\n"
        f"Research notes:{state['findings']}\n\n"
        f"Fact-checker's note: {state.get('analysis', {}).get('reasoning', '')}"
    )
    final_text, tool_log, _ = await _tool_loop(llm, tools, system, user, state["emit"], "writer", max_rounds=3)
    saved = [e["args"].get("fact", "") for e in tool_log if e["tool"] == "save_memory"]

    await state["emit"]({"type": "status", "node": "writer", "status": "done"})
    return {"final_answer": final_text, "report_url": None, "saved_memories": saved}


# --------------------------------------------------------- SIMPLE_RESPONDER --

async def simple_responder_node(state: AgentState) -> dict:
    await state["emit"]({"type": "status", "node": "simple_responder", "status": "active"})
    llm = get_llm(temperature=0.4)
    by_name = {t.name: t for t in build_tools(state["session_id"])}
    tools = [by_name["recall_memory"], by_name["save_memory"]]

    system = (
        "You are the Desk Clerk at a research desk, handling a quick message that doesn't need the "
        "full research process. Reply warmly and briefly. If the visitor refers to something earlier, "
        "use recall_memory to check saved notes before answering. If they share a durable preference "
        "or fact, save it with save_memory. Keep your final reply to a few sentences."
    )
    user = f"Recent conversation:\n{_history_block(state['chat_history'])}\n\nMessage: {state['user_query']}"
    final_text, tool_log, _ = await _tool_loop(llm, tools, system, user, state["emit"], "simple_responder", max_rounds=3)
    saved = [e["args"].get("fact", "") for e in tool_log if e["tool"] == "save_memory"]

    await state["emit"]({"type": "status", "node": "simple_responder", "status": "done"})
    return {"final_answer": final_text, "saved_memories": saved}


# ------------------------------------------------------------ graph wiring --

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("analyst", analyst_node)
    g.add_node("writer", writer_node)
    g.add_node("simple_responder", simple_responder_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {"research": "planner", "chat": "simple_responder"},
    )
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "analyst")
    g.add_conditional_edges(
        "analyst",
        lambda s: "researcher" if s.get("pending_tasks") else "writer",
        {"researcher": "researcher", "writer": "writer"},
    )
    g.add_edge("writer", END)
    g.add_edge("simple_responder", END)
    return g.compile()


GRAPH = build_graph()