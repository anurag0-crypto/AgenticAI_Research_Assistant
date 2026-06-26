/* =========================================================================
   app.js — talks to the FastAPI backend, drives the corkboard animation,
   and renders the conversation. No build step, no framework: just fetch,
   WebSocket, and DOM updates.
   ========================================================================= */

(() => {
  "use strict";

  const els = {
    sidebar: document.getElementById("sidebar"),
    sidebarToggle: document.getElementById("sidebarToggle"),
    sessionList: document.getElementById("sessionList"),
    memoryList: document.getElementById("memoryList"),
    newInquiryBtn: document.getElementById("newInquiryBtn"),
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    sessionTitle: document.getElementById("sessionTitle"),
    fileInput: document.getElementById("fileInput"),
    logFeed: document.getElementById("logFeed"),
    conversation: document.getElementById("conversation"),
    conversationEmpty: document.getElementById("conversationEmpty"),
    planNotes: document.getElementById("planNotes"),
    composerForm: document.getElementById("composerForm"),
    composerInput: document.getElementById("composerInput"),
    sendBtn: document.getElementById("sendBtn"),
  };

  const NODE_IDS = ["router", "planner", "researcher", "analyst", "writer", "simple_responder"];

  let currentSessionId = null;
  let ws = null;
  let thinkingEl = null;
  let busy = false;

  // ------------------------------------------------------------- helpers --

  async function api(path, opts = {}) {
    const resp = await fetch(path, {
      headers: opts.body instanceof FormData ? {} : { "Content-Type": "application/json" },
      ...opts,
    });
    if (!resp.ok) {
      let msg = resp.statusText;
      try {
        const data = await resp.json();
        msg = data.detail || msg;
      } catch (_) {}
      throw new Error(msg);
    }
    if (resp.status === 204) return null;
    return resp.json();
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function toast(message) {
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3000);
  }

  function renderMarkdown(text) {
    if (window.marked && window.DOMPurify) {
      return window.DOMPurify.sanitize(window.marked.parse(text));
    }
    return escapeHtml(text).replace(/\n/g, "<br>");
  }

  // -------------------------------------------------------------- sessions --

  async function loadSessions() {
    const sessions = await api("/api/sessions");
    els.sessionList.innerHTML = "";
    if (sessions.length === 0) {
      const s = await api("/api/sessions", { method: "POST" });
      return selectSession(s.id);
    }
    for (const s of sessions) {
      els.sessionList.appendChild(sessionListItem(s));
    }
    if (!currentSessionId) {
      selectSession(sessions[0].id);
    }
  }

  function sessionListItem(s) {
    const li = document.createElement("li");
    li.className = "session-item" + (s.id === currentSessionId ? " active" : "");
    li.dataset.id = s.id;
    li.innerHTML = `
      <span class="session-item__tab"></span>
      <span class="session-item__title">${escapeHtml(s.title)}</span>
      <button class="session-item__del" title="Delete inquiry" type="button">✕</button>
    `;
    li.addEventListener("click", (e) => {
      if (e.target.closest(".session-item__del")) return;
      selectSession(s.id);
    });
    li.querySelector(".session-item__del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${s.title}"? This removes its messages and documents too.`)) return;
      await api(`/api/sessions/${s.id}`, { method: "DELETE" });
      if (s.id === currentSessionId) currentSessionId = null;
      loadSessions();
    });
    return li;
  }

  async function createSession() {
    const s = await api("/api/sessions", { method: "POST" });
    await loadSessions();
    selectSession(s.id);
  }

  async function selectSession(id) {
    currentSessionId = id;
    document.querySelectorAll(".session-item").forEach((li) => {
      li.classList.toggle("active", li.dataset.id === id);
    });
    resetDesk();
    els.logFeed.innerHTML = '<p class="log__empty">Ask a question below and watch the desk work.</p>';
    els.conversation.innerHTML = "";
    els.conversation.appendChild(els.conversationEmpty);

    try {
      const [session, messages] = await Promise.all([
        api("/api/sessions").then((all) => all.find((s) => s.id === id)),
        api(`/api/sessions/${id}/messages`),
      ]);
      els.sessionTitle.textContent = (session && session.title) || "Untitled inquiry";
      if (messages.length) {
        els.conversationEmpty.remove();
        for (const m of messages) {
          appendBubble(m.role === "user" ? "user" : "assistant", m.content);
        }
      } else if (!els.conversation.contains(els.conversationEmpty)) {
        els.conversation.appendChild(els.conversationEmpty);
      }
    } catch (e) {
      console.error(e);
    }

    connectWS(id);
    closeSidebarOnMobile();
  }

  // ------------------------------------------------------------ websocket --

  function connectWS(sessionId) {
    if (ws) {
      ws.onclose = null;
      ws.close();
    }
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws/${sessionId}`);

    ws.onopen = () => setStatus("online", "online");
    ws.onclose = () => setStatus("", "disconnected");
    ws.onerror = () => setStatus("error", "connection error");
    ws.onmessage = (ev) => {
      let event;
      try {
        event = JSON.parse(ev.data);
      } catch (_) {
        return;
      }
      handleEvent(event);
    };
  }

  function setStatus(dotClass, text) {
    els.statusDot.className = "dot" + (dotClass ? " " + dotClass : "");
    els.statusText.textContent = text;
  }

  function handleEvent(event) {
    switch (event.type) {
      case "run_start":
        resetDesk();
        els.logFeed.innerHTML = "";
        addThinkingBubble();
        break;
      case "route":
        if (event.path === "chat") {
          document.querySelectorAll('.card[data-node]').forEach((c) => {
            if (["planner", "researcher", "analyst", "writer"].includes(c.dataset.node)) {
              c.classList.add("skipped");
            }
          });
          document.querySelector(".thread--down")?.classList.add("lit");
        } else {
          document.querySelector('.card--branch')?.classList.add("skipped");
          document.querySelectorAll(".desk__chain .thread").forEach((t) => t.classList.add("lit"));
        }
        break;
      case "status":
        setNodeStatus(event.node, event.status);
        break;
      case "plan":
        renderPlanNotes(event.plan);
        break;
      case "analysis":
        if (!event.analysis.sufficient) {
          appendLog("analyst", "🧾 Fact-check: " + event.analysis.reasoning + " — sending it back for another pass.");
          document.querySelector('.card[data-node="analyst"]')?.classList.add("looping");
        } else if (event.analysis.reasoning) {
          appendLog("analyst", "🧾 Fact-check: " + event.analysis.reasoning);
        }
        break;
      case "log":
        appendLog(event.node, event.message);
        break;
      case "final":
        removeThinkingBubble();
        appendBubble("assistant", event.content, {
          sources: event.sources,
          reportUrl: event.report_url,
          savedMemories: event.saved_memories,
        });
        if (event.saved_memories && event.saved_memories.length) loadMemory();
        setBusy(false);
        break;
      case "error":
        removeThinkingBubble();
        appendError(event.message);
        setBusy(false);
        break;
    }
  }

  // ------------------------------------------------------------ the desk --

  function resetDesk() {
    document.querySelectorAll(".card").forEach((c) => c.classList.remove("active", "done", "skipped", "looping"));
    document.querySelectorAll(".thread").forEach((t) => t.classList.remove("lit"));
    els.planNotes.innerHTML = "";
  }

  function setNodeStatus(node, status) {
    const card = document.querySelector(`.card[data-node="${node}"]`);
    if (!card) return;
    if (status === "active") {
      card.classList.add("active");
      card.classList.remove("done");
    } else if (status === "done") {
      card.classList.remove("active", "looping");
      card.classList.add("done");
    }
  }

  function renderPlanNotes(plan) {
    els.planNotes.innerHTML = "";
    (plan.sub_tasks || []).forEach((task, i) => {
      const chip = document.createElement("span");
      chip.className = "note-chip";
      chip.style.animationDelay = `${i * 80}ms`;
      chip.textContent = "📍 " + task;
      els.planNotes.appendChild(chip);
    });
  }

  // -------------------------------------------------------------- log feed --

  function appendLog(node, message) {
    if (els.logFeed.querySelector(".log__empty")) els.logFeed.innerHTML = "";
    const line = document.createElement("div");
    line.className = "log__line";
    const ts = new Date().toLocaleTimeString([], { hour12: false });
    line.innerHTML = `<span class="ts">${ts}</span>${escapeHtml(message)}`;
    els.logFeed.appendChild(line);
    els.logFeed.scrollTop = els.logFeed.scrollHeight;
  }

  // ---------------------------------------------------------- conversation --

  function appendBubble(role, content, extra = {}) {
    if (els.conversation.contains(els.conversationEmpty)) els.conversationEmpty.remove();
    const bubble = document.createElement("div");
    if (role === "user") {
      bubble.className = "bubble bubble--user";
      bubble.textContent = content;
    } else {
      bubble.className = "bubble bubble--assistant";
      bubble.innerHTML = renderMarkdown(content);

      if (extra.reportUrl) {
        const a = document.createElement("a");
        a.className = "bubble__report";
        a.href = extra.reportUrl;
        a.target = "_blank";
        a.rel = "noopener";
        a.innerHTML = "🖨️ Download PDF report";
        bubble.appendChild(a);
      }

      if (extra.sources && extra.sources.length) {
        const wrap = document.createElement("div");
        wrap.className = "bubble__sources";
        extra.sources.slice(0, 8).forEach((s) => {
          const a = document.createElement("a");
          a.className = "bubble__source-tag";
          a.href = s.url;
          a.target = "_blank";
          a.rel = "noopener";
          let host = s.url;
          try { host = new URL(s.url).hostname.replace("www.", ""); } catch (_) {}
          a.textContent = "🔗 " + host;
          a.title = s.title;
          wrap.appendChild(a);
        });
        bubble.appendChild(wrap);
      }

      if (extra.savedMemories && extra.savedMemories.filter(Boolean).length) {
        const memo = document.createElement("div");
        memo.className = "bubble__memo";
        memo.textContent = "📌 Filed to The Index: " + extra.savedMemories.filter(Boolean).join("; ");
        bubble.appendChild(memo);
      }
    }
    els.conversation.appendChild(bubble);
    els.conversation.scrollTop = els.conversation.scrollHeight;
  }

  function addThinkingBubble() {
    if (els.conversation.contains(els.conversationEmpty)) els.conversationEmpty.remove();
    thinkingEl = document.createElement("div");
    thinkingEl.className = "bubble bubble--thinking";
    thinkingEl.innerHTML = '<span>The desk is working<span class="ellipsis"></span></span>';
    els.conversation.appendChild(thinkingEl);
    els.conversation.scrollTop = els.conversation.scrollHeight;
  }

  function removeThinkingBubble() {
    if (thinkingEl) {
      thinkingEl.remove();
      thinkingEl = null;
    }
  }

  function appendError(message) {
    if (els.conversation.contains(els.conversationEmpty)) els.conversationEmpty.remove();
    const bubble = document.createElement("div");
    bubble.className = "bubble bubble--error";
    bubble.textContent = "⚠️ " + message;
    els.conversation.appendChild(bubble);
    els.conversation.scrollTop = els.conversation.scrollHeight;
  }

  // --------------------------------------------------------------- memory --

  async function loadMemory() {
    const memories = await api("/api/memory");
    els.memoryList.innerHTML = "";
    if (!memories.length) {
      els.memoryList.innerHTML =
        '<li class="memory-empty">Nothing filed yet. The desk saves durable facts here as it learns them — across every inquiry.</li>';
      return;
    }
    for (const m of memories) {
      const li = document.createElement("li");
      li.className = "memory-item";
      li.innerHTML = `
        <span class="memory-item__tag">${escapeHtml(m.tag)}</span>
        <button class="memory-item__del" title="Forget this" type="button">✕</button>
        <div>${escapeHtml(m.fact)}</div>
      `;
      li.querySelector(".memory-item__del").addEventListener("click", async () => {
        await api(`/api/memory/${m.id}`, { method: "DELETE" });
        loadMemory();
      });
      els.memoryList.appendChild(li);
    }
  }

  // --------------------------------------------------------------- upload --

  async function uploadFile(file) {
    if (!currentSessionId) return;
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await api(`/api/sessions/${currentSessionId}/upload`, { method: "POST", body: form });
      toast(`📥 ${res.filename} added — ${res.chunks} chunks indexed for this inquiry`);
    } catch (e) {
      toast("Couldn't read that file: " + e.message);
    }
  }

  // -------------------------------------------------------------- composer --

  function setBusy(value) {
    busy = value;
    els.sendBtn.disabled = value;
  }

  function sendQuery(text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      toast("Not connected yet — try again in a moment.");
      return;
    }
    appendBubble("user", text);
    setBusy(true);
    ws.send(JSON.stringify({ type: "query", content: text }));
  }

  function closeSidebarOnMobile() {
    els.sidebar.classList.remove("open");
  }

  // ----------------------------------------------------------------- wire --

  els.newInquiryBtn.addEventListener("click", createSession);

  els.sidebarToggle.addEventListener("click", () => {
    els.sidebar.classList.toggle("open");
  });

  els.fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) uploadFile(file);
    e.target.value = "";
  });

  els.composerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    if (busy) return;
    const text = els.composerInput.value.trim();
    if (!text) return;
    sendQuery(text);
    els.composerInput.value = "";
    els.composerInput.style.height = "auto";
  });

  els.composerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.composerForm.requestSubmit();
    }
  });

  els.composerInput.addEventListener("input", () => {
    els.composerInput.style.height = "auto";
    els.composerInput.style.height = Math.min(els.composerInput.scrollHeight, 160) + "px";
  });

  // ----------------------------------------------------------------- boot --

  loadSessions();
  loadMemory();
})();
