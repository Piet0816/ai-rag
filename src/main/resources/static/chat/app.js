// ======== RAG Chat UI (vanilla JS) — streaming + spacing fix + compact red trash ========

// --- DOM refs
const chatListEl = document.getElementById("chatList");
const newChatBtn = document.getElementById("newChatBtn");
const reloadIndexBtn = document.getElementById("reloadIndexBtn");
const ingestAllBtn = document.getElementById("ingestAllBtn");
const sidebarFooter = document.querySelector(".sidebar__footer");

const messagesEl = document.getElementById("messages");
const composerForm = document.getElementById("composerForm");
const composerInput = document.getElementById("composerInput");
const sendBtn = document.getElementById("sendBtn");

const modelNameEl = document.getElementById("modelName");
const topKInput = document.getElementById("topK");

const contextModal = document.getElementById("contextModal");
const contextText = document.getElementById("contextText");
const closeContextBtn = document.getElementById("closeContextBtn");

// --- helpers
const nowIso = () => new Date().toISOString();
const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
const sanitize = (s) => (s ?? "").toString();
function setModelName(name) { if (name && name.trim()) modelNameEl.textContent = name; }
function autoResizeTextarea(ta) { ta.style.height = "auto"; ta.style.height = Math.min(200, ta.scrollHeight) + "px"; }
function uid() { return Math.random().toString(36).slice(2) + Date.now().toString(36); }

// --- spacing helper (fix for models that stream no leading spaces)
const isWord = c => !!c && /[A-Za-z0-9]/.test(c); // keep simple; works well for our test set
function smartAppend(prev, delta) {
  if (!delta) return prev;
  return prev + delta;
}

let userCanceled = false;

function ensureStopButton() {
  let btn = document.getElementById("stopBtn");
  if (btn) return btn;

  btn = document.createElement("button");
  btn.id = "stopBtn";
  btn.type = "button";
  btn.textContent = "Stop";
  btn.title = "Stop streaming";
  btn.className = "btn";
  btn.style.marginLeft = "8px";
  btn.style.display = "none"; // hidden until streaming
  btn.onclick = () => {
    userCanceled = true;
    if (activeStream && activeStream.cancel) {
      try { activeStream.cancel(); } catch {}
    }
    btn.disabled = true;
    btn.textContent = "Stopping…";
  };

  // place it right next to the Send button
  sendBtn.parentElement.insertBefore(btn, sendBtn.nextElementSibling);
  return btn;
}

// --- local storage
const STORAGE_KEY = "ragChatConvosV1";
const state = { conversations: [], activeId: null };

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (Array.isArray(data.conversations)) {
      state.conversations = data.conversations;
      state.activeId = data.activeId || data.conversations[0]?.id || null;
    }
  } catch {}
}
function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ conversations: state.conversations, activeId: state.activeId }));
}

// --- conversations
function createConversation(title = "New chat") {
  const c = { id: uid(), title, messages: [], createdAt: nowIso(), updatedAt: nowIso() };
  state.conversations.unshift(c);
  state.activeId = c.id;
  saveState(); renderChatList(); renderMessages();
  return c;
}
function getActiveConversation() { return state.conversations.find(c => c.id === state.activeId) || createConversation(); }
function setActiveConversation(id) { state.activeId = id; saveState(); renderChatList(); renderMessages(); }
function renameConversationIfEmptyTitle(conv, proposal) {
  if (!conv) return;
  if (!conv.title || conv.title === "New chat") {
    const t = sanitize(proposal).trim();
    if (t) { conv.title = t.slice(0, 80); saveState(); renderChatList(); }
  }
}
function deleteConversation(id) {
  const idx = state.conversations.findIndex(c => c.id === id);
  if (idx === -1) return;
  state.conversations.splice(idx, 1);
  if (state.activeId === id) {
    state.activeId = state.conversations[0]?.id || null;
    if (!state.activeId) createConversation("New chat");
  }
  saveState(); renderChatList(); renderMessages();
}
function clearConversations() {
  state.conversations = [];
  state.activeId = null;
  saveState();
  createConversation("New chat");
}

// --- rendering
function renderChatList() {
  chatListEl.innerHTML = "";
  state.conversations.forEach(c => {
    const li = document.createElement("li");
    li.style.display = "flex"; li.style.gap = "6px"; li.style.alignItems = "center";

    // open/select button
    const openBtn = document.createElement("button");
    openBtn.textContent = c.title || "Untitled";
    openBtn.className = c.id === state.activeId ? "is-active" : "";
    openBtn.style.flex = "1 1 auto";
    openBtn.onclick = () => setActiveConversation(c.id);

    // compact red trash
    const delBtn = document.createElement("button");
    delBtn.title = "Delete chat";
    delBtn.setAttribute("aria-label", "Delete chat");
    delBtn.textContent = "🗑";
    delBtn.className = "btn btn--icon";
    // inline style: compact & red
    delBtn.style.width = "28px";
    delBtn.style.height = "28px";
    delBtn.style.padding = "0";
    delBtn.style.display = "grid";
    delBtn.style.placeItems = "center";
    delBtn.style.borderRadius = "8px";
    delBtn.style.border = "1px solid var(--danger)";
    delBtn.style.color = "var(--danger)";
    delBtn.style.background = "transparent";
    delBtn.addEventListener("mouseenter", () => {
      delBtn.style.background = "color-mix(in oklab, var(--danger) 20%, transparent)";
      delBtn.style.color = "#fff";
    });
    delBtn.addEventListener("mouseleave", () => {
      delBtn.style.background = "transparent";
      delBtn.style.color = "var(--danger)";
    });
    delBtn.onclick = (e) => {
      e.stopPropagation();
      const ok = confirm(`Delete chat "${c.title || "Untitled"}"?`);
      if (ok) deleteConversation(c.id);
    };

    li.appendChild(openBtn);
    li.appendChild(delBtn);
    chatListEl.appendChild(li);
  });

  ensureClearAllButton();
}

function ensureClearAllButton() {
  if (!sidebarFooter) return;
  if (document.getElementById("clearAllBtn")) return;
  const btn = document.createElement("button");
  btn.id = "clearAllBtn";
  btn.className = "btn btn--ghost";
  btn.textContent = "Clear chats";
  btn.title = "Delete all conversations";
  btn.onclick = () => { if (confirm("Delete all chats?")) clearConversations(); };
  sidebarFooter.appendChild(btn);
}

function renderMessages() {
  const conv = getActiveConversation();
  messagesEl.innerHTML = "";

  if (!conv.messages.length) {
    const placeholder = document.createElement("div");
    placeholder.className = "messages__placeholder";
    placeholder.innerHTML = `
      <p>Ask something like: <em>“What does Alice like to eat?”</em></p>
      <p class="note">Make sure you’ve ingested your library files first.</p>
    `;
    messagesEl.appendChild(placeholder);
    return;
  }

  conv.messages.forEach((m, idx) => messagesEl.appendChild(renderMessageBubble(m, idx)));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderMessageBubble(m, idx) {
  const wrapper = document.createElement("div");
  wrapper.className = "msg " + (m.role === "user" ? "msg--user" : "msg--assistant");

  const row = document.createElement("div");
  row.className = "msg__row";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = m.role === "user" ? "U" : "A";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  // content
  const content = document.createElement("div");
  if (m.content && m.content.includes("```")) {
    const parts = m.content.split("```");
    parts.forEach((part, i) => {
      if (i % 2 === 1) {
        const pre = document.createElement("pre");
        pre.textContent = part.replace(/^\w+\n/, "");
        content.appendChild(pre);
      } else {
        const p = document.createElement("p"); p.textContent = part; content.appendChild(p);
      }
    });
  } else {
    const p = document.createElement("p"); p.textContent = m.content ?? ""; content.appendChild(p);
  }
  bubble.appendChild(content);

  // meta (retrieved/context)
  if (m.meta && (m.meta.retrieved?.length || m.meta.context)) {
    const meta = document.createElement("div"); meta.className = "meta";
    if (Array.isArray(m.meta.retrieved) && m.meta.retrieved.length) {
      const hits = document.createElement("div"); hits.className = "hits";
      m.meta.retrieved.forEach(h => {
        const chip = document.createElement("span");
        chip.className = "hit";
        const score = typeof h.score === "number" ? ` (${h.score.toFixed(3)})` : "";
        chip.textContent = `${h.source}#${h.chunkIndex}${score}`;
        hits.appendChild(chip);
      });
      meta.appendChild(hits);
    }
    if (m.meta.context) {
      const btn = document.createElement("button");
      btn.className = "btn btn--ghost small"; btn.textContent = "Show context";
      btn.onclick = () => { contextText.textContent = m.meta.context; contextModal.showModal(); };
      meta.appendChild(btn);
    }
    bubble.appendChild(meta);
  }

  row.appendChild(avatar); row.appendChild(bubble); wrapper.appendChild(row);
  return wrapper;
}

function appendMessage(role, content, meta) {
  const conv = getActiveConversation();
  conv.messages.push({ role, content, meta });
  conv.updatedAt = nowIso();
  saveState(); renderMessages();
}

// --- SSE parsing (keep raw payload; no trimming) ---
function parseSSEChunk(block) {
  let event = "message";
  const dataLines = [];
  block.split(/\r?\n/).forEach(line => {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5)); // exact bytes after "data:"
    }
  });
  return { event, data: dataLines.join("\n") };
}

async function postSSE(path, bodyObj, query) {
  const url = new URL(path, window.location.origin);
  if (query) for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== null) url.searchParams.set(k, String(v));

  const res = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj)
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(text || res.statusText || "Request failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  return {
    async *events() {
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          if (buffer !== "") {
            for (const block of buffer.split(/\n\n/)) if (block !== "") yield parseSSEChunk(block);
          }
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          if (block !== "") yield parseSSEChunk(block);
        }
      }
    },
    cancel() { try { reader.cancel(); } catch {} }
  };
}

// --- actions
let activeStream = null;

async function sendMessage() {
  const text = sanitize(composerInput.value).trim();
  if (!text) return;

  composerInput.value = ""; autoResizeTextarea(composerInput);
  const prevLabel = sendBtn.textContent; sendBtn.textContent = "Sending…"; sendBtn.disabled = true;

  if (activeStream && activeStream.cancel) { try { activeStream.cancel(); } catch {} }

  appendMessage("user", text);

  const conv = getActiveConversation();
  appendMessage("assistant", "", { retrieved: [], context: null });
  const assistantIndex = conv.messages.length - 1;

  // --- NEW: show Stop button while streaming ---
  userCanceled = false;
  const stopBtn = ensureStopButton();
  stopBtn.style.display = "inline-flex";
  stopBtn.disabled = false;
  stopBtn.textContent = "Stop";

  try {
    const topK = clamp(parseInt(topKInput.value, 10) || 6, 1, 50);
    const debug = true;
    const payload = { messages: conv.messages.map(m => ({ role: m.role, content: m.content })) };

    const stream = await postSSE("/api/chat/stream", payload, { topK, debug /* , think: 'FAST' etc. */ });
    activeStream = stream;

    for await (const ev of stream.events()) {
		if (ev.event === "meta") {
		  const meta = JSON.parse(ev.data);

		  const conv = getActiveConversation();
		  const msg  = conv.messages[assistantIndex];   // the assistant bubble we just created
		  msg.meta = msg.meta || {};                    // <-- ensure meta object

		  // write where renderMessageBubble expects them:
		  msg.meta.retrieved = Array.isArray(meta.retrieved) ? meta.retrieved : [];
		  msg.meta.context   = (typeof meta.context === 'string' && meta.context.trim()) ? meta.context : null;

		  // optional extras
		  msg.model = meta.model || msg.model;
		  msg.think = meta.think || msg.think;

		  saveState();
		  renderMessages();
		  continue;
		} else if (ev.event === "delta") {
        const prev = conv.messages[assistantIndex].content;
        conv.messages[assistantIndex].content = prev + ev.data; // your spacing fix
        saveState(); renderMessages();
        messagesEl.scrollTop = messagesEl.scrollHeight;
      } else if (ev.event === "done") {
        // (unchanged)
      } else if (ev.event === "error") {
        conv.messages[assistantIndex].content = `Error: ${ev.data}`;
        saveState(); renderMessages();
      }
    }
  } catch (err) {
    const conv2 = getActiveConversation();
    // --- NEW: don't show an error if the user pressed Stop ---
    if (!userCanceled) {
      conv2.messages[assistantIndex].content = `Error: ${err.message || err}`;
      saveState(); renderMessages();
    }
  } finally {
    sendBtn.textContent = prevLabel; sendBtn.disabled = false; composerInput.focus();
    // --- NEW: hide Stop button when finished or canceled ---
    if (stopBtn) { stopBtn.style.display = "none"; stopBtn.disabled = true; stopBtn.textContent = "Stop"; }
  }
}


async function reloadIndex() {
  const prev = reloadIndexBtn.textContent;
  reloadIndexBtn.textContent = "Reloading…"; reloadIndexBtn.disabled = true;
  try {
    const r = await apiPost("/api/index/load", {}, { clear: true, batchSize: 200, logEvery: 100 });
    toast(`Loaded ${r.loadedRecords} records in ${r.seconds.toFixed ? r.seconds.toFixed(2) : r.seconds}s`);
  } catch (e) { toast("Index reload failed: " + (e.message || e), true); }
  finally { reloadIndexBtn.textContent = prev; reloadIndexBtn.disabled = false; }
}

async function ingestAll() {
  const prev = ingestAllBtn.textContent;
  ingestAllBtn.textContent = "Ingesting…"; ingestAllBtn.disabled = true;
  try {
    const r = await apiPost("/api/ingest/all", {}, {});
    toast(`Ingested ${r.files} file(s), ${r.chunks} chunk(s).`);
  } catch (e) { toast("Ingest failed: " + (e.message || e), true); }
  finally { ingestAllBtn.textContent = prev; ingestAllBtn.disabled = false; }
}

// non-stream helper we still use
async function apiPost(path, body, params) {
  const url = new URL(path, window.location.origin);
  if (params) Object.entries(params).forEach(([k, v]) => v != null && url.searchParams.set(k, String(v)));
  const res = await fetch(url.toString(), {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {})
  });
  if (!res.ok) { throw new Error((await res.text().catch(() => "")) || res.statusText); }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

// --- toast
let toastTimer = null;
function toast(msg, danger = false) {
  let el = document.getElementById("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    el.style.position = "fixed";
    el.style.bottom = "16px"; el.style.right = "16px";
    el.style.padding = "10px 14px";
    el.style.borderRadius = "12px";
    el.style.border = "1px solid var(--border)";
    el.style.background = "var(--panel)"; el.style.color = "var(--text)";
    el.style.boxShadow = "var(--shadow)"; el.style.transition = "opacity .25s ease";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.style.borderColor = danger ? "var(--danger)" : "var(--border)";
  clearTimeout(toastTimer); el.style.opacity = "1";
  toastTimer = setTimeout(() => { el.style.opacity = "0"; }, 3500);
}

// --- events
composerForm.addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
composerInput.addEventListener("input", () => autoResizeTextarea(composerInput));
composerInput.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
newChatBtn.addEventListener("click", () => createConversation("New chat"));
reloadIndexBtn.addEventListener("click", reloadIndex);
ingestAllBtn.addEventListener("click", ingestAll);
closeContextBtn.addEventListener("click", () => contextModal.close());

// --- init
(function init() {
  loadState();
  if (state.conversations.length === 0) createConversation("New chat");
  else { renderChatList(); renderMessages(); }
  autoResizeTextarea(composerInput);
  setModelName("(awaiting reply)");
  ensureClearAllButton();
})();
