// Knowledge Copilot front end: streams the agent's steps, renders cited answers,
// and manages the knowledge base. No frameworks, no inline scripts (strict CSP).
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    messages: $("messages"), welcome: $("welcome"), composer: $("composer"), question: $("question"),
    send: $("send"), steps: $("steps"), summary: $("summary"), sumSource: $("sumSource"),
    sumCheck: $("sumCheck"), sumQuery: $("sumQuery"), sumTime: $("sumTime"), readyDot: $("readyDot"),
    readyText: $("readyText"), kbList: $("kbList"), kbCount: $("kbCount"), newChat: $("newChat"),
    dialog: $("addDialog"), openAdd: $("openAdd"), closeAdd: $("closeAdd"), tabFile: $("tabFile"),
    tabUrl: $("tabUrl"), paneFile: $("paneFile"), paneUrl: $("paneUrl"), fileInput: $("fileInput"),
    urlInput: $("urlInput"), adminKey: $("adminKey"), addBtn: $("addBtn"), addStatus: $("addStatus"),
    accessBar: $("accessBar"), accessCode: $("accessCode"),
  };
  const accessRequired = document.body.dataset.accessRequired === "true";

  const STEP_LABELS = {
    route_question: "Route the question",
    retrieve_kb: "Search the knowledge base",
    grade_kb: "Grade the passages",
    rewrite_query: "Rewrite the search",
    web_search: "Search the web",
    grade_web: "Grade web results",
    generate: "Write the answer",
    check_answer: "Fact-check the answer",
    direct_answer: "Reply directly",
    insufficient: "Not enough evidence",
    finalize: "Finish",
  };
  const SOURCE_LABELS = {
    private_kb: ["From your knowledge base", "kb"],
    web_search: ["From the web", "web"],
    "private_kb + web_search": ["Knowledge base and web", "mixed"],
    direct: ["Direct reply", "neutral"],
    insufficient_evidence: ["Not enough evidence", "bad"],
    uncited: ["Answer without citations", "bad"],
  };

  // ---------- Small helpers ----------
  const store = {
    get(key) { try { return sessionStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { sessionStorage.setItem(key, value); } catch { /* private mode: keep in memory */ } },
  };
  const newId = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).replace(/[^a-z0-9]/gi, "").slice(0, 32);
  let threadId = store.get("thread_id") || newId();
  store.set("thread_id", threadId);
  let busy = false;
  let msgCount = 0;
  if (accessRequired) {
    els.accessBar.hidden = false;
    els.accessCode.value = store.get("access_code") || "";
    els.accessCode.addEventListener("change", () => store.set("access_code", els.accessCode.value.trim()));
  }

  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  const escapeHtml = (s = "") => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Minimal, safe Markdown: text is escaped first, then a few patterns become tags.
  function renderMarkdown(text, msgId) {
    const blocks = [];
    const codeStash = [];
    let src = escapeHtml(text).replace(/```[a-z]*\n?([\s\S]*?)```/g, (_, code) => {
      codeStash.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
      return `\u0000${codeStash.length - 1}\u0000`;
    });
    const inline = (s) => s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\s?\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) => `<span class="cites">${nums.split(",").map((n) => {
        const k = n.trim();
        return `<a class="cite" href="#src-${msgId}-${k}" aria-label="Source ${k}">${k}</a>`;
      }).join(",")}</span>`);

    for (const para of src.split(/\n{2,}/)) {
      const lines = para.split("\n").filter((l) => l.trim() !== "");
      if (!lines.length) continue;
      if (/^\u0000\d+\u0000$/.test(lines.join("").trim())) { blocks.push(codeStash[Number(lines.join("").trim().slice(1, -1))]); continue; }
      if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
        blocks.push(`<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*[-*]\s+/, ""))}</li>`).join("")}</ul>`);
      } else if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
        blocks.push(`<ol>${lines.map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s+/, ""))}</li>`).join("")}</ol>`);
      } else if (lines[0].startsWith("⚠️")) {
        blocks.push(`<div class="warn">${inline(lines.join(" "))}</div>`);
      } else {
        blocks.push(`<p>${inline(lines.join("<br>"))}</p>`);
      }
    }
    return blocks.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => codeStash[Number(i)]);
  }

  function scrollDown() { els.messages.scrollTop = els.messages.scrollHeight; }

  // ---------- Messages ----------
  function addUserMessage(text) {
    els.welcome?.remove();
    const div = document.createElement("div");
    div.className = "msg msg-user";
    div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    els.messages.appendChild(div);
    scrollDown();
  }

  function addPendingAnswer() {
    const div = document.createElement("div");
    div.className = "msg msg-bot";
    div.innerHTML = `<div class="answer"><p class="thinking">Working on it…</p></div>`;
    els.messages.appendChild(div);
    scrollDown();
    return div;
  }

  function fillAnswer(div, data) {
    const id = ++msgCount;
    const [label, kind] = SOURCE_LABELS[data.source_used] || [data.source_used, "neutral"];
    const sources = (data.citations || []).map((c) => {
      const isWeb = c.type === "Web";
      const title = escapeHtml(c.title || c.url || "Source");
      const link = c.url && /^https?:\/\//.test(c.url) ? `<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer">${title}</a>` : title;
      const section = c.section ? ` <span class="src-sec">${escapeHtml(c.section.split(" > ").slice(-1)[0])}</span>` : "";
      return `<li id="src-${id}-${c.id}"><span class="src-num">${c.id}</span><span><span class="src-kind ${isWeb ? "web" : ""}">${isWeb ? "Web" : "KB"}</span>${link}${section}</span></li>`;
    }).join("");
    div.innerHTML = `
      <span class="badge ${kind}">${escapeHtml(label)}</span>
      <div class="answer">${renderMarkdown(data.answer || "", id)}</div>
      ${sources ? `<div class="sources"><ol>${sources}</ol></div>` : ""}`;
    scrollDown();
  }

  function failAnswer(div, message) {
    div.innerHTML = `<p class="error">${escapeHtml(message)}</p>`;
    scrollDown();
  }

  // ---------- Evidence path ----------
  function resetPath() {
    els.steps.innerHTML = "";
    els.summary.hidden = true;
  }

  function addStep(node, line) {
    els.steps.querySelector(".running")?.classList.replace("running", "done");
    const li = document.createElement("li");
    const tone = node === "web_search" || node === "grade_web" ? "web" : node === "insufficient" ? "bad" : "";
    li.className = `step running ${tone}`.trim();
    li.innerHTML = `<span class="step-node">${escapeHtml(STEP_LABELS[node] || node)}</span><span class="step-line">${escapeHtml(line)}</span>`;
    els.steps.appendChild(li);
  }

  function finishPath(data) {
    els.steps.querySelectorAll(".running").forEach((li) => li.classList.replace("running", "done"));
    const [label] = SOURCE_LABELS[data.source_used] || [data.source_used];
    els.sumSource.textContent = label;
    els.sumCheck.textContent = data.source_used === "direct" ? "Not needed" : data.grounded ? "Every claim matched a source" : "Some claims unverified";
    els.sumQuery.textContent = data.standalone_question || "";
    els.sumTime.textContent = `${data.latency_s.toFixed(1)} s`;
    els.summary.hidden = false;
  }

  // ---------- Asking (Server-Sent Events over fetch) ----------
  async function ask(question) {
    if (busy || !question.trim()) return;
    if (accessRequired) store.set("access_code", els.accessCode.value.trim());
    const code = accessRequired ? els.accessCode.value.trim() : "";
    if (accessRequired && !code) { els.accessCode.focus(); return; }
    busy = true;
    els.send.disabled = true;
    addUserMessage(question);
    const pending = addPendingAnswer();
    resetPath();

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(code ? { "X-Access-Code": code } : {}) },
        body: JSON.stringify({ question, thread_id: threadId }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const detail = typeof body.detail === "string" ? body.detail : "The request was rejected. Check the question and try again.";
        if (res.status === 401) throw new Error("That access code isn't valid. Check it and ask again.");
        if (res.status === 429 && !/daily/i.test(detail)) throw new Error("You're asking quickly. Wait a few seconds, then ask again.");
        throw new Error(detail);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let cut;
        while ((cut = buffer.indexOf("\n\n")) >= 0) {
          const raw = buffer.slice(0, cut);
          buffer = buffer.slice(cut + 2);
          const event = (raw.match(/^event: (.*)$/m) || [])[1];
          const data = JSON.parse((raw.match(/^data: (.*)$/m) || [, "{}"])[1]);
          if (event === "step") addStep(data.node, data.line);
          else if (event === "final") { fillAnswer(pending, data); finishPath(data); finished = true; }
          else if (event === "error") throw new Error(`${data.detail} (reference ${data.request_id})`);
        }
      }
      if (!finished) throw new Error("The connection closed before the answer arrived. Ask again.");
    } catch (err) {
      failAnswer(pending, err.message || "Something went wrong. Ask again.");
      els.steps.querySelector(".running")?.classList.replace("running", "done");
    } finally {
      busy = false;
      els.send.disabled = false;
      els.question.focus();
    }
  }

  els.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = els.question.value;
    if (accessRequired && !els.accessCode.value.trim()) { els.accessCode.focus(); return; }  // keep the typed question
    els.question.value = "";
    autoGrow();
    ask(q);
  });
  els.question.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); els.composer.requestSubmit(); }
  });
  function autoGrow() { els.question.style.height = "auto"; els.question.style.height = `${Math.min(els.question.scrollHeight, 180)}px`; }
  els.question.addEventListener("input", autoGrow);
  document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => ask(b.textContent.trim())));

  els.newChat.addEventListener("click", () => {
    threadId = newId();
    store.set("thread_id", threadId);
    els.messages.innerHTML = "";
    resetPath();
    els.steps.innerHTML = '<li class="step-empty">New conversation started. Earlier questions are forgotten.</li>';
    els.question.focus();
  });

  // ---------- Status and knowledge base ----------
  async function checkReady() {
    try {
      const res = await fetch("/api/ready");
      const data = await res.json();
      if (res.ok) {
        els.readyDot.className = "ready-dot is-ready";
        els.readyText.textContent = `Ready: ${plural(data.chunks, "passage")} from ${plural(data.sources, "document")}${data.web_enabled ? ", web fallback on" : ""}`;
        loadSources();
        return;
      }
      const starting = (data.reason || "").toLowerCase() === "starting";
      els.readyDot.className = starting ? "ready-dot" : "ready-dot is-down";
      els.readyText.textContent = starting ? "Loading models and connecting…" : `Not available: ${data.reason}`;
    } catch {
      els.readyDot.className = "ready-dot is-down";
      els.readyText.textContent = "Can't reach the server";
    }
    setTimeout(checkReady, 3000);
  }

  async function loadSources() {
    try {
      const res = await fetch("/api/sources");
      if (!res.ok) return;
      const items = await res.json();
      els.kbCount.textContent = items.length ? String(items.length) : "";
      els.kbList.innerHTML = items.length
        ? items.map((s) => `<li title="${escapeHtml(s.source)}"><span class="kb-title">${escapeHtml(s.title || s.source)}</span><span class="kb-meta">${plural(s.chunks, "passage")}</span></li>`).join("")
        : '<li class="kb-empty">No documents yet. Add one to start.</li>';
    } catch { /* status line already shows connection problems */ }
  }

  // ---------- Add a document ----------
  let mode = "file";
  function setMode(next) {
    mode = next;
    els.tabFile.classList.toggle("is-active", next === "file");
    els.tabUrl.classList.toggle("is-active", next === "url");
    els.tabFile.setAttribute("aria-selected", String(next === "file"));
    els.tabUrl.setAttribute("aria-selected", String(next === "url"));
    els.paneFile.hidden = next !== "file";
    els.paneUrl.hidden = next !== "url";
  }
  els.tabFile.addEventListener("click", () => setMode("file"));
  els.tabUrl.addEventListener("click", () => setMode("url"));
  els.openAdd.addEventListener("click", () => { els.addStatus.textContent = ""; els.dialog.showModal(); });
  els.closeAdd.addEventListener("click", () => els.dialog.close());

  function status(text, tone = "") { els.addStatus.textContent = text; els.addStatus.className = `add-status ${tone}`; }

  els.addBtn.addEventListener("click", async () => {
    const key = els.adminKey.value.trim();
    if (!key) return status("Enter the admin key first.", "fail");
    let request;
    if (mode === "file") {
      const file = els.fileInput.files[0];
      if (!file) return status("Choose a file first.", "fail");
      if (file.size > 10 * 1024 * 1024) return status("That file is over 10 MB.", "fail");
      const form = new FormData();
      form.append("file", file);
      request = fetch("/api/ingest/file", { method: "POST", headers: { "X-Admin-Key": key }, body: form });
    } else {
      const url = els.urlInput.value.trim();
      if (!/^https?:\/\//.test(url)) return status("Enter a full address starting with https://", "fail");
      request = fetch("/api/ingest/url", { method: "POST", headers: { "X-Admin-Key": key, "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
    }
    els.addBtn.disabled = true;
    status("Reading, splitting and indexing…");
    try {
      const res = await request;
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(res.status === 401 ? "That admin key isn't valid." : typeof data.detail === "string" ? data.detail : "The document couldn't be added.");
      status(`Added "${data.title}": ${plural(data.chunks, "passage")} (${data.uploaded} new, ${data.unchanged} unchanged, ${data.removed} removed).`, "ok");
      loadSources();
      checkReady();
    } catch (err) {
      status(err.message, "fail");
    } finally {
      els.addBtn.disabled = false;
    }
  });

  checkReady();
})();
