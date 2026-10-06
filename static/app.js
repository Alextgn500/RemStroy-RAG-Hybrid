"use strict";

const chatEl = document.getElementById("chat");
const questionEl = document.getElementById("question");
const sendBtn = document.getElementById("btn-send");
const reindexBtn = document.getElementById("btn-reindex");
const resetBtn = document.getElementById("btn-reset");
const statusEl = document.getElementById("status");

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function appendUserMessage(text) {
  const wrap = document.createElement("div");
  wrap.className = "message user";
  wrap.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatEl.appendChild(wrap);
  scrollToBottom();
}

function renderChunks(titleText, badgeClass, badgeLabel, chunks) {
  if (!Array.isArray(chunks) || chunks.length === 0) return null;

  const section = document.createElement("div");
  section.className = "rag-section";

  const title = document.createElement("div");
  title.className = "rag-section-title";
  title.innerHTML = `${titleText} <span class="rag-badge ${badgeClass}">${badgeLabel}</span>`;
  section.appendChild(title);

  chunks.forEach((c, i) => {
    const item = document.createElement("div");
    item.className = "chunk";

    const head = document.createElement("div");
    head.className = "chunk-head";
    let scoreLabel = "";
    if (c.score !== undefined) scoreLabel = `score: ${c.score}`;
    else if (c.bm25_score !== undefined) scoreLabel = `BM25: ${c.bm25_score}`;
    else if (c.rrf_score !== undefined) scoreLabel = `RRF: ${c.rrf_score}`;

    head.innerHTML =
      `<span>#${i + 1} · ${escapeHtml(c.source || "unknown")}` +
      ` · чанк ${c.chunk ?? "-"}</span>` +
      `<span class="chunk-score">${scoreLabel}</span>`;
    item.appendChild(head);

    const txt = document.createElement("div");
    txt.className = "chunk-text";
    txt.textContent = c.text;
    item.appendChild(txt);

    section.appendChild(item);
  });

  return section;
}

function appendAssistantMessage(answer, data) {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const answerDiv = document.createElement("div");
  answerDiv.textContent = answer;
  bubble.appendChild(answerDiv);

  const ragBox = document.createElement("div");
  ragBox.className = "rag-logic";

  const ragTitle = document.createElement("div");
  ragTitle.className = "rag-logic-title";
  ragTitle.textContent = "▼ Логика работы RAG";
  ragBox.appendChild(ragTitle);

  const ragContent = document.createElement("div");

  // 1. Ключевые слова
  if (Array.isArray(data.keywords) && data.keywords.length > 0) {
    const kwSection = document.createElement("div");
    kwSection.className = "rag-section";
    kwSection.innerHTML = `
      <div class="rag-section-title">
        Ключевые слова из вопроса
        <span class="rag-badge keyword">без LLM</span>
      </div>
      <div class="keywords">
        ${data.keywords.map(k => `<span class="keyword-tag">${escapeHtml(k)}</span>`).join("")}
      </div>`;
    ragContent.appendChild(kwSection);
  }

  // 2. Embedding-поиск
  const embedSection = renderChunks(
    "Embedding-поиск (топ-5)",
    "embed", "cosine",
    data.embedding_chunks || data.chunks || []
  );
  if (embedSection) ragContent.appendChild(embedSection);

  // 3. BM25
  const kwChunksSection = renderChunks(
    "Поиск по ключевым словам (BM25)",
    "keyword", "BM25",
    data.keyword_chunks || []
  );
  if (kwChunksSection) ragContent.appendChild(kwChunksSection);

  // 4. Финальный набор (RRF)
  if (Array.isArray(data.final_chunks) && data.final_chunks.length > 0) {
    const finalSection = renderChunks(
      "Финальный набор после объединения (RRF)",
      "final", "→ в LLM",
      data.final_chunks
    );
    if (finalSection) ragContent.appendChild(finalSection);
  }

  ragBox.appendChild(ragContent);
  bubble.appendChild(ragBox);

  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  scrollToBottom();

  ragTitle.addEventListener("click", () => {
    const open = ragContent.style.display !== "none";
    ragContent.style.display = open ? "none" : "block";
    ragTitle.textContent = (open ? "▶" : "▼") + " Логика работы RAG";
  });
}

function appendError(text) {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";
  wrap.innerHTML = `<div class="bubble error">${escapeHtml(text)}</div>`;
  chatEl.appendChild(wrap);
  scrollToBottom();
}

function appendLoading() {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";
  wrap.innerHTML = `<div class="bubble loading">Ищу ответ…</div>`;
  chatEl.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function setBusy(isBusy) {
  sendBtn.disabled = isBusy;
  sendBtn.textContent = isBusy ? "Отправка…" : "Спросить";
}

async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const data = await r.json();
    statusEl.textContent = `чанков в индексе: ${data.indexed_chunks}`;
  } catch (e) {
    statusEl.textContent = "";
  }
}

async function ask() {
  const question = questionEl.value.trim();
  if (!question) return;

  appendUserMessage(question);
  questionEl.value = "";
  setBusy(true);

  const loading = appendLoading();

  try {
    const resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await resp.json();
    loading.remove();

    if (data.error) {
      appendError("Ошибка: " + data.error);
    } else {
      appendAssistantMessage(data.answer || "", data);
    }
  } catch (e) {
    loading.remove();
    appendError("Не удалось получить ответ: " + e.message);
  } finally {
    setBusy(false);
    questionEl.focus();
  }
}

async function reindex() {
  reindexBtn.disabled = true;
  reindexBtn.textContent = "Индексация…";
  try {
    const r = await fetch("/api/reindex", { method: "POST" });
    const data = await r.json();
    if (data.error) {
      appendError("Ошибка индексации: " + data.error);
    } else {
      appendAssistantMessage(
        `База знаний проиндексирована. Чанков: ${data.indexed_chunks}.`,
        { chunks: [] }
      );
      refreshStatus();
    }
  } catch (e) {
    appendError("Не удалось переиндексировать: " + e.message);
  } finally {
    reindexBtn.disabled = false;
    reindexBtn.textContent = "Переиндексировать";
  }
}

async function resetChat() {
  try { await fetch("/api/reset", { method: "POST" }); } catch (e) {}
  chatEl.innerHTML = "";
  appendAssistantMessage("Диалог сброшен. Задайте новый вопрос.", { chunks: [] });
}

sendBtn.addEventListener("click", ask);
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});
reindexBtn.addEventListener("click", reindex);
resetBtn.addEventListener("click", resetChat);

refreshStatus();
