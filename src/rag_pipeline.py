"""RAG-логика: чанкинг, индексация, гибридный поиск (BM25 + embedding + RRF), генерация."""

import os
import re
import glob
from collections import Counter

from rank_bm25 import BM25Okapi

from config import (
    KNOWLEDGE_DIR, CHUNK_MAX_PARAGRAPHS, CHUNK_OVERLAP, TOP_K,
    RRF_K, HISTORY_MAX_MESSAGES,
)
from openai_client import get_embedding, get_embeddings_batch, chat
from chroma_store import (
    add_chunks, query_chunks, reset_collection, count, get_all_documents,
)

conversation_history = []

# ---------- Русские стоп-слова ----------
STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
    "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
    "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для",
    "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
    "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж",
    "тогда", "кто", "этот", "того", "потому", "этого", "какой",
    "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем",
    "всех", "никогда", "можно", "при", "наконец", "два", "об",
    "другой", "хоть", "после", "над", "больше", "тот", "через",
    "эти", "нас", "про", "всего", "них", "какая", "много",
    "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою",
    "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "также", "какие", "сколько", "которые", "который", "которая",
    "которое", "которых", "ваш", "ваша", "ваше", "ваши",
    "наш", "наша", "наше", "наши", "эта", "те", "та",
}

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def tokenize(text):
    """Простая токенизация с фильтрацией стоп-слов."""
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ---------- Чанкинг ----------

def smart_chunk(text, max_paragraphs=CHUNK_MAX_PARAGRAPHS, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    i = 0
    while i < len(paragraphs):
        end = min(i + max_paragraphs, len(paragraphs))
        chunk = "\n\n".join(paragraphs[i:end])
        chunks.append(chunk)
        if end >= len(paragraphs):
            break
        i = end - overlap
    return chunks


def load_knowledge_files():
    files = []
    pattern = os.path.join(KNOWLEDGE_DIR, "*.md")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as fh:
            files.append((os.path.basename(path), fh.read()))
    return files


# ---------- BM25 ----------

_bm25_index = None
_bm25_docs = None
_bm25_metas = None
_bm25_ids = None


def build_bm25_index():
    global _bm25_index, _bm25_docs, _bm25_metas, _bm25_ids

    ids, docs, metas = get_all_documents()
    if not docs:
        _bm25_index = None
        return 0

    tokenized = [tokenize(d) for d in docs]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_docs = docs
    _bm25_metas = metas
    _bm25_ids = ids
    return len(docs)


def search_bm25(query, n_results=TOP_K):
    global _bm25_index
    if _bm25_index is None:
        build_bm25_index()
    if _bm25_index is None:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    scores = _bm25_index.get_scores(tokens)
    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:n_results]

    hits = []
    for idx, score in ranked:
        if score <= 0:
            continue
        meta = _bm25_metas[idx]
        hits.append({
            "text": _bm25_docs[idx],
            "source": meta.get("source", "unknown"),
            "chunk": meta.get("chunk", -1),
            "bm25_score": round(float(score), 4),
        })
    return hits


# ---------- Ключевые слова без LLM ----------

def extract_keywords(question, top_n=6):
    tokens = tokenize(question)
    if not tokens:
        return []
    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
    freq = Counter(tokens)
    seen.sort(key=lambda w: (-len(w), -freq[w]))
    return seen[:top_n]


# ---------- Индексация ----------

def build_index(verbose=True):
    reset_collection()
    files = load_knowledge_files()
    if not files:
        if verbose:
            print("[RAG] Папка knowledge_base пуста.")
        return 0

    ids, documents, metadatas, embeddings = [], [], [], []
    for filename, text in files:
        chunks = smart_chunk(text)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{filename}::{idx}")
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk": idx})

    BATCH = 64
    for start in range(0, len(documents), BATCH):
        batch = documents[start:start + BATCH]
        embeddings.extend(get_embeddings_batch(batch))

    add_chunks(ids, documents, embeddings, metadatas)
    build_bm25_index()

    if verbose:
        print(f"[RAG] Проиндексировано чанков: {len(ids)}")
    return len(ids)


def ensure_index():
    if count() == 0:
        build_index()


# ---------- Embedding-поиск ----------

def search_embedding(query, n_results=TOP_K):
    query_emb = get_embedding(query)
    results = query_chunks(query_emb, n_results=n_results)
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    hits = []
    for doc, dist, meta in zip(docs, dists, metas):
        score = 1.0 - float(dist)
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk": meta.get("chunk", -1),
            "score": round(score, 4),
        })
    return hits


# ---------- RRF ----------

def reciprocal_rank_fusion(*ranked_lists, k=RRF_K, top_n=TOP_K):
    scores = {}
    payload = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = (item["source"], item["chunk"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in payload:
                payload[key] = item

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    result = []
    for key, rrf_score in ordered:
        item = dict(payload[key])
        item["rrf_score"] = round(rrf_score, 6)
        result.append(item)
    return result


# ---------- Генерация ----------

SYSTEM_PROMPT = (
    "Ты — ассистент компании RemStroy. Компания занимается ремонтом "
    "жилых квартир и установкой оконных и балконных блоков "
    "(2- и 3-слойных) по различным технологиям.\n\n"
    "ПРАВИЛА:\n"
    "1. Отвечай ТОЛЬКО на последний заданный вопрос. "
    "Игнорируй предыдущие вопросы, если они не относятся к делу.\n"
    "2. Не начинай ответ с фраз про отсутствие информации, "
    "если ответ на заданный вопрос есть в контексте.\n"
    "3. Не упоминай темы, которых нет в текущем вопросе.\n"
    "4. Если информации действительно нет — скажи об этом одной "
    "краткой фразой и остановись.\n"
    "5. Отвечай на русском языке, по существу, без вводных слов.\n"
    "6. Если уместно — структурируй ответ списком или выделяй ключевые цифры."
)


def generate_answer(question, hits):
    context_parts = []
    for i, h in enumerate(hits, 1):
        context_parts.append(
            f"[Фрагмент {i} | источник: {h['source']}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Контекст:\n{context}\n\nВопрос: {question}",
        },
    ]
    return chat(messages)


# ---------- Основной вход ----------

def rag_query(question):
    ensure_index()

    keywords = extract_keywords(question)
    embedding_hits = search_embedding(question, n_results=TOP_K)
    keyword_hits = search_bm25(question, n_results=TOP_K)
    final_hits = reciprocal_rank_fusion(
        embedding_hits, keyword_hits, k=RRF_K, top_n=TOP_K
    )

    answer = generate_answer(question, final_hits)

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})
    if len(conversation_history) > HISTORY_MAX_MESSAGES:
        conversation_history[:] = conversation_history[-HISTORY_MAX_MESSAGES:]

    return {
        "keywords": keywords,
        "embedding_chunks": embedding_hits,
        "keyword_chunks": keyword_hits,
        "final_chunks": final_hits,
        "chunks": final_hits,
        "answer": answer,
    }


def reset_history():
    conversation_history.clear()
