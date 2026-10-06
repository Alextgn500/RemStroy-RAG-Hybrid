"""Настройки приложения RemStroy RAG Hybrid.

Пути определяются относительно расположения файла, поэтому проект
запускается и из Colab, и локально, и на сервере без правок.
"""

import os

# ---------- Корневые пути ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                     # корень проекта

KNOWLEDGE_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

# ChromaDB — можно переопределить через переменную окружения,
# если на сервере удобнее хранить данные в отдельной директории
CHROMA_PATH = os.environ.get(
    "CHROMA_PATH",
    os.path.join(PROJECT_ROOT, "chroma_db"),
)
COLLECTION_NAME = "remstroy_knowledge"

# ---------- Gemini API (через OpenAI-совместимый интерфейс) ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")

# ---------- Генерация ----------
TEMPERATURE = 0.3
MAX_TOKENS = 2048   # Gemini иногда игнорирует меньшие значения и обрывает ответ

# ---------- RAG ----------
CHUNK_MAX_PARAGRAPHS = 3
CHUNK_OVERLAP = 1
TOP_K = 5
RRF_K = 60

# ---------- Память диалога ----------
HISTORY_MAX_MESSAGES = 10

# ---------- Сервер ----------
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))
