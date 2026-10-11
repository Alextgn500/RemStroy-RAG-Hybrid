"""Настройки приложения RemStroy RAG Hybrid (RouterAI)."""

import os

# ---------- Корневые пути ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

KNOWLEDGE_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

CHROMA_PATH = os.environ.get(
    "CHROMA_PATH",
    os.path.join(PROJECT_ROOT, "chroma_db"),
)
COLLECTION_NAME = "remstroy_knowledge"

# ---------- RouterAI (OpenAI-совместимый API) ----------
ROUTERAI_API_KEY = os.environ.get("ROUTERAI_API_KEY", "")
BASE_URL = "https://routerai.ru/api/v1"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "deepseek/deepseek-v4.1-flash")

TEMPERATURE = 0.3
MAX_TOKENS = 2048

CHUNK_MAX_PARAGRAPHS = 3
CHUNK_OVERLAP = 1
TOP_K = 5
RRF_K = 60

HISTORY_MAX_MESSAGES = 10

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))
