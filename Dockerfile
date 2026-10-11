# Базовый образ Python
FROM python:3.11-slim

# Метаданные
LABEL maintainer="Alextgn500"
LABEL description="RemStroy RAG Hybrid - RAG assistant with hybrid search"

# Переменные окружения Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Рабочая директория
WORKDIR /app

# Системные зависимости (нужны для chromadb / onnxruntime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем requirements — для кэширования слоёв
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Копируем код проекта
COPY src/ ./src/
COPY knowledge_base/ ./knowledge_base/
COPY static/ ./static/

# Создаём директорию для ChromaDB (будет смонтирована как volume)
RUN mkdir -p /app/chroma_db

# Переменные окружения приложения (переопределяются через docker-compose)
ENV ROUTERAI_API_KEY="" \
    CHAT_MODEL="deepseek/deepseek-v4.1-flash" \
    EMBEDDING_MODEL="qwen/qwen3-embedding-8b" \
    PORT=8000 \
    CHROMA_PATH="/app/chroma_db"

# Открываем порт
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Точка входа
WORKDIR /app/src
CMD ["python", "main.py"]
