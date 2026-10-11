# RemStroy RAG Hybrid

Учебный RAG-ассистент для компании **RemStroy** — ремонт квартир и установка оконных/балконных блоков. Демонстрирует продвинутый retrieval с **гибридным поиском** (BM25 + embeddings) и объединением результатов через **Reciprocal Rank Fusion**.

## ✨ Возможности

- **Гибридный поиск**: параллельно работают два независимых механизма:
  - **Embedding-поиск** — семантическое сходство через Qwen `qwen3-embedding-8b`
  - **BM25** — поиск по ключевым словам без LLM (`rank_bm25`)
- **Извлечение ключевых слов** из вопроса: регулярки + фильтр стоп-слов + частотный анализ (без LLM)
- **Reciprocal Rank Fusion (RRF, k=60)** — корректное объединение результатов двух поисковиков
- **Чат-интерфейс** с блоком «Логика работы RAG»: ключевые слова, embedding-топ-5, BM25-топ-5, финальный набор после RRF
- **Многотуровая память** диалога
- **ChromaDB PersistentClient** — постоянное векторное хранилище
- **Retry-логика** при 429 (rate limit) и 503 (перегрузка) на стороне RouterAI
- **Docker + docker compose** — запуск одной командой, готов к деплою на VPS

## 🏗 Архитектура

    Пользователь
        │
        ▼
    [HTML/CSS/JS чат] ──► POST /api/query
                            │
                            ▼
                    [Python HTTP-сервер]
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      Ключевые слова   Embedding-поиск   BM25-поиск
      (без LLM)        (Qwen3 API)       (rank_bm25)
            │               │               │
            │               ▼               ▼
            │           ChromaDB        BM25-индекс
            │               │               │
            │               └───────┬───────┘
            │                       ▼
            │                RRF (k=60)
            │                       │
            └──────────► LLM (deepseek-v4.1-flash)
                                    │
                                    ▼
                            AI-ответ + все чанки

## 📁 Структура проекта

    RemStroy-RAG-Hybrid/
    ├── README.md
    ├── requirements.txt            # openai, chromadb, rank_bm25
    ├── .gitignore                  # содержит .env
    ├── .env.example                # шаблон переменных окружения
    ├── Dockerfile
    ├── docker-compose.yml          # volume chroma_data, ROUTERAI_API_KEY
    ├── src/
    │   ├── __init__.py
    │   ├── config.py               # BASE_URL RouterAI, модели, порт
    │   ├── openai_client.py        # обёртка RouterAI API + retry
    │   ├── chroma_store.py         # ChromaDB PersistentClient
    │   ├── rag_pipeline.py         # гибридный поиск + RRF + генерация
    │   └── main.py                 # HTTP-сервер, точка входа
    ├── knowledge_base/             # 7 демо-документов RemStroy (30 чанков)
    ├── static/                     # фронтенд (index.html, style.css, app.js)
    └── notebooks/                  # демонстрационный ноутбук

## 🚀 Запуск

### Через Docker (рекомендуется)

    git clone https://github.com/Alextgn500/RemStroy-RAG-Hybrid.git
    cd RemStroy-RAG-Hybrid

    cp .env.example .env
    # откройте .env и вставьте свой ROUTERAI_API_KEY

    docker compose up -d
    docker compose logs -f

Откройте http://localhost:8000.

При первом запуске индекс ChromaDB пуст. Нажмите «Переиндексировать» в интерфейсе или вызовите:

    curl -X POST http://localhost:8000/api/reindex

### Локально (без Docker)

    git clone https://github.com/Alextgn500/RemStroy-RAG-Hybrid.git
    cd RemStroy-RAG-Hybrid
    pip install -r requirements.txt

    export ROUTERAI_API_KEY="sk-..."

    cd src
    python main.py

Откройте http://localhost:8000.

## ⚙️ Технические решения

| Компонент | Выбор | Почему |
|---|---|---|
| **Провайдер** | [RouterAI](https://routerai.ru) | Единый OpenAI-совместимый API к DeepSeek и Qwen, работает без VPN |
| **Embedding** | `qwen/qwen3-embedding-8b` | Сильное качество на русскоязычных текстах |
| **LLM** | `deepseek/deepseek-v4.1-flash` | Быстрый, точный, отвечает ~10 сек |
| **Keyword search** | `rank_bm25` (BM25Okapi) | Классический алгоритм, без LLM |
| **Fusion** | RRF (k=60) | Не требует нормализации скоров |
| **Vector DB** | ChromaDB PersistentClient | Простота, локальное хранилище, volume между перезапусками |
| **HTTP** | `http.server` из stdlib | Без внешних фреймворков |
| **Контейнеризация** | Docker + docker compose | Один образ, готов к деплою на VPS |

## 🔬 Как работает RRF

BM25 и cosine similarity имеют **разные шкалы** — их нельзя складывать напрямую. RRF использует **ранги**:

    score(doc) = Σ 1 / (k + rank_i(doc))

где `k = 60`, `rank_i` — позиция документа в i-м ранжированном списке.

## 📝 Примеры запросов

- «Сколько стоит установка трёхслойного окна на балконе?»
- «Какие гарантии вы даёте на монтаж окон?»
- «Что такое тёплый монтаж и чем он отличается от стандартного?»

## 🐳 Полезные команды Docker

    # Запуск
    docker compose up -d
    docker compose logs -f

    # Статус
    docker compose ps

    # Остановка (данные ChromaDB сохраняются)
    docker compose stop

    # Полный сброс (включая volume chroma_data — индекс будет потерян)
    docker compose down -v

## ⚠️ Известные ограничения

- **Лимиты RouterAI** зависят от тарифа. При высокой частоте запросов возможны 429 — retry в `openai_client.py` ждёт и повторяет
- **Cold start**: первый запрос после запуска контейнера может занять дольше из-за инициализации ChromaDB
- **Переиндексация** занимает время пропорционально объёму базы знаний (30 чанков — несколько секунд)

## 🔮 Что дальше

Возможные направления развития проекта:

- **Реранкинг через cross-encoder** — переупорядочивание топ-чанков после гибридного поиска моделью `bge-reranker-v2-m3` для повышения точности
- **Метаданные-фильтрация** — фильтрация чанков по типу услуги (окна / ремонт), ценовому диапазону, дате
- **Кэширование частых запросов** — сохранение ответов на типовые вопросы в Redis или локально, чтобы не тратить токены RouterAI
- **Загрузка документов через веб-интерфейс** — админ-панель для добавления новых `.md` в базу знаний без правки кода
- **Мультиязычность** — поддержка английского и других языков для расширения аудитории
- **Экспорт диалогов** — выгрузка истории чата в PDF/CSV для отчётности
- **Развёртывание на VPS** — постоянный сервер (Timeweb Cloud, Beget, Selectel) с доменом и SSL

---

## 📄 Лицензия

Учебный проект. Демо-данные RemStroy вымышлены.
