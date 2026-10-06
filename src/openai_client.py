"""Обёртка Gemini API с retry для 503/429."""

import time
from openai import OpenAI
from config import (
    GEMINI_API_KEY, BASE_URL, CHAT_MODEL, EMBEDDING_MODEL,
    TEMPERATURE, MAX_TOKENS,
)

_client = OpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)


def get_embedding(text):
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def get_embeddings_batch(texts):
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def chat(messages, max_tokens=MAX_TOKENS, max_retries=3):
    """Retry с паузами: 30, 60, 90 сек при 503; 65 сек при 429."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = _client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            print(f"[LLM] finish_reason={choice.finish_reason}, "
                  f"completion_tokens={response.usage.completion_tokens}")
            return choice.message.content
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str:
                wait = 65
                print(f"[LLM] Rate limit (429). Ждём {wait} сек...")
                time.sleep(wait)
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                wait = 30 * attempt  # 30, 60, 90
                print(f"[LLM] Перегрузка (503). Ждём {wait} сек... "
                      f"(попытка {attempt}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise last_error
