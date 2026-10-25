"""HTTP-сервер RemStroy RAG Hybrid.

Запуск:
    cd src && python main.py
или (для продакшена):
    gunicorn -b 0.0.0.0:8000 main:app
"""

import json
import os
import sys
import http.server
import socketserver
import urllib.parse  # <-- ДОБАВЛЕНО: для декодирования URL

# Разрешаем запуск как `python src/main.py` из корня проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HOST, PORT, STATIC_DIR
from rag_pipeline import rag_query, build_index, reset_history
from chroma_store import count


class Handler(http.server.BaseHTTPRequestHandler):

    # ---------- helpers ----------
    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _serve_static(self, rel_path):
        # <-- ИЗМЕНЕНО: Убираем query-параметры (например, ?v=1) и декодируем URL
        # Это важно, чтобы /widget/widget.html?foo=bar корректно находил файл
        parsed_path = urllib.parse.urlparse(rel_path).path
        safe_rel = parsed_path.lstrip("/").replace("..", "")

        full = os.path.join(STATIC_DIR, safe_rel)
        if not os.path.isfile(full):
            self.send_error(404, "Not found")
            return
        ext = os.path.splitext(full)[1].lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
        }.get(ext, "application/octet-stream")
        with open(full, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---------- routing ----------
    def do_GET(self):
        # <-- ИЗМЕНЕНО: Более гибкая логика роутинга
        # Отрезаем query-параметры для проверки пути
        path = urllib.parse.urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif path == "/demo":
            self._serve_static("demo.html")
        elif path == "/widget":
            # <-- ДОБАВЛЕНО: Отдаем страницу виджета
            self._serve_static("widget/widget.html")
        elif path == "/api/status":
            self._send_json({"indexed_chunks": count()})
        elif path == "/health":
            self._send_json({"status": "ok"})
        else:
            # <-- ИЗМЕНЕНО: Все остальные пути пытаемся отдать как статику
            # (это нужно для widget_style.css и widget_app.js)
            self._serve_static(self.path)

    def do_POST(self):
        # <-- ИЗМЕНЕНО: Игнорируем query-параметры при проверке пути
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/query":
            data = self._read_json()
            question = (data.get("question") or "").strip()
            if not question:
                self._send_json({"error": "Пустой вопрос"}, status=400)
                return
            try:
                result = rag_query(question)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/reindex":
            try:
                n = build_index()
                reset_history()
                self._send_json({"indexed_chunks": n})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/reset":
            reset_history()
            self._send_json({"ok": True})

        else:
            self.send_error(404, "Not found")

    def log_message(self, fmt, *args):
        return


def run():
    with socketserver.ThreadingTCPServer((HOST, PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"[RemStroy] Сервер запущен: http://{HOST}:{PORT}")
        print(f"[RemStroy] Static директория: {STATIC_DIR}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[RemStroy] Остановлено.")


if __name__ == "__main__":
    run()
