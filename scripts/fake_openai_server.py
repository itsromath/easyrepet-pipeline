from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "qwen/qwen3-4b-2507", "object": "model"},
                        {"id": "qwen/qwen3.5-9b", "object": "model"},
                    ],
                },
            )
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        model = str(payload.get("model", "fake-model"))
        content = (
            "# Отчёт по занятию\n\n"
            "## 1. Краткое содержание\n\n"
            "Тестовый сервер подтвердил, что pipeline получил запрос и вернул Markdown.\n\n"
            "## 2. Карта занятия\n\n"
            "- Разбор темы.\n"
            "- Практика.\n"
            "- Домашнее задание.\n\n"
            "## 3. Что требует подтверждения педагога\n\n"
            "Это smoke-тест, не педагогический вывод.\n"
        )
        self._send_json(
            200,
            {
                "id": "chatcmpl-easyrepet-smoke",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 1234), FakeOpenAIHandler)
    print("Fake OpenAI-compatible server running on http://127.0.0.1:1234/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFake server stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
