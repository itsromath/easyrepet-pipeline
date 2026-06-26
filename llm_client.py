import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_TEST_PROMPT = "Ответь одним словом: OK"


class LLMConfigError(RuntimeError):
    """Ошибка в локальной конфигурации LLM."""


class LLMRequestError(RuntimeError):
    """Ошибка при обращении к локальной LLM."""


def load_model_preset(preset_name: str, *, base_dir: Path | None = None) -> Dict[str, Any]:
    root = base_dir or BASE_DIR
    config_path = root / "config" / "model_presets.json"

    if not config_path.exists():
        raise LLMConfigError(f"Model presets file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            presets = json.load(file)
    except json.JSONDecodeError as exc:
        raise LLMConfigError(f"Invalid JSON in model presets file: {config_path}") from exc

    if not isinstance(presets, dict):
        raise LLMConfigError(f"Model presets file must contain a JSON object: {config_path}")

    if preset_name not in presets:
        available = ", ".join(sorted(presets)) or "none"
        raise LLMConfigError(f"Unknown model preset: {preset_name}. Available presets: {available}")

    preset = dict(presets[preset_name])
    required_fields = ("model", "system_prompt_file", "temperature", "top_p")
    missing = [field for field in required_fields if field not in preset]
    if missing:
        raise LLMConfigError(
            f"Model preset {preset_name} is missing required field(s): {', '.join(missing)}"
        )

    prompt_path = root / str(preset["system_prompt_file"])
    if not prompt_path.exists():
        raise LLMConfigError(
            f"System prompt file not found for preset {preset_name}: {prompt_path}"
        )

    preset["system_prompt"] = prompt_path.read_text(encoding="utf-8").strip()
    return preset


class OpenAICompatibleClient:
    """
    Минимальный клиент для OpenAI-compatible API.

    Подходит для локальных серверов, которые поддерживают:
    POST /v1/chat/completions

    Примеры:
    - LM Studio Local Server
    - llama.cpp server с OpenAI-compatible endpoint
    - некоторые режимы Ollama / прокси-обёртки
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "local-key",
        timeout: int = 600,
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float | None = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if top_p is not None:
            payload["top_p"] = top_p

        if extra_payload:
            payload.update(extra_payload)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                )

                if response.status_code >= 400:
                    raise LLMRequestError(
                        f"LLM API вернул HTTP {response.status_code}: {response.text[:1000]}"
                    )

                data = response.json()

                try:
                    return data["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError, TypeError) as exc:
                    raise LLMRequestError(
                        "Неожиданный формат ответа LLM API: "
                        f"{json.dumps(data, ensure_ascii=False)[:1000]}"
                    ) from exc

            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    break

        raise LLMRequestError(f"Не удалось получить ответ от LLM: {last_error}")

    def chat_with_preset(
        self,
        preset_name: str,
        user_prompt: str,
        extra_messages: Sequence[Dict[str, str]] | None = None,
    ) -> str:
        preset = load_model_preset(preset_name)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": str(preset["system_prompt"])},
        ]

        if extra_messages:
            messages.extend(extra_messages)

        messages.append({"role": "user", "content": user_prompt})

        return self.chat(
            model=str(preset["model"]),
            messages=messages,
            temperature=float(preset["temperature"]),
            top_p=float(preset["top_p"]),
            max_tokens=int(preset.get("max_tokens") or 4096),
        )


def call_lmstudio_chat(
    preset_name: str,
    user_prompt: str,
    extra_messages: Sequence[Dict[str, str]] | None = None,
    timeout: int = 600,
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
    api_key: str = "local-key",
) -> str:
    client = OpenAICompatibleClient(base_url=base_url, api_key=api_key, timeout=timeout)
    return client.chat_with_preset(
        preset_name=preset_name,
        user_prompt=user_prompt,
        extra_messages=extra_messages,
    )


def test_model_presets(
    preset_names: Sequence[str] = ("draft_4b", "final_9b"),
    *,
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
    api_key: str = "local-key",
    timeout: int = 600,
) -> bool:
    client = OpenAICompatibleClient(base_url=base_url, api_key=api_key, timeout=timeout)
    all_ok = True

    for preset_name in preset_names:
        try:
            client.chat_with_preset(preset_name, DEFAULT_TEST_PROMPT)
        except Exception as exc:
            all_ok = False
            print(f"Failed to call preset {preset_name}.")
            print("Check LM Studio server, model id, JIT loading, VRAM, and config/model_presets.json.")
            print(f"Error: {exc}")
        else:
            print(f"{preset_name} preset: OK")

    return all_ok
