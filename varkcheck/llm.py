"""Provider-agnostic chat client (DeepSeek / OpenAI-compatible / Ollama).

Why not the OpenAI SDK? Because the same code then works against DeepSeek, a
university gateway, vLLM, or a laptop running Ollama, with no dependency and no
version pinning. The wire format is the only contract we need.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import Settings


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns unusable output."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens)

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total,
        }


@dataclass
class LLMReply:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0
    cached: bool = False


class ChatClient:
    """Blocking chat completions with a small retry policy and usage reporting."""

    def __init__(self, settings: Settings, trace: Any | None = None):
        self.settings = settings
        self.trace = trace
        self.usage = Usage()

    # -- public ------------------------------------------------------------ #
    def chat(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMReply:
        if self.settings.provider == "mock":
            raise LLMError("Provider 'mock' is offline-only; use MockLLM in tests.")

        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_output_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if json_mode:
            # Ollama, DeepSeek and OpenAI all honour this on the /v1 surface;
            # without it small local models happily answer in prose.
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        data = self._post(payload)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        try:
            text = data["choices"][0]["message"]["content"] or ""
            finish = data["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from exc

        if not text.strip():
            text, finish, data = self._recover_empty(payload, text, finish, data, json_mode)

        if not text.strip():
            raise LLMError(
                f"The model returned an empty reply (finish_reason={finish!r}). "
                "Try a larger max output token budget or a different model."
            )

        raw_usage = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
        )
        self.usage = self.usage + usage

        if self.trace is not None:
            self.trace.log(
                "llm.call",
                model=self.settings.model,
                latency_ms=latency_ms,
                usage=usage.to_dict(),
                json_mode=json_mode,
            )
        return LLMReply(text=text, usage=usage, model=self.settings.model, latency_ms=latency_ms)

    # -- transport --------------------------------------------------------- #
    def _recover_empty(
        self,
        payload: dict[str, Any],
        text: str,
        finish: Any,
        data: dict[str, Any],
        json_mode: bool,
    ) -> tuple[str, Any, dict[str, Any]]:
        """Two cheap recoveries for a reply that came back blank.

        Small local models blank out for two different reasons, and they are
        worth telling apart:

        * ``finish_reason == "length"`` — the model spent the whole budget
          thinking (``qwen3``, ``gemma`` with a long prompt) and never emitted
          content. Raising the ceiling is the fix.
        * anything else, with ``json_mode`` on — some models answer a
          ``json_object`` request with nothing at all. Asking again without the
          flag works, and costs one call.
        """
        attempts: list[dict[str, Any]] = []
        if finish == "length":
            ceiling = min(16_000, int(payload.get("max_tokens") or 1000) * 2)
            if ceiling > int(payload.get("max_tokens") or 0):
                attempts.append({**payload, "max_tokens": ceiling})
        if json_mode:
            without = {key: value for key, value in payload.items() if key != "response_format"}
            attempts.append(without)
            if finish == "length":
                ceiling = min(16_000, int(payload.get("max_tokens") or 1000) * 2)
                attempts.append({**without, "max_tokens": ceiling})

        for candidate in attempts:
            try:
                data = self._post(candidate)
                text = data["choices"][0]["message"]["content"] or ""
                finish = data["choices"][0].get("finish_reason")
            except (KeyError, IndexError, TypeError, LLMError):
                continue
            if text.strip():
                return text, finish, data
        return text, finish, data

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.settings.request_timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # noqa: PERF203 - deliberate retry loop
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                last_error = LLMError(f"HTTP {exc.code} from {url}: {detail}")
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = LLMError(f"Cannot reach {url}: {exc}")
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last_error from exc
        raise last_error or LLMError("unknown transport failure")


class MockLLM:
    """Deterministic stand-in used by the test suite (and by --offline demos).

    It plays back a script: each call returns the next item. Dicts are JSON
    encoded, which is exactly what the orchestrator expects from a real model.
    """

    def __init__(self, script: list[Any] | None = None, default: Any = None):
        self.script = list(script or [])
        self.default = default if default is not None else {
            "action": "final",
            "message": "(mock) nothing left in the script",
        }
        self.calls: list[list[dict[str, str]]] = []
        self.usage = Usage()

    def chat(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> LLMReply:
        self.calls.append(list(messages))
        item = self.script.pop(0) if self.script else self.default
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMReply(text=text, usage=Usage(0, 0), model="mock")


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in prose and code fences often enough that every
    orchestrator needs this.
    """
    if not text or not text.strip():
        raise LLMError("empty model response")

    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        # ```json\n{...}\n```  ->  parts[0] == "", parts[1] == "json\n{...}\n"
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                stripped = candidate
                break

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMError(f"No JSON object in model response: {text[:200]!r}")
