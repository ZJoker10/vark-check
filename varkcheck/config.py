"""Configuration: one object built from the environment, used everywhere."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import env

ROOT = Path(__file__).resolve().parent.parent

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.1"},
    "mock": {"base_url": "mock://offline", "model": "mock-1"},
}


@dataclass
class Settings:
    # --- LLM ---
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_output_tokens: int = 1200
    request_timeout: int = 60

    # --- agent limits ---
    max_steps: int = 6
    token_budget: int = 25000

    # --- retrieval ---
    top_k: int = 4
    chunk_chars: int = 1200
    min_similarity: float = 0.02

    # --- paths ---
    root: Path = field(default_factory=lambda: ROOT)
    course_dir: Path = field(default_factory=lambda: ROOT / "data" / "course")
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    db_path: Path = field(default_factory=lambda: ROOT / "data" / "varkcheck.db")
    index_dir: Path = field(default_factory=lambda: ROOT / "data" / "index")
    log_dir: Path = field(default_factory=lambda: ROOT / "logs")
    prompt_path: Path = field(default_factory=lambda: ROOT / "prompts" / "system.md")

    # --- misc ---
    demo_seed: int = 7
    synthetic_students: int = 10

    @classmethod
    def load(cls, env_file: str | Path | None = None) -> "Settings":
        env.load_dotenv(env_file)

        provider = (env.get("LLM_PROVIDER", "deepseek") or "deepseek").lower()
        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["deepseek"])

        settings = cls(
            provider=provider,
            base_url=env.get("LLM_BASE_URL", preset["base_url"]) or preset["base_url"],
            api_key=env.get("LLM_API_KEY", "") or "",
            model=env.get("LLM_MODEL", preset["model"]) or preset["model"],
            temperature=env.get_float("LLM_TEMPERATURE", 0.2),
            max_output_tokens=env.get_int("LLM_MAX_OUTPUT_TOKENS", 1200),
            request_timeout=env.get_int("LLM_TIMEOUT", 60),
            max_steps=env.get_int("AGENT_MAX_STEPS", 6),
            token_budget=env.get_int("AGENT_TOKEN_BUDGET", 25000),
            top_k=env.get_int("RAG_TOP_K", 4),
            chunk_chars=env.get_int("RAG_CHUNK_CHARS", 1200),
            min_similarity=env.get_float("RAG_MIN_SIMILARITY", 0.02),
            db_path=ROOT / (env.get("DB_PATH", "data/varkcheck.db") or "data/varkcheck.db"),
            log_dir=ROOT / (env.get("LOG_DIR", "logs") or "logs"),
        )
        settings.ensure_dirs()
        return settings

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.index_dir, self.log_dir, self.course_dir):
            Path(path).mkdir(parents=True, exist_ok=True)

    @property
    def is_offline(self) -> bool:
        """True when no network call will be made (mock provider or ollama)."""
        return self.provider in ("mock", "ollama")

    def describe(self) -> dict[str, Any]:
        """Safe summary for the UI / trace log (never includes the API key)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_steps": self.max_steps,
            "token_budget": self.token_budget,
            "top_k": self.top_k,
        }
