"""Minimal .env loader.

Ninety-nine percent of what `python-dotenv` does, in forty lines and with no
dependency. Real environment variables always win over values in the file, so
`LLM_PROVIDER=openai streamlit run app.py` overrides .env for a single run.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ (non-destructive)."""
    env_path = Path(path) if path else ROOT / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def get_int(name: str, default: int) -> int:
    try:
        return int(str(get(name, str(default))))
    except (TypeError, ValueError):
        return default


def get_float(name: str, default: float) -> float:
    try:
        return float(str(get(name, str(default))))
    except (TypeError, ValueError):
        return default


def get_bool(name: str, default: bool = False) -> bool:
    value = get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
