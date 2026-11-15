"""``python -m varkcheck.web`` — start the local app.

    python -m varkcheck.web                 # http://127.0.0.1:8765
    python -m varkcheck.web --port 9000
    python -m varkcheck.web --provider ollama --model llama3.1

Provider and model come from the same environment variables as v1
(``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_BASE_URL``, ``LLM_API_KEY``);
the flags just override them for one run.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from ..config import Settings
from .server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="varkcheck.web", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    parser.add_argument("--provider", help="ollama | deepseek | openai")
    parser.add_argument("--model", help="model name for the provider")
    parser.add_argument("--base-url", help="override the provider's API base URL")
    parser.add_argument("--timeout", type=int, help="request timeout in seconds (default 60)")
    parser.add_argument("--open", action="store_true", help="open the browser once the server starts")
    args = parser.parse_args(argv)

    settings = Settings.load()
    if args.provider:
        from ..config import PROVIDER_PRESETS

        preset = PROVIDER_PRESETS.get(args.provider.lower())
        settings.provider = args.provider.lower()
        if preset:
            settings.base_url = preset["base_url"]
            settings.model = preset["model"]
    if args.base_url:
        settings.base_url = args.base_url
    if args.model:
        settings.model = args.model
    if args.timeout:
        settings.request_timeout = args.timeout

    if settings.provider == "mock":
        print("Provider 'mock' cannot write reports. Choose ollama, deepseek or openai.", file=sys.stderr)
        return 2
    if settings.provider in ("deepseek", "openai") and not settings.api_key:
        print(
            f"LLM_API_KEY is not set, so the {settings.provider} provider cannot be called.\n"
            "Either export LLM_API_KEY, or run a local model with --provider ollama.",
            file=sys.stderr,
        )
        return 2

    url = f"http://{args.host}:{args.port}"
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    serve(settings, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
