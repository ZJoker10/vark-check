"""The local web app: upload material, take the quiz once, get the report.

Why a server instead of a static page
-------------------------------------
A ``file://`` page cannot keep an API key off the client, cannot assemble a PDF,
and cannot call a local model without CORS games. A small stdlib server on
``127.0.0.1`` solves all three and adds no dependency.

What the server knows about you
-------------------------------
Nothing that survives a restart. Uploads live in an in-memory session, keyed by
a random token, discarded after ``SESSION_TTL`` seconds. The VARK profile is
never sent to be stored — it arrives with each request and the browser keeps it
in ``localStorage``. There is no account, no cookie and no log of who asked.

Endpoints
---------
``GET  /``              the app
``GET  /static/<name>`` css, js, vendored Mermaid
``GET  /api/health``    which model is configured (no secrets)
``GET  /api/quiz``      the eight self-report items
``POST /api/extract``   multipart upload -> a session token and what was read
``POST /api/score``     questionnaire answers -> a profile
``POST /api/report``    material + profile + style -> report JSON and preview HTML
``POST /api/export``    the same, plus client-rendered diagrams -> a file
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .. import profile as profile_mod, questionnaire, render, summarize, uploads
from ..config import Settings
from ..llm import ChatClient, LLMError

STATIC = Path(__file__).resolve().parent / "static"
SESSION_TTL = 2 * 60 * 60      # seconds; a tab left open all afternoon is fine
MAX_SESSIONS = 8
MAX_BODY = 220 * 1024 * 1024   # multipart overhead on top of the per-file caps

ALLOWED_STATIC = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "mermaid.min.js": "text/javascript; charset=utf-8",
}

FORMATS = {"pdf", "docx", "md", "html"}
STYLES = {"V", "A", "R", "K"}


# --------------------------------------------------------------------------- #
# Session storage: in memory, expiring, never written to disk
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    token: str
    document: uploads.Document
    workdir: Path
    created: float = field(default_factory=time.monotonic)
    reports: dict[str, summarize.Report] = field(default_factory=dict)

    def key(self, style: str, profile: profile_mod.Profile) -> str:
        digest = hashlib.sha256(
            json.dumps(profile.scores, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        return f"{style.upper()}:{digest}"


class SessionStore:
    """Bounded, expiring map. Deliberately forgetful."""

    def __init__(self, ttl: int = SESSION_TTL, limit: int = MAX_SESSIONS):
        self.ttl = ttl
        self.limit = limit
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, document: uploads.Document, workdir: Path) -> Session:
        token = secrets.token_urlsafe(18)
        session = Session(token=token, document=document, workdir=workdir)
        with self._lock:
            self._evict()
            while len(self._sessions) >= self.limit:
                oldest = min(self._sessions.values(), key=lambda item: item.created)
                self._discard(oldest.token)
            self._sessions[token] = session
        return session

    def get(self, token: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            if time.monotonic() - session.created > self.ttl:
                self._discard(token)
                return None
            return session

    def _evict(self) -> None:
        now = time.monotonic()
        for token in [key for key, session in self._sessions.items() if now - session.created > self.ttl]:
            self._discard(token)

    def _discard(self, token: str) -> None:
        session = self._sessions.pop(token, None)
        if session is not None:
            _remove_tree(session.workdir)


def _remove_tree(path: Path) -> None:
    try:
        for child in sorted(path.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
        path.rmdir()
    except OSError:
        pass  # a locked temp file is not worth crashing an export over


# --------------------------------------------------------------------------- #
# Multipart parsing, stdlib only
# --------------------------------------------------------------------------- #
def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    """Return ``(fields, files)`` from a multipart/form-data body.

    ``email`` is used rather than the removed ``cgi`` module; it handles binary
    parts correctly as long as the header block is reconstructed.
    """
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8", "replace") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition") or ""
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files.append((Path(filename).name, payload))
        elif name:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, files


# --------------------------------------------------------------------------- #
# The application
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sessions = SessionStore()
        self.root = Path(settings.data_dir) / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._llm: ChatClient | None = None
        self._llm_lock = threading.Lock()

    @property
    def llm(self) -> ChatClient:
        with self._llm_lock:
            if self._llm is None:
                self._llm = ChatClient(self.settings)
            return self._llm

    # -- handlers ---------------------------------------------------------- #
    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "provider": self.settings.provider,
            "model": self.settings.model,
            "offline": self.settings.is_offline,
            "styles": sorted(STYLES),
            "formats": sorted(FORMATS),
        }

    def quiz(self) -> dict[str, Any]:
        """The questionnaire itself. Scoring stays on the server."""
        return {
            "questions": [
                {"id": question.id, "dimension": question.dimension, "text": question.text}
                for question in questionnaire.QUESTIONS
            ],
            "scale": {"min": 1, "max": 5},
        }

    def extract(self, fields: dict[str, str], files: list[tuple[str, bytes]]) -> dict[str, Any]:
        pasted = (fields.get("notes") or "").strip()
        if pasted:
            files = files + [("pasted-notes.md", pasted.encode("utf-8"))]
        if not files:
            raise ValueError("No files and no pasted text — there is nothing to summarise.")

        token = secrets.token_urlsafe(8)
        workdir = self.root / token
        document = uploads.extract(files, workdir, title=(fields.get("title") or "").strip() or None)
        session = self.sessions.create(document, workdir)

        return {
            "upload_id": session.token,
            "title": document.title,
            "files": document.files,
            "chars": document.chars,
            "sections": len(document.sections),
            "outline": document.outline()[:24],
            "figures": len(document.figures),
            "warnings": document.warnings,
            "usable": document.has_text,
            "char_limit": uploads.MAX_TOTAL_CHARS,
        }

    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        built = profile_mod.score(payload.get("responses"))
        return {"profile": built.to_dict(), "description": profile_mod.describe(built)}

    def report(self, payload: dict[str, Any]) -> dict[str, Any]:
        session, built, style = self._context(payload)
        cached = session.reports.get(session.key(style, built))
        if cached is None:
            cached = summarize.generate(session.document, built, style, llm=self.llm)
            session.reports[session.key(style, built)] = cached
        return {
            "report": cached.to_dict(),
            "html": render.to_html(cached, built, document=session.document, preview=True),
            "profile": built.to_dict(),
        }

    def export(self, payload: dict[str, Any]) -> tuple[bytes, str, str]:
        session, built, style = self._context(payload)
        fmt = str(payload.get("format") or "pdf").lower()
        if fmt not in FORMATS:
            raise ValueError(f"Unsupported format {fmt!r}. Choose one of {', '.join(sorted(FORMATS))}.")
        report = session.reports.get(session.key(style, built))
        if report is None:
            report = summarize.generate(session.document, built, style, llm=self.llm)
            session.reports[session.key(style, built)] = report
        svgs = _clean_svgs(payload.get("svgs"))
        return render.render(report, built, fmt, document=session.document, svgs=svgs)

    # -- helpers ----------------------------------------------------------- #
    def _context(self, payload: dict[str, Any]) -> tuple[Session, profile_mod.Profile, str]:
        session = self.sessions.get(str(payload.get("upload_id") or ""))
        if session is None:
            raise ValueError("That upload has expired. Upload the material again.")
        if not session.document.has_text:
            raise ValueError("That upload has no usable text to summarise.")
        built = profile_mod.Profile.from_dict(payload.get("profile") or {})
        style = str(payload.get("style") or built.dominant).upper()
        if style not in STYLES:
            raise ValueError(f"Unknown learning style {style!r}.")
        return session, built, style


def _clean_svgs(raw: Any) -> dict[int, str]:
    """Accept only small, genuine SVG documents from the browser."""
    out: dict[int, str] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        if not isinstance(value, str) or len(value) > 2_000_000:
            continue
        text = value.strip()
        if not text.startswith("<svg") or "<script" in text.lower():
            continue
        try:
            out[int(key)] = text
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    app: App
    server_version = "VARKCheck/2.0"

    # -- routing ----------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/") :])
        if path == "/api/health":
            return self._json(HTTPStatus.OK, self.app.health())
        if path == "/api/quiz":
            return self._json(HTTPStatus.OK, self.app.quiz())
        if path == "/favicon.ico":
            return self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/extract":
                return self._extract()
            if path == "/api/score":
                return self._json(HTTPStatus.OK, self.app.score(self._read_json()))
            if path == "/api/report":
                return self._report()
            if path == "/api/export":
                return self._export()
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except LLMError as exc:
            self._json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the page
            traceback.print_exc()
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

    # -- request bodies ---------------------------------------------------- #
    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_BODY:
            raise ValueError("That upload is too large for this app.")
        return self.rfile.read(length)

    def _read_json(self) -> dict[str, Any]:
        raw = self._body()
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Expected a JSON body.") from None
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    # -- endpoints --------------------------------------------------------- #
    def _extract(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Upload must be sent as multipart/form-data.")
        fields, files = parse_multipart(content_type, self._body())
        self._json(HTTPStatus.OK, self.app.extract(fields, files))

    def _report(self) -> None:
        payload = self._read_json()
        self._json(HTTPStatus.OK, self.app.report(payload))

    def _export(self) -> None:
        payload = self._read_json()
        body, mimetype, filename = self.app.export(payload)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- responses --------------------------------------------------------- #
    def _static(self, name: str) -> None:
        # Whitelist rather than path arithmetic: no traversal to reason about.
        if name not in ALLOWED_STATIC:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        path = STATIC / name
        if not path.is_file():
            return self._json(HTTPStatus.NOT_FOUND, {"error": f"{name} is missing from the installation."})
        self._send(HTTPStatus.OK, path.read_bytes(), ALLOWED_STATIC[name], cache="no-store")

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str, *, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; frame-src 'self'; connect-src 'self'; base-uri 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Deliberately quiet: no IP addresses in a tool that promises anonymity.
        if "/api/" in (args[0] if args else "") and str(args[1] if len(args) > 1 else "").startswith("5"):
            print(f"[vark] {fmt % args}")


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> None:
    handler = type("BoundHandler", (Handler,), {"app": App(settings)})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    print(f"VARK Check v2 — http://{host}:{port}")
    print(f"  provider: {settings.provider}  model: {settings.model}")
    print("  anonymous: nothing is stored beyond this process; Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()
