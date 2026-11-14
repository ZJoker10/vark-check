"""Retrieval over the real course material in data/course/.

Design notes (also summarised in docs/architecture.md):

* Chunking is heading-aware. A citation is therefore always "file + section"
  (plus a page number for PDFs), which is what the brief requires and what the
  output guardrail can verify.
* The default vector space is a hashing-free, corpus-fitted TF-IDF with cosine
  similarity — implemented in pure Python so the prototype has no embedding
  download, no GPU and no network dependency at demo time. It is a real vector
  space: vectors are L2-normalised and scored by cosine, and the store is a
  plain matrix persisted to JSON.
* If `sentence-transformers` is installed and VARK_EMBEDDER=sentence-transformers,
  a dense MiniLM space is used instead, with the same interface and the same
  persisted format. Nothing else in the codebase changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "also", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "one", "two",
    "using", "used", "use", "may", "much", "many", "make", "made", "like",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Text processing
# --------------------------------------------------------------------------- #
def _stem(token: str) -> str:
    if len(token) > 4:
        for suffix in ("ies", "ing", "ers", "er", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [
        _stem(token)
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    section: str
    page: int | None
    text: str

    @property
    def citation(self) -> str:
        if self.page is not None:
            return f"{self.doc} p.{self.page} §{self.section}"
        return f"{self.doc} §{self.section}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            doc=data["doc"],
            section=data["section"],
            page=data.get("page"),
            text=data["text"],
        )


@dataclass
class Retrieved:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"citation": self.chunk.citation, "score": round(self.score, 4), "text": self.chunk.text}


# --------------------------------------------------------------------------- #
# Loading and chunking
# --------------------------------------------------------------------------- #
def _paragraphs_from_markdown(text: str) -> list[tuple[str, int | None, str]]:
    section = "Preamble"
    paragraph_lines: list[str] = []
    blocks: list[tuple[str, int | None, str]] = []

    def flush() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            blocks.append((section, None, "\n".join(paragraph_lines).strip()))
            paragraph_lines = []

    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            section = heading.group(2).strip()
            continue
        if not line.strip():
            flush()
            continue
        paragraph_lines.append(line.rstrip())
    flush()
    return [block for block in blocks if block[2]]


def _paragraphs_from_pdf(data: bytes, doc: str) -> list[tuple[str, int | None, str]]:
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"Cannot read '{doc}': install pypdf (pip install pypdf) or convert the PDF to Markdown."
        ) from exc

    import io

    reader = PdfReader(io.BytesIO(data))
    blocks: list[tuple[str, int | None, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            for paragraph in re.split(r"\n\s*\n", page_text):
                if paragraph.strip():
                    blocks.append((f"page {page_number}", page_number, paragraph.strip()))
    return blocks


def _split_block(text: str, max_chars: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_RE.split(text):
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _pack(blocks: Iterable[tuple[str, int | None, str]], max_chars: int, overlap: int) -> list[tuple[str, int | None, str]]:
    """Greedily pack paragraphs into chunks of at most max_chars, with overlap."""
    chunks: list[tuple[str, int | None, str]] = []
    for section, page, text in blocks:
        for piece in _split_block(text, max_chars):
            if chunks and chunks[-1][0] == section and chunks[-1][1] == page:
                previous = chunks[-1][2]
                if len(previous) + len(piece) + 2 <= max_chars:
                    chunks[-1] = (section, page, f"{previous}\n\n{piece}")
                    continue
                tail = previous[-overlap:] if overlap else ""
                separator = " " if tail else ""
                chunks.append((section, page, f"{tail}{separator}{piece}".strip()))
                continue
            chunks.append((section, page, piece))
    return chunks


def load_chunks(course_dir: str | Path, chunk_chars: int = 1200, overlap: int = 180) -> list[Chunk]:
    course_path = Path(course_dir)
    if not course_path.exists():
        return []

    chunks: list[Chunk] = []
    for path in sorted(course_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(course_path).as_posix()
        if path.suffix.lower() == ".pdf":
            blocks = _paragraphs_from_pdf(path.read_bytes(), relative)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            blocks = _paragraphs_from_markdown(text)
            if path.suffix.lower() == ".txt":
                blocks = [(path.stem, None, text.strip())] if text.strip() else []

        for index, (section, page, body) in enumerate(_pack(blocks, chunk_chars, overlap)):
            digest = hashlib.sha256(f"{relative}|{section}|{page}|{body}".encode()).hexdigest()[:12]
            chunks.append(
                Chunk(
                    id=f"{Path(relative).stem}:{index}:{digest}",
                    doc=relative,
                    section=section,
                    page=page,
                    text=body,
                )
            )
    return chunks


def material_hash(course_dir: str | Path) -> str:
    """Fingerprint of the course folder, used to decide when to rebuild the index."""
    digest = hashlib.sha256()
    for path in sorted(Path(course_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            digest.update(path.relative_to(course_dir).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Vector spaces
# --------------------------------------------------------------------------- #
class TfidfSpace:
    """Corpus-fitted TF-IDF with cosine similarity. Pure Python, no downloads."""

    kind = "tfidf"

    def __init__(self, idf: dict[str, float] | None = None):
        self.idf: dict[str, float] = idf or {}

    def fit(self, tokenised_documents: list[list[str]]) -> None:
        total = max(1, len(tokenised_documents))
        document_frequency: dict[str, int] = {}
        for tokens in tokenised_documents:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log((1 + total) / (1 + freq)) + 1.0
            for token, freq in document_frequency.items()
        }

    def embed(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        if not tokens:
            return {}
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        length = len(tokens)
        # Unknown query terms get the maximum-ish weight so that rare words from
        # a student's question still match the course vocabulary.
        default_idf = max(self.idf.values(), default=1.0)
        vector = {
            token: (count / length) * self.idf.get(token, default_idf)
            for token, count in counts.items()
        }
        return _l2_normalize_sparse(vector)

    @staticmethod
    def similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(token, 0.0) for token, weight in left.items())


class DenseSpace:
    """Optional sentence-transformers space (same interface, same file format)."""

    kind = "dense"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)

    def fit(self, tokenised_documents: list[list[str]]) -> None:  # pragma: no cover
        return None

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]

    @staticmethod
    def similarity(left: list[float], right: list[float]) -> float:
        return float(sum(a * b for a, b in zip(left, right)))


def _l2_normalize_sparse(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    if norm == 0:
        return {}
    return {token: weight / norm for token, weight in vector.items()}


def make_space(kind: str | None = None, idf: dict[str, float] | None = None):
    kind = (kind or "tfidf").lower()
    if kind in ("sentence-transformers", "st", "dense"):
        try:
            return DenseSpace()
        except Exception:  # noqa: BLE001 - fall back, never crash the demo
            return TfidfSpace(idf)
    return TfidfSpace(idf)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
@dataclass
class Index:
    chunks: list[Chunk]
    vectors: list[Any]
    space: Any
    material_hash: str = ""
    chunk_chars: int = 1200

    # -- querying ---------------------------------------------------------- #
    def search(self, query: str, k: int = 4, min_score: float = 0.02) -> list[Retrieved]:
        if not self.chunks:
            return []
        query_vector = self.space.embed(query)
        query_terms = set(tokenize(query))
        scored: list[Retrieved] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = self.space.similarity(query_vector, vector)
            # A section heading is the strongest human signal in lecture notes:
            # "2.3 The learning rate..." should beat a passing mention in a
            # debugging checklist. Small, bounded bonus, never decisive alone.
            title_terms = set(tokenize(chunk.section))
            if title_terms and query_terms:
                score += 0.25 * (len(title_terms & query_terms) / len(title_terms))
            if score >= min_score:
                scored.append(Retrieved(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def sections(self, min_chars: int = 120) -> list[dict[str, Any]]:
        """Distinct (doc, section) pairs — the topic menu the student picks from."""
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for chunk in self.chunks:
            key = (chunk.doc, chunk.section)
            entry = seen.setdefault(
                key,
                {"doc": chunk.doc, "section": chunk.section, "chars": 0, "citation": chunk.citation},
            )
            entry["chars"] += len(chunk.text)
        topics = [
            entry
            for entry in seen.values()
            if entry["chars"] >= min_chars and entry["section"] != "Preamble"
        ]
        topics.sort(key=lambda entry: (entry["doc"], entry["section"]))
        return topics

    def sources_digest(self) -> list[dict[str, Any]]:
        return [
            {"doc": doc, "sections": sections}
            for doc, sections in _group_sections(self.chunks).items()
        ]

    # -- persistence -------------------------------------------------------- #
    def save(self, index_dir: str | Path) -> Path:
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "kind": self.space.kind,
            "idf": getattr(self.space, "idf", {}),
            "material_hash": self.material_hash,
            "chunk_chars": self.chunk_chars,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "vectors": [
                {term: round(weight, 5) for term, weight in vector.items()}
                if isinstance(vector, dict)
                else [round(value, 5) for value in vector]
                for vector in self.vectors
            ],
        }
        target = index_path / "index.json"
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, index_dir: str | Path) -> "Index":
        payload = json.loads((Path(index_dir) / "index.json").read_text(encoding="utf-8"))
        space = make_space(payload.get("kind"), payload.get("idf"))
        return cls(
            chunks=[Chunk.from_dict(item) for item in payload["chunks"]],
            vectors=payload["vectors"],
            space=space,
            material_hash=payload.get("material_hash", ""),
            chunk_chars=payload.get("chunk_chars", 1200),
        )

    @classmethod
    def build(cls, course_dir: str | Path, chunk_chars: int = 1200, space_kind: str = "tfidf") -> "Index":
        chunks = load_chunks(course_dir, chunk_chars=chunk_chars)
        space = make_space(space_kind)
        space.fit([tokenize(chunk.text) for chunk in chunks])
        vectors = [space.embed(chunk.text) for chunk in chunks]
        return cls(
            chunks=chunks,
            vectors=vectors,
            space=space,
            material_hash=material_hash(course_dir),
            chunk_chars=chunk_chars,
        )


def _group_sections(chunks: list[Chunk]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        sections = grouped.setdefault(chunk.doc, [])
        if chunk.section not in sections:
            sections.append(chunk.section)
    return grouped


def load_or_build(
    course_dir: str | Path,
    index_dir: str | Path,
    chunk_chars: int = 1200,
    force: bool = False,
    space_kind: str = "tfidf",
) -> tuple[Index, bool]:
    """Load the persisted index, rebuilding it when the course folder changed."""
    index_file = Path(index_dir) / "index.json"
    current_hash = material_hash(course_dir)
    if not force and index_file.exists():
        try:
            index = Index.load(index_dir)
            if index.material_hash == current_hash and index.chunks:
                return index, False
        except (OSError, ValueError, KeyError):
            pass  # corrupt index -> rebuild below
    index = Index.build(course_dir, chunk_chars=chunk_chars, space_kind=space_kind)
    index.save(index_dir)
    return index, True


def format_passages(results: list[Retrieved]) -> str:
    """Render retrieved passages for inclusion in an LLM prompt."""
    if not results:
        return "NO COURSE MATERIAL FOUND."
    blocks = []
    for result in results:
        blocks.append(f"[source: {result.chunk.citation}]\n{result.chunk.text}")
    return "\n\n---\n\n".join(blocks)
