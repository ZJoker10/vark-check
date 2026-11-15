"""Extract text and figures from whatever the user uploads.

The v2 flow takes arbitrary material, not a fixed course folder, so this module
is the boundary between "a file someone dragged in" and "structured sections the
summariser can reason about".

Design rules:

* Every section carries a provenance string (``file`` or ``file p.N``) so a
  citation in the report can be checked against something that really exists.
* Extraction never raises on a bad file. It returns warnings, because one
  unreadable PDF should not kill a six-file upload.
* Nothing here calls a model. Parsing is local, deterministic and free.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import rag

# --------------------------------------------------------------------------- #
# Limits. A local app should still refuse to eat all the RAM.
# --------------------------------------------------------------------------- #
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
DOC_SUFFIXES = {".pdf", ".docx"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | DOC_SUFFIXES | IMAGE_SUFFIXES

MAX_FILE_BYTES = 25 * 1024 * 1024   # 25 MB per file
MAX_FILES = 20
MAX_TOTAL_CHARS = 600_000           # hard cap on extracted text
MIN_USEFUL_CHARS = 200              # below this, there is nothing to summarise


@dataclass
class Section:
    """A heading-delimited slice of source material."""

    heading: str
    text: str
    source: str          # "notes.pdf p.3" or "lecture.md §2.1"
    order: int = 0
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "text": self.text,
            "source": self.source,
            "order": self.order,
            "page": self.page,
        }


@dataclass
class Figure:
    """An image found in the upload: embedded in DOCX, or uploaded directly."""

    name: str
    path: str            # absolute path on disk
    origin: str          # where it came from, for the caption
    caption: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "origin": self.origin, "caption": self.caption}


@dataclass
class Document:
    """Everything extracted from one upload batch."""

    title: str = "Uploaded material"
    sections: list[Section] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(section.text) for section in self.sections)

    @property
    def has_text(self) -> bool:
        return self.chars >= MIN_USEFUL_CHARS

    def outline(self) -> list[str]:
        """Heading map for the whole document, used to keep the summary global.

        Markdown parsing emits one section per paragraph, so the same heading
        repeats; the outline is de-duplicated to stay readable.
        """
        seen: list[str] = []
        for section in self.sections:
            if section.heading not in seen:
                seen.append(section.heading)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "files": self.files,
            "chars": self.chars,
            "sections": [section.to_dict() for section in self.sections],
            "figures": [figure.to_dict() for figure in self.figures],
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Per-format extraction
# --------------------------------------------------------------------------- #
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _clean(text: str) -> str:
    """Collapse the ragged whitespace that PDF extraction leaves behind."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _from_markdown(name: str, text: str) -> list[Section]:
    """Heading-aware split, reusing the same parser as the course index."""
    fallback = _clean_stem(Path(name).stem).title() or name
    sections: list[Section] = []
    for order, (heading, _page, body) in enumerate(rag._paragraphs_from_markdown(text)):
        if not body.strip():
            continue
        if not heading or heading == "Preamble":
            # Untitled prose still needs a name, and "Preamble" is not one.
            heading = fallback
        sections.append(
            Section(
                heading=heading,
                text=body,
                source=f"{name} §{heading}",
                order=order,
            )
        )
    if not sections and text.strip():
        sections.append(Section(heading=name, text=_clean(text), source=name, order=0))
    return sections


def _from_text(name: str, text: str) -> list[Section]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return [
        Section(heading=f"{name} — part {index + 1}", text=part, source=f"{name} ¶{index + 1}", order=index)
        for index, part in enumerate(paragraphs)
    ]


def _from_pdf(name: str, data: bytes, warnings: list[str]) -> list[Section]:
    try:
        from pypdf import PdfReader
    except ImportError:
        warnings.append(f"{name}: pypdf is not installed, PDF skipped (pip install pypdf).")
        return []

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - a corrupt PDF is a warning, not a crash
        warnings.append(f"{name}: could not be opened ({type(exc).__name__}).")
        return []

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            warnings.append(f"{name}: encrypted PDF, text not extracted.")
            return []

    sections: list[Section] = []
    empty_pages: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            empty_pages.append(page_number)
            continue
        text = _clean(raw)
        if len(text) < 20:
            empty_pages.append(page_number)
            continue
        heading = _first_heading(text) or f"{name} — page {page_number}"
        sections.append(
            Section(
                heading=heading,
                text=text,
                source=f"{name} p.{page_number}",
                order=len(sections),
                page=page_number,
            )
        )

    if empty_pages and len(empty_pages) == len(reader.pages):
        warnings.append(
            f"{name}: no extractable text on any page — likely a scanned PDF. "
            "OCR is not part of this build."
        )
    elif empty_pages:
        warnings.append(
            f"{name}: {len(empty_pages)} page(s) had no text (scanned/figures): "
            + ", ".join(str(p) for p in empty_pages[:6])
        )
    return sections


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if HEADING.match(line):
            return HEADING.match(line).group(2)
        if len(line) < 80 and not line.endswith("."):
            return line
        return None
    return None


def _from_docx(name: str, data: bytes, warnings: list[str], workdir: Path) -> tuple[list[Section], list[Figure]]:
    try:
        import docx  # python-docx
    except ImportError:
        warnings.append(f"{name}: python-docx is not installed, DOCX skipped (pip install python-docx).")
        return [], []

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{name}: could not be opened ({type(exc).__name__}).")
        return [], []

    sections: list[Section] = []
    heading = name
    buffer: list[str] = []
    order = 0

    def flush() -> None:
        nonlocal buffer, order
        body = _clean("\n\n".join(buffer))
        if body:
            sections.append(
                Section(heading=heading, text=body, source=f"{name} §{heading}", order=order)
            )
            order += 1
        buffer = []

    for paragraph in document.paragraphs:
        text = (paragraph.text or "").strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
        if style.startswith("heading") or style == "title":
            flush()
            heading = text[:120]
            continue
        buffer.append(text)
    flush()

    for table_index, table in enumerate(document.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            sections.append(
                Section(
                    heading=f"Table {table_index}",
                    text="\n".join(rows),
                    source=f"{name} table {table_index}",
                    order=order,
                )
            )
            order += 1

    figures = _docx_images(name, data, workdir, warnings)
    if not sections:
        warnings.append(f"{name}: no text found (document may be all images).")
    return sections, figures


def _docx_images(name: str, data: bytes, workdir: Path, warnings: list[str]) -> list[Figure]:
    """Pull embedded media straight out of the OOXML zip."""
    figures: list[Figure] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for entry in archive.namelist():
                if not entry.startswith("word/media/"):
                    continue
                suffix = Path(entry).suffix.lower()
                if suffix not in IMAGE_SUFFIXES:
                    continue
                payload = archive.read(entry)
                if len(payload) > 8 * 1024 * 1024:
                    warnings.append(f"{name}: skipped an oversized embedded image.")
                    continue
                target = workdir / f"{Path(name).stem}_{Path(entry).name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                figures.append(
                    Figure(
                        name=target.name,
                        path=str(target),
                        origin=f"{name} ({entry})",
                        caption=f"Figure extracted from {name}",
                    )
                )
    except zipfile.BadZipFile:
        warnings.append(f"{name}: not a valid DOCX archive.")
    return figures


def _save_uploaded_image(name: str, data: bytes, workdir: Path) -> Figure | None:
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return None
    target = workdir / f"upload_{Path(name).stem}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return Figure(
        name=target.name,
        path=str(target),
        origin=name,
        caption=f"Image supplied by the student ({name})",
    )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def extract(
    files: list[tuple[str, bytes]],
    workdir: str | Path,
    *,
    title: str | None = None,
) -> Document:
    """Turn ``[(filename, bytes), ...]`` into a :class:`Document`.

    ``workdir`` is where extracted figures are written. Errors become warnings.
    """
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    document = Document(title=title or _title_from(files))

    if len(files) > MAX_FILES:
        document.warnings.append(f"Only the first {MAX_FILES} files were read.")
        files = files[:MAX_FILES]

    for filename, data in files:
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.lower()

        if suffix not in ALLOWED_SUFFIXES:
            document.warnings.append(f"{safe_name}: unsupported type {suffix or '(none)'} — skipped.")
            continue
        if len(data) > MAX_FILE_BYTES:
            document.warnings.append(
                f"{safe_name}: {len(data) / 1e6:.1f} MB exceeds the {MAX_FILE_BYTES // 1024 // 1024} MB limit — skipped."
            )
            continue
        if not data:
            document.warnings.append(f"{safe_name}: empty file — skipped.")
            continue

        document.files.append(safe_name)

        if suffix in IMAGE_SUFFIXES:
            figure = _save_uploaded_image(safe_name, data, workdir_path)
            if figure:
                document.figures.append(figure)
                document.warnings.append(
                    f"{safe_name}: stored as a figure. Images are embedded, not interpreted — "
                    "reading a diagram's content needs a vision model."
                )
            continue

        if suffix in TEXT_SUFFIXES:
            text = data.decode("utf-8", errors="replace")
            sections = _from_markdown(safe_name, text) if suffix != ".txt" else _from_text(safe_name, text)
        elif suffix == ".pdf":
            sections = _from_pdf(safe_name, data, document.warnings)
        else:  # .docx
            sections, figures = _from_docx(safe_name, data, document.warnings, workdir_path)
            document.figures.extend(figures)

        if not sections and suffix in DOC_SUFFIXES:
            continue
        document.sections.extend(sections)

    # Files are processed in upload order and each parser emits sections in
    # document order, so this only renumbers — sorting by (order, heading) here
    # would interleave the sections of different files.
    for index, section in enumerate(document.sections):
        section.order = index

    total = document.chars
    if total > MAX_TOTAL_CHARS:
        document.warnings.append(
            f"Material is {total:,} chars; only the first {MAX_TOTAL_CHARS:,} are summarised."
        )
        document.sections = _truncate(document.sections, MAX_TOTAL_CHARS)

    if not document.has_text:
        document.warnings.append(
            "No usable text was extracted. If this is a scanned PDF or a slide deck exported as "
            "images, OCR is required and is not part of this build."
        )
    return document


def _truncate(sections: list[Section], limit: int) -> list[Section]:
    kept: list[Section] = []
    used = 0
    for section in sections:
        if used >= limit:
            break
        remaining = limit - used
        if len(section.text) > remaining:
            section = Section(
                heading=section.heading,
                text=section.text[:remaining],
                source=section.source,
                order=section.order,
                page=section.page,
            )
        kept.append(section)
        used += len(section.text)
    return kept


def _title_from(files: list[tuple[str, bytes]]) -> str:
    if not files:
        return "Uploaded material"
    stems = [_clean_stem(Path(name).stem) for name, _ in files[:3]]
    stems = [stem for stem in stems if stem]
    if not stems:
        return "Uploaded material"
    if len(files) > 1:
        return f"{stems[0].title()} (+{len(files) - 1} more)"
    return stems[0].title()


def _clean_stem(stem: str) -> str:
    """Drop the ``01_`` / ``2-`` ordering prefixes that make titles look like files."""
    stem = re.sub(r"^\s*\d+[._\-\s]+", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip()


def chunk(document: Document, max_chars: int = 1800) -> list[Section]:
    """Split over-long sections so a single chunk fits the model's context."""
    pieces: list[Section] = []
    for section in document.sections:
        if len(section.text) <= max_chars:
            pieces.append(section)
            continue
        chunks = rag._split_block(section.text, max_chars)
        for index, chunk in enumerate(chunks, start=1):
            pieces.append(
                Section(
                    heading=f"{section.heading} ({index}/{len(chunks)})",
                    text=chunk,
                    source=section.source,
                    order=section.order,
                    page=section.page,
                )
            )
    return pieces
