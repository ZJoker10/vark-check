"""Report model -> HTML -> PDF / DOCX / Markdown.

Each style gets a genuinely different document, not a recoloured one:

| Style | Structure | Visual machinery |
|---|---|---|
| V | summary, diagrams, tables, then short explanatory sections | Mermaid diagrams up front, comparison tables, tight captions |
| A | summary, then spoken-style sections | large serif type, dialogue blocks, no diagrams or tables |
| R | summary, key terms, numbered sections, tables | small-caps headings, justified dense prose, glossary |
| K | summary, numbered steps, self-checks, then theory | step cards, dashed check boxes, monospace commands |

Diagrams arrive as SVG from the browser (Mermaid runs client-side, which avoids
needing Node or a headless browser). They are embedded as vector images in the
PDF and rasterised to PNG for DOCX. If no SVG arrives — JavaScript disabled, or
Mermaid failed — the block degrades to the Mermaid source with a visible note
rather than silently vanishing.
"""

from __future__ import annotations

import base64
import html as html_module
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown as markdown_lib
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .profile import Profile, label
from .summarize import PROMPT_VERSION, Report

TEMPLATES = Path(__file__).resolve().parent / "templates"

_SVG_ROOT = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_VIEWBOX = re.compile(r'viewBox\s*=\s*"([-\d.\s,]+)"', re.IGNORECASE)

PALETTE: dict[str, tuple[str, str]] = {
    "V": ("#0e7490", "#e6f6f9"),
    "A": ("#b4530a", "#fdf3e7"),
    "R": ("#3730a3", "#ecebfb"),
    "K": ("#166534", "#eaf5ec"),
}

BLOCK_ORDER: dict[str, list[str]] = {
    "V": ["summary", "diagrams", "tables", "sections", "terms", "plan"],
    "A": ["summary", "sections", "terms", "plan"],
    "R": ["summary", "terms", "sections", "tables", "plan"],
    "K": ["summary", "sections", "checks", "diagrams", "terms", "plan"],
}

_SCRIPT_BLOCK = re.compile(
    r"<\s*(script|style|iframe|object|embed)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_DANGEROUS = re.compile(
    r"<\s*/?\s*(script|iframe|object|embed|link|style|meta|form|input|button)\b[^>]*>",
    re.IGNORECASE,
)
_HANDLER = re.compile(r"\son[a-z]+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)", re.IGNORECASE)
_JS_URL = re.compile(r"(javascript|vbscript|data\s*:\s*text/html)\s*:", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def sanitize(fragment: str) -> str:
    """Strip active content from model-generated HTML.

    The model's prose ends up in a browser preview and in a PDF built from HTML.
    Even in a single-user local app, executing whatever a model emits is a bad
    habit to ship.
    """
    fragment = _SCRIPT_BLOCK.sub("", fragment)
    fragment = _DANGEROUS.sub("", fragment)
    fragment = _HANDLER.sub("", fragment)
    fragment = _JS_URL.sub("", fragment)
    return fragment


def to_markdown_html(text: str) -> str:
    rendered = markdown_lib.markdown(
        text or "",
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        output_format="html5",
    )
    return sanitize(rendered)


def _plain(text: str) -> str:
    """Markdown to plain text, for formats that do not render it (DOCX runs)."""
    text = re.sub(r"```.*?```", "", text or "", flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _esc(text: Any) -> str:
    return html_module.escape(str(text or ""))


# --------------------------------------------------------------------------- #
# Profile chart (PIL, no matplotlib dependency)
# --------------------------------------------------------------------------- #
def profile_chart(profile: Profile, width: int = 820, height: int = 300) -> str:
    """Horizontal bar chart of the four scores, as base64 PNG."""
    from PIL import Image, ImageDraw, ImageFont

    scale = 2
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22 * scale)
        small = ImageFont.truetype("DejaVuSans.ttf", 18 * scale)
    except OSError:  # pragma: no cover - font always present on Linux boxes here
        font = ImageFont.load_default()
        small = font

    ranked = profile.ranked()
    top = ranked[0][0] if ranked else "V"
    bar_height = 34 * scale
    gap = 18 * scale
    left = 150 * scale
    right = width * scale - 70 * scale
    y = 40 * scale

    for dimension, value in ranked:
        accent = PALETTE.get(dimension, ("#555", "#eee"))[0]
        draw.text((30 * scale, y - 2 * scale), label(dimension), font=font, fill="#1c1f24")
        draw.rectangle([left, y, right, y + bar_height], fill="#eef1f4")
        filled = int((right - left) * max(0.0, min(1.0, value)))
        colour = accent if dimension == top else "#9aa5b1"
        draw.rectangle([left, y, left + filled, y + bar_height], fill=colour)
        draw.text(
            (right + 12 * scale, y + 4 * scale),
            f"{value:.2f}",
            font=small,
            fill="#1c1f24" if dimension == top else "#6b7280",
        )
        y += bar_height + gap

    draw.text((30 * scale, height * scale - 40 * scale), "0", font=small, fill="#9aa0a8")
    draw.text((left, height * scale - 40 * scale), "0.5", font=small, fill="#9aa0a8")
    draw.text((right - 20 * scale, height * scale - 40 * scale), "1.0", font=small, fill="#9aa0a8")

    buffer = io.BytesIO()
    image.resize((width, height)).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# HTML blocks
# --------------------------------------------------------------------------- #
def _fit_svg(svg: str) -> str:
    """Give an SVG an explicit intrinsic size, taken from its own viewBox.

    Mermaid emits ``width="100%"`` plus a ``max-width`` on the root element.
    Inside an ``<img>`` that leaves no aspect ratio at all, and WeasyPrint then
    stretches the drawing to the page width — a portrait flowchart ends up
    squashed and unreadable. An explicit width/height restores the ratio the
    author already encoded in the viewBox.
    """
    root = _SVG_ROOT.search(svg)
    if not root:
        return svg
    tag = root.group(0)
    box = _VIEWBOX.search(tag)
    if not box:
        return svg
    parts = [part for part in re.split(r"[\s,]+", box.group(1).strip()) if part]
    if len(parts) != 4:
        return svg
    try:
        width, height = float(parts[2]), float(parts[3])
    except ValueError:
        return svg
    if width <= 0 or height <= 0:
        return svg
    cleaned = re.sub(r'\s(?:width|height|style)\s*=\s*"[^"]*"', "", tag)
    cleaned = cleaned[:-1].rstrip() + f' width="{width:g}" height="{height:g}">'
    return cleaned + svg[root.end() :]


def _diagram_block(diagram: dict[str, Any], index: int, svg: str | None, preview: bool = False) -> str:
    title = _esc(diagram.get("title") or "Diagram")
    caption = _esc(diagram.get("caption"))
    code = _esc(diagram.get("mermaid"))
    if svg:
        payload = base64.b64encode(_fit_svg(svg).encode("utf-8")).decode("ascii")
        body = f'<img src="data:image/svg+xml;base64,{payload}" alt="{title}">'
    elif preview:
        # The browser draws this with Mermaid, then posts the SVG back for export.
        body = f'<div class="mermaid" data-diagram="{index}">{code}</div>'
    else:
        body = (
            f'<div class="pending">{code}</div>'
            '<div class="note">This diagram could not be drawn, so its source is shown '
            "instead. The Markdown download keeps the same source.</div>"
        )
    caption_html = f'<div class="caption">{caption}</div>' if caption else ""
    return f'<div class="diagram"><div class="title">{title}</div>{body}{caption_html}</div>'


def _table_block(table: dict[str, Any]) -> str:
    columns = table.get("columns") or []
    rows = table.get("rows") or []
    head = "".join(f"<th>{_esc(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    title = f'<div class="table-title">{_esc(table.get("title"))}</div>' if table.get("title") else ""
    return (
        f"{title}<table class=\"data\"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _section_block(section: dict[str, Any], index: int, style: str) -> str:
    heading = _esc(section.get("heading"))
    body = to_markdown_html(section.get("body", ""))
    source = _esc(section.get("source"))
    chip = f'<div class="source">{source}</div>' if source else ""
    if style == "K":
        return (
            f'<div class="step"><div class="n">STEP {index}</div>'
            f"<h3>{heading}</h3>{body}{chip}</div>"
        )
    return f"<h3>{heading}</h3>{body}{chip}"


def _term_block(term: dict[str, Any]) -> str:
    source = f' <span class="source">{_esc(term.get("source"))}</span>' if term.get("source") else ""
    return f"<dt>{_esc(term.get('term'))}</dt><dd>{_esc(term.get('meaning'))}{source}</dd>"


def _check_block(check: dict[str, Any]) -> str:
    hint = f'<div class="h">Hint: {_esc(check.get("hint"))}</div>' if check.get("hint") else ""
    return f'<div class="check"><div class="q">{_esc(check.get("question"))}</div>{hint}</div>'


def overview_title(style: str) -> str:
    """What the overview block is called. Read/write gets the formal word."""
    return "Abstract" if style.upper() == "R" else "The whole picture"


def build_blocks(report: Report, scholarly_h2: Any, svgs: dict[int, str], preview: bool = False) -> str:
    style = report.style.upper()
    order = BLOCK_ORDER.get(style, BLOCK_ORDER["V"])
    pieces: dict[str, str] = {}

    if report.summary:
        pieces["summary"] = report.summary

    diagram_index = 0
    if report.diagrams:
        diagrams = []
        for diagram in report.diagrams:
            diagrams.append(_diagram_block(diagram, diagram_index, svgs.get(diagram_index), preview))
            diagram_index += 1
        pieces["diagrams"] = "\n".join(diagrams)

    if report.tables:
        pieces["tables"] = "\n".join(_table_block(table) for table in report.tables)

    if report.sections:
        pieces["sections"] = "\n".join(
            _section_block(section, index + 1, style) for index, section in enumerate(report.sections)
        )

    if report.key_terms:
        pieces["terms"] = "<dl class=\"terms\">" + "".join(_term_block(term) for term in report.key_terms) + "</dl>"

    if report.checks:
        pieces["checks"] = "\n".join(_check_block(check) for check in report.checks)

    if report.study_plan:
        items = "".join(f"<li>{_esc(item)}</li>" for item in report.study_plan)
        pieces["plan"] = f'<div class="plan"><h2>What to do next</h2><ol>{items}</ol></div>'

    body: list[str] = []
    number = 0
    for name in order:
        if name not in pieces:
            continue
        if name == "summary":
            number += 1
            body.append(
                scholarly_h2(number, overview_title(style))
                + f'<div class="summary">{to_markdown_html(pieces["summary"])}</div>'
            )
            continue
        titles = {
            "diagrams": "Diagrams",
            "tables": "Comparisons",
            "sections": "Steps" if style == "K" else "The material, explained",
            "terms": "Terms to know",
            "checks": "Check yourself",
        }
        if name != "plan":
            number += 1
            body.append(scholarly_h2(number, titles.get(name, name.title())))
        body.append(pieces[name])
    return "\n".join(body)


def _h2_factory(style: str):
    def h2(number: int, text: str) -> str:
        prefix = f'<span class="num">{number}.</span>' if style in ("R", "K", "V") else ""
        return f"<h2>{prefix}{_esc(text)}</h2>"

    return h2


# --------------------------------------------------------------------------- #
# Public renderers
# --------------------------------------------------------------------------- #
def to_html(
    report: Report,
    profile: Profile,
    *,
    document: Any | None = None,
    svgs: dict[int, str] | None = None,
    preview: bool = False,
    generated_at: str | None = None,
) -> str:
    style = report.style.upper()
    accent, tint = PALETTE.get(style, PALETTE["V"])
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")
    ranked = profile.ranked()
    dominant = ranked[0] if ranked else (style, 1.0)
    secondary = ranked[1] if len(ranked) > 1 and ranked[1][1] >= 0.5 else None

    return template.render(
        report=report,
        style=style,
        style_label=label(style),
        accent=accent,
        tint=tint,
        profile_chart=profile_chart(profile),
        dominant_score=f"{dominant[1]:.2f}",
        secondary=bool(secondary),
        secondary_label=label(secondary[0]) if secondary else "",
        secondary_score=f"{secondary[1]:.2f}" if secondary else "",
        files=", ".join(getattr(document, "files", []) or []) or "pasted text",
        source_chars=f"{getattr(document, 'chars', 0):,}",
        section_count=len(getattr(document, "sections", []) or []),
        assessed=(profile.completed_at or "").replace("T", " ").replace("+00:00", " UTC"),
        warnings=list(report.warnings),
        blocks=build_blocks(report, _h2_factory(style), svgs or {}, preview),
        generated_at=generated_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
        prompt_version=PROMPT_VERSION,
        tokens=f"{report.tokens:,}",
    )


def to_pdf(html: str, base_dir: str | Path | None = None) -> bytes:
    from weasyprint import HTML

    document = HTML(string=html, base_url=str(base_dir) if base_dir else None).render()
    buffer = io.BytesIO()
    document.write_pdf(buffer)
    return buffer.getvalue()


def _svg_to_png(svg: str, scale: float = 2.0) -> bytes | None:
    try:
        import cairosvg
    except ImportError:  # pragma: no cover
        return None
    try:
        return cairosvg.svg2png(bytestring=svg.encode("utf-8"), scale=scale)
    except Exception:  # noqa: BLE001 - a bad SVG must not break the export
        return None


def to_docx(report: Report, profile: Profile, *, svgs: dict[int, str] | None = None) -> bytes:
    """Editable Word export. Diagrams become PNGs, Markdown becomes plain runs."""
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor, Inches

    svgs = svgs or {}
    document = DocxDocument()
    core = document.core_properties
    core.title = report.title
    core.author = "VARK Check"

    document.add_heading(report.title, level=0)
    if report.one_line:
        paragraph = document.add_paragraph(report.one_line)
        paragraph.runs[0].italic = True

    chart = io.BytesIO(base64.b64decode(profile_chart(profile)))
    document.add_picture(chart, width=Inches(5.5))

    caveat = document.add_paragraph()
    run = caveat.add_run(
        f"Tuned for a {label(report.style)} preference because that is how you described yourself, "
        "not because you have been diagnosed. Evidence that matching teaching to a preferred style "
        "improves learning is weak (Pashler et al., 2008)."
    )
    run.italic = True
    run.font.size = Pt(9)

    for warning in report.warnings:
        warn = document.add_paragraph()
        warn_run = warn.add_run(f"Note: {warning}")
        warn_run.font.size = Pt(9)
        warn_run.font.color.rgb = RGBColor(0x8A, 0x5A, 0x00)

    document.add_heading(overview_title(report.style), level=1)
    for paragraph in _plain(report.summary).split("\n\n"):
        if paragraph.strip():
            document.add_paragraph(paragraph.strip())

    if report.diagrams:
        document.add_heading("Diagrams", level=1)
        for index, diagram in enumerate(report.diagrams):
            document.add_paragraph().add_run(diagram.get("title") or "Diagram").bold = True
            png = _svg_to_png(svgs.get(index, "")) if svgs.get(index) else None
            if png:
                document.add_picture(io.BytesIO(png), width=Inches(6.0))
            else:
                note = document.add_paragraph(f"[Mermaid source] {diagram.get('mermaid', '')}")
                note.runs[0].font.size = Pt(8)
            if diagram.get("caption"):
                caption = document.add_paragraph(diagram["caption"])
                caption.runs[0].italic = True
                caption.runs[0].font.size = Pt(9)

    if report.tables:
        document.add_heading("Comparisons", level=1)
        for table in report.tables:
            document.add_paragraph().add_run(table.get("title") or "Comparison").bold = True
            columns = table.get("columns") or []
            rows = table.get("rows") or []
            if not columns:
                continue
            grid = document.add_table(rows=1, cols=len(columns))
            grid.style = "Light Grid Accent 1"
            for position, column in enumerate(columns):
                grid.rows[0].cells[position].text = str(column)
            for row in rows:
                cells = grid.add_row().cells
                for position, cell in enumerate(row[: len(columns)]):
                    cells[position].text = str(cell)

    document.add_heading("Steps" if report.style == "K" else "The material, explained", level=1)
    for index, section in enumerate(report.sections, start=1):
        prefix = f"{index}. " if report.style == "K" else ""
        document.add_heading(prefix + str(section.get("heading", "")), level=2)
        for paragraph in _plain(section.get("body", "")).split("\n\n"):
            if paragraph.strip():
                document.add_paragraph(paragraph.strip())
        if section.get("source"):
            source = document.add_paragraph(f"Source: {section['source']}")
            source.runs[0].font.size = Pt(8)
            source.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

    if report.key_terms:
        document.add_heading("Terms to know", level=1)
        for term in report.key_terms:
            paragraph = document.add_paragraph()
            paragraph.add_run(f"{term.get('term')}: ").bold = True
            paragraph.add_run(str(term.get("meaning") or ""))

    if report.checks:
        document.add_heading("Check yourself", level=1)
        for check in report.checks:
            paragraph = document.add_paragraph(style="List Number")
            paragraph.add_run(str(check.get("question") or "")).bold = True
            if check.get("hint"):
                hint = document.add_paragraph(f"Hint: {check['hint']}")
                hint.paragraph_format.left_indent = Inches(0.3)
                hint.runs[0].font.size = Pt(9)

    if report.study_plan:
        document.add_heading("What to do next", level=1)
        for item in report.study_plan:
            document.add_paragraph(str(item), style="List Bullet")

    if report.citations:
        document.add_heading("Sources", level=1)
        for citation in report.citations:
            paragraph = document.add_paragraph(str(citation))
            paragraph.runs[0].font.size = Pt(8)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def to_markdown(report: Report, profile: Profile) -> str:
    lines: list[str] = [
        f"# {report.title}",
        "",
        f"*{report.one_line}*" if report.one_line else "",
        "",
        f"> Tuned for a **{label(report.style)}** preference "
        f"({', '.join(f'{k}={v:.2f}' for k, v in profile.ranked())}) — a starting hypothesis, "
        "not a diagnosis. Pashler et al. (2008).",
        "",
    ]
    if report.warnings:
        lines.extend(["> **Worth knowing:**", *[f"> - {w}" for w in report.warnings], ""])

    lines.extend([f"## {overview_title(report.style)}", "", report.summary, ""])

    if report.diagrams:
        lines.extend(["## Diagrams", ""])
        for diagram in report.diagrams:
            lines.extend([f"### {diagram.get('title')}", "", "```mermaid", diagram.get("mermaid", ""), "```", ""])
            if diagram.get("caption"):
                lines.extend([f"*{diagram['caption']}*", ""])

    if report.tables:
        lines.extend(["## Comparisons", ""])
        for table in report.tables:
            lines.append(f"**{table.get('title') or 'Comparison'}**")
            lines.append("")
            columns = table.get("columns") or []
            lines.append("| " + " | ".join(str(c) for c in columns) + " |")
            lines.append("|" + "|".join("---" for _ in columns) + "|")
            for row in table.get("rows") or []:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("")

    lines.extend(["## The material, explained", ""])
    for section in report.sections:
        lines.extend([f"### {section.get('heading')}", "", str(section.get("body") or ""), ""])
        if section.get("source"):
            lines.extend([f"`source: {section['source']}`", ""])

    if report.key_terms:
        lines.extend(["## Terms to know", ""])
        for term in report.key_terms:
            lines.append(f"- **{term.get('term')}** — {term.get('meaning')}")
        lines.append("")

    if report.checks:
        lines.extend(["## Check yourself", ""])
        for check in report.checks:
            lines.append(f"1. {check.get('question')}")
            if check.get("hint"):
                lines.append(f"   - _hint: {check['hint']}_")
        lines.append("")

    if report.study_plan:
        lines.extend(["## What to do next", ""])
        lines.extend(f"- {item}" for item in report.study_plan)
        lines.append("")

    if report.citations:
        lines.extend(["## Sources", ""])
        lines.extend(f"- {citation}" for citation in report.citations)
        lines.append("")

    return "\n".join(line for line in lines if line is not None)


def render(
    report: Report,
    profile: Profile,
    fmt: str,
    *,
    document: Any | None = None,
    svgs: dict[int, str] | None = None,
) -> tuple[bytes, str, str]:
    """Return ``(payload, mimetype, filename)`` for the requested format."""
    fmt = (fmt or "pdf").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", (report.title or "study-report").lower()).strip("-")[:60] or "study-report"

    if fmt == "pdf":
        html = to_html(report, profile, document=document, svgs=svgs)
        return to_pdf(html), "application/pdf", f"{slug}.pdf"
    if fmt == "docx":
        return (
            to_docx(report, profile, svgs=svgs),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            f"{slug}.docx",
        )
    if fmt in ("md", "markdown"):
        return to_markdown(report, profile).encode("utf-8"), "text/markdown; charset=utf-8", f"{slug}.md"
    if fmt == "html":
        html = to_html(report, profile, document=document, svgs=svgs)
        return html.encode("utf-8"), "text/html; charset=utf-8", f"{slug}.html"
    raise ValueError(f"Unsupported format {fmt!r}. Use pdf, docx, md or html.")
