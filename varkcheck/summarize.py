"""Turn an uploaded document plus a VARK profile into a genuine study report.

Two things make this a *summary* and not a reshuffle of excerpts:

1. **Coverage.** Passages are selected by striding across the whole document, and
   the model is given the complete heading outline. A summariser that only sees
   the first five chunks writes a summary of the introduction.
2. **A copy ceiling.** After generation, :mod:`varkcheck.synthesis` measures how
   much of the report is verbatim source. A report over the ceiling is sent back
   once with the offending sentences quoted at it.

Grounding is enforced differently from v1. With arbitrary uploads there is no
fixed corpus to allow-list, so citations are resolved against the passages that
were actually sent: an unresolvable source is replaced by the best-matching real
passage, or the item is dropped. Either way it is logged.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import synthesis, uploads
from .llm import LLMError, extract_json
from .prompts import load_task_prompts
from .profile import Profile, label

PASSAGE_CHARS = 1800
DEFAULT_BUDGET_CHARS = 16_000
PROMPT_VERSION = "2.0.0"
REPORT_MAX_TOKENS = 4000

_MERMAID_START = re.compile(r"^\s*(flowchart|graph|sequenceDiagram|mindmap|stateDiagram|classDiagram)\b", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    title: str
    one_line: str
    summary: str
    style: str
    sections: list[dict[str, Any]] = field(default_factory=list)
    key_terms: list[dict[str, Any]] = field(default_factory=list)
    diagrams: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    study_plan: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Process facts ("a first draft was rewritten") are shown beside the preview
    # but kept out of the exported document: whoever reads the PDF wants to know
    # what is wrong with *this* report, not how many drafts produced it.
    notes: list[str] = field(default_factory=list)
    synthesis_ratio: float = 0.0
    synthesis_ok: bool = True
    coverage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    tokens: int = 0
    dropped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "one_line": self.one_line,
            "summary": self.summary,
            "style": self.style,
            "style_label": label(self.style),
            "sections": self.sections,
            "key_terms": self.key_terms,
            "diagrams": self.diagrams,
            "tables": self.tables,
            "checks": self.checks,
            "study_plan": self.study_plan,
            "citations": self.citations,
            "warnings": self.warnings,
            "notes": self.notes,
            "synthesis": {"ratio": round(self.synthesis_ratio, 3), "ok": self.synthesis_ok},
            "coverage": self.coverage,
            "attempts": self.attempts,
            "tokens": self.tokens,
            "dropped": self.dropped,
        }

    def plain_text(self) -> str:
        """Markdown-ish flattening, used by the synthesis check and .md export."""
        parts = [self.summary]
        for section in self.sections:
            parts.append(f"{section.get('heading', '')}\n{section.get('body', '')}")
        for term in self.key_terms:
            parts.append(f"{term.get('term', '')}: {term.get('meaning', '')}")
        for check in self.checks:
            parts.append(str(check.get("question", "")))
        parts.extend(str(item) for item in self.study_plan)
        return "\n\n".join(part for part in parts if part)


# --------------------------------------------------------------------------- #
# Passage selection
# --------------------------------------------------------------------------- #
def select_passages(document: uploads.Document, budget_chars: int = DEFAULT_BUDGET_CHARS) -> list[uploads.Section]:
    """Evenly strided selection, so the report reflects the whole document."""
    chunks = uploads.chunk(document, max_chars=PASSAGE_CHARS)
    if not chunks:
        return []

    total = sum(len(chunk.text) for chunk in chunks)
    if total <= budget_chars:
        return chunks

    room = max(1, budget_chars // PASSAGE_CHARS)
    if room >= len(chunks):
        return chunks

    # Include the first and last chunk, then fill evenly in between.
    step = (len(chunks) - 1) / max(1, room - 1)
    indices = sorted({min(len(chunks) - 1, round(index * step)) for index in range(room)})
    return [chunks[index] for index in indices]


def _major_parts(document: uploads.Document) -> list[str]:
    """Top-level parts of the material, for the "cover the whole document" rule.

    Markdown and DOCX give real headings. PDFs give one heading per page, which
    would inflate the count, so those fall back to the file list.
    """
    generated = re.compile(r"—\s*(page|part)\s+\d+\s*$", re.IGNORECASE)
    subsection = re.compile(r"^\s*\d+\.\d+")  # "2.3 The learning rate" is not a major part
    headings = [
        heading
        for heading in document.outline()
        if not generated.search(heading) and not subsection.match(heading)
    ]
    if len(headings) >= 2:
        return _unique(headings)
    return _unique(document.files) or [document.title]


def _node_label(text: str, limit: int = 5) -> str:
    """Mermaid-safe node label: no parentheses, quotes, colons or punctuation."""
    words = re.sub(r"[^A-Za-z0-9+\- ]", " ", text).split()
    label = " ".join(words[:limit]) or "Part"
    return label.replace("+", " plus ").strip() if "+" in label else label


def fallback_diagram(document: uploads.Document, headings: list[str]) -> dict[str, Any] | None:
    """A hierarchy chart built from the document's own structure.

    Visual learners are promised a diagram. Weaker local models sometimes
    return an unusable one, so rather than shipping an empty section we draw the
    outline we already parsed. It is honest — every label comes from the
    material — and the caption says where it came from.
    """
    parts = _unique(heading for heading in headings if heading)[:6]
    if len(parts) < 2:
        return None
    lines = ["flowchart TD", f"  R[{_node_label(document.title, 4)}]"]
    for index, part in enumerate(parts, start=1):
        lines.append(f"  P{index}[{_node_label(part)}]")
        lines.append(f"  R --> P{index}")
    return {
        "title": "How the material fits together",
        "caption": "Built from the document's own headings because the generated diagram was unusable.",
        "mermaid": "\n".join(lines),
    }


_META = re.compile(
    r"^\s*(this|the)\s+(report|summary|document|page)\b|^\s*in this (report|summary)\b",
    re.IGNORECASE,
)


def _quality_issues(
    report: "Report", document: uploads.Document, style: str, coverage: list[dict[str, Any]]
) -> list[str]:
    """Things the copy check cannot see but that still make the report wrong."""
    issues: list[str] = []
    words = len(report.summary.split())
    if words < 90:
        issues.append(f"The summary is {words} words. Write 120–220 words that name the material's parts.")
    elif words > 420:
        issues.append(f"The summary is {words} words. Cut it to 120–220; compression is the job.")
    if _META.match(report.summary):
        issues.append(
            "The summary describes the report ('This report covers...'). Describe the subject instead, "
            "as though the reader had not seen the material."
        )
    missing = _missing_parts(report, coverage)
    if missing:
        issues.append(
            "Nothing in the report comes from these parts: "
            + "; ".join(missing[:4])
            + ". Give each of them its own section."
        )
    if style == "V" and not report.diagrams:
        issues.append("There is no usable diagram. Produce at least one flowchart with three labelled nodes and two arrows.")
    return issues


def _section_part(section: uploads.Section, parts: list[str]) -> str | None:
    if section.heading in parts:
        return section.heading
    file_part = section.source.split(" §")[0].split(" p.")[0].strip()
    if file_part in parts:
        return file_part
    return None


def part_coverage(document: uploads.Document, limit: int = 6) -> list[dict[str, Any]]:
    """Group the document into parts, each with the sources that belong to it.

    Coverage is checked against this, so "summarise the whole document" stops
    being a prompt request and becomes a testable property. Parts are taken by
    even stride when there are many, so the report still spans the whole thing.
    """
    parts = _major_parts(document)
    if not parts:
        return []

    current = parts[0]
    groups: dict[str, list[str]] = {part: [] for part in parts}
    for section in document.sections:
        named = _section_part(section, parts)
        if named is not None:
            current = named
        groups[current].append(section.source)

    filled = [{"label": part, "sources": _unique(groups[part])} for part in parts if groups[part]]
    if len(filled) > limit:
        step = (len(filled) - 1) / (limit - 1)
        filled = [filled[round(index * step)] for index in range(limit)]
    return filled


def _missing_parts(report: "Report", coverage: list[dict[str, Any]]) -> list[str]:
    cited = set(report.citations)
    missing = []
    for part in coverage:
        if not cited.intersection(part["sources"]):
            missing.append(part["label"])
    return missing


def _passage_block(passages: list[uploads.Section]) -> str:
    blocks = []
    for passage in passages:
        blocks.append(f"[source: {passage.source}]\n{passage.text}")
    return "\n\n---\n\n".join(blocks)


def material_hash(document: uploads.Document) -> str:
    digest = hashlib.sha256()
    for section in document.sections:
        digest.update(section.source.encode())
        digest.update(section.text.encode())
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]{3,}", text.lower())}


def _resolve(citation: str, passages: list[uploads.Section]) -> str | None:
    """Map a model-written citation onto a real passage source."""
    citation = (citation or "").strip()
    if not citation:
        return None
    normalised = citation.lower().replace("§", "").strip()
    for passage in passages:
        source = passage.source
        if citation == source:
            return source
        if normalised in source.lower().replace("§", "") or source.lower().replace("§", "") in normalised:
            return source
    # Fall back to the file name, which is the part the model is most likely to
    # reproduce correctly.
    for passage in passages:
        file_part = passage.source.split(" §")[0].split(" p.")[0].strip().lower()
        if file_part and file_part in normalised:
            return passage.source
    return None


def _best_source(text: str, passages: list[uploads.Section]) -> str | None:
    tokens = _tokens(text)
    if not tokens:
        return None
    scored = []
    for passage in passages:
        overlap = len(tokens & _tokens(passage.heading + " " + passage.text))
        if overlap:
            scored.append((overlap, passage.source))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


# --------------------------------------------------------------------------- #
# Normalisation of model output
# --------------------------------------------------------------------------- #
def _text(value: Any, limit: int = 4000) -> str:
    return str(value or "").strip()[:limit]


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _unique(values: Any) -> list[str]:
    """Order-preserving de-duplication; several passages share one section source."""
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


_EDGE = re.compile(r"(-->|---|==>|-.->)")
_LABEL = re.compile(r"[\[\{\(]\s*([^\]\}\)]+?)\s*[\]\}\)]")


def _valid_mermaid(code: str) -> bool:
    code = (code or "").strip()
    if len(code) < 12 or len(code) > 2500:
        return False
    if not _MERMAID_START.search(code):
        return False
    # Node labels containing raw parentheses or quotes break Mermaid parsing.
    if re.search(r"\[[^\]]*[()\"][^\]]*\]", code):
        return False
    # Angle brackets outside <br> are read as HTML by Mermaid's lexer and abort
    # the whole diagram ("expecting 'SPACE', got 'TAGEND'"). Weaker models like
    # to write edge labels such as -->|<weighted sum>|, so refuse them here
    # rather than shipping a diagram the browser cannot draw. Arrows themselves
    # are made of these characters, so remove them before looking.
    stripped = re.sub(r"<br\s*/?>", "", code, flags=re.IGNORECASE)
    stripped = re.sub(r"<?[-=.]+>?", "", stripped)
    if re.search(r"[<>]", stripped):
        return False
    # A diagram of one arrow is a placeholder, not an explanation.
    if len(_EDGE.findall(code)) < 2:
        return False
    labels = [label.strip() for label in _LABEL.findall(code)]
    if len({label.lower() for label in labels}) < 3:
        return False
    # Reject the schema example, which weaker models sometimes echo verbatim.
    if {label.lower() for label in labels} <= {"a", "b"}:
        return False
    return True


def _normalise(raw: dict[str, Any], *, style: str, passages: list[uploads.Section]) -> Report:
    warnings: list[str] = []
    dropped = 0

    allowed = {passage.source for passage in passages}
    seen: set[str] = set()

    def resolve(value: Any, context: str) -> str | None:
        nonlocal dropped
        resolved = _resolve(_text(value, 200), passages)
        if resolved is None:
            resolved = _best_source(context, passages)
        if resolved is None:
            dropped += 1
            return None
        seen.add(resolved)
        return resolved

    sections: list[dict[str, Any]] = []
    for item in _list(raw.get("sections"))[:12]:
        if not isinstance(item, dict):
            continue
        heading = _text(item.get("heading"), 120)
        body = _text(item.get("body"))
        if not heading or not body:
            continue
        source = resolve(item.get("source"), f"{heading} {body}")
        if source is None:
            continue
        sections.append({"heading": heading, "body": body, "source": source})

    key_terms: list[dict[str, Any]] = []
    for item in _list(raw.get("key_terms"))[:10]:
        if not isinstance(item, dict):
            continue
        term = _text(item.get("term"), 80)
        meaning = _text(item.get("meaning"), 400)
        if not term or not meaning:
            continue
        source = resolve(item.get("source"), f"{term} {meaning}")
        if source is None:
            continue
        key_terms.append({"term": term, "meaning": meaning, "source": source})

    diagrams: list[dict[str, Any]] = []
    for item in _list(raw.get("diagrams"))[:4]:
        if not isinstance(item, dict):
            continue
        code = _text(item.get("mermaid"), 2500)
        if not _valid_mermaid(code):
            warnings.append(f"Diagram '{_text(item.get('title'), 60) or 'untitled'}' was malformed and is not shown.")
            continue
        diagrams.append(
            {
                "title": _text(item.get("title"), 100) or "Diagram",
                "caption": _text(item.get("caption"), 300),
                "mermaid": code,
            }
        )

    tables: list[dict[str, Any]] = []
    for item in _list(raw.get("tables"))[:3]:
        if not isinstance(item, dict):
            continue
        columns = [_text(column, 60) for column in _list(item.get("columns"))][:6]
        rows = [
            [_text(cell, 120) for cell in _list(row)][: len(columns)]
            for row in _list(item.get("rows"))[:12]
            if isinstance(row, list)
        ]
        rows = [row for row in rows if len(row) == len(columns) and columns]
        if columns and rows:
            tables.append(
                {
                    "title": _text(item.get("title"), 100) or "Comparison",
                    "columns": columns,
                    "rows": rows,
                }
            )

    checks: list[dict[str, Any]] = []
    for item in _list(raw.get("checks"))[:6]:
        if not isinstance(item, dict):
            continue
        question = _text(item.get("question"), 300)
        if question:
            checks.append({"question": question, "hint": _text(item.get("hint"), 300)})

    study_plan = [item for item in (_text(entry, 200) for entry in _list(raw.get("study_plan"))[:5]) if item]

    # The shape of the document is part of the deliverable, so it is enforced
    # here rather than trusted to the prompt: an aural report with diagrams is
    # not an aural report.
    if style == "A":
        diagrams, tables = [], []
    elif style == "R":
        diagrams = []
    elif style == "K":
        diagrams = diagrams[:1]

    summary = _text(raw.get("summary"), 2600)
    if not summary:
        summary = " ".join(section["body"] for section in sections[:2])[:1200]

    report = Report(
        title=_text(raw.get("title"), 120) or "Study report",
        one_line=_text(raw.get("one_line"), 300),
        summary=summary,
        style=style,
        sections=sections,
        key_terms=key_terms,
        diagrams=diagrams,
        tables=tables,
        checks=checks,
        study_plan=study_plan,
        citations=_unique(passage.source for passage in passages if passage.source in seen),
        warnings=warnings,
        dropped=dropped,
    )
    if not sections:
        raise ValueError("The model returned no usable sections.")
    return report


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def build_prompt(
    document: uploads.Document,
    profile: Profile,
    style: str,
    passages: list[uploads.Section],
    tasks: dict[str, str],
) -> tuple[str, str]:
    style = style.upper()
    shared = tasks.get("SHARED_RULES", "")
    specific = tasks.get(f"REPORT_{style}", tasks.get("SHARED_RULES", ""))
    schema = tasks.get("SCHEMA", "")

    outline = "\n".join(f"- {heading}" for heading in document.outline()[:60])
    coverage = part_coverage(document)
    if coverage:
        parts_block = "\n".join(
            f"{index}. {part['label']}\n   sources: "
            + "; ".join(part["sources"][:6])
            + (" …" if len(part["sources"]) > 6 else "")
            for index, part in enumerate(coverage, start=1)
        )
        section_target = len(coverage)
    else:
        parts_block = f"1. {document.title}\n   sources: (see passages)"
        section_target = 3
    system = (
        "You are a study-report writer. You summarise material the student uploaded. "
        "You never copy source sentences verbatim, you never invent facts, and every "
        "claim you make carries the exact source string of the passage it came from. "
        "Reply with a single JSON object and nothing else.\n\n"
        f"{shared}"
    )
    user = (
        f"DOCUMENT TITLE: {document.title}\n"
        f"FILES: {', '.join(document.files) or 'pasted text'}\n"
        f"LEARNER PROFILE: dominant {label(style)} "
        f"(scores: {', '.join(f'{k}={v:.2f}' for k, v in profile.ranked())})\n\n"
        f"THE MATERIAL HAS THESE PARTS — the report must contain at least one section drawing on "
        f"each, not just the first:\n{parts_block}\n\n"
        f"FULL OUTLINE OF THE DOCUMENT (for structure only — do not invent sources):\n{outline}\n\n"
        f"SELECTED PASSAGES:\n{_passage_block(passages)}\n\n"
        f"STYLE REQUIREMENTS:\n{specific}\n\n"
        f"OUTPUT SCHEMA:\n{schema}\n\n"
        f"DELIVERABLE CHECKLIST:\n"
        f"- `sections`: {section_target} entries — one per part listed above, in the material's order. "
        "Each section's `source` must be copied exactly from a `sources:` line belonging to that part.\n"
        f"- `summary`: {120 if style != 'R' else 140}–{220 if style != 'R' else 190} words that name "
        "the parts and say how they connect. A reader must be able to tell this is about the whole "
        "document.\n"
        "- `title`: names the actual subject of this material, not \"Study Report\" or \"Report on X\".\n"
        "- `one_line`: what the learner can do after reading; not a restatement of the title.\n\n"
        "Remember: synthesise. Do not reuse the passages' sentences. Reply with one JSON object and "
        "nothing else — no preamble, no markdown fences."
    )
    return system, user


def generate(
    document: uploads.Document,
    profile: Profile,
    style: str,
    *,
    llm: Any,
    prompts_dir: Any = None,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    max_attempts: int = 2,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> Report:
    """Generate the report: check the draft in code, retry once, then ship.

    The checks are the same on both attempts. A draft that fails the second time
    is returned anyway, with each failure written into ``report.warnings`` so the
    page can say what it is holding — an imperfect report that admits it beats an
    error page on the night before an exam.
    """
    from pathlib import Path

    style = style.upper()
    prompts_dir = Path(prompts_dir) if prompts_dir else Path(__file__).resolve().parent.parent / "prompts"
    tasks = load_task_prompts(prompts_dir / "report.md")

    passages = select_passages(document, budget_chars=budget_chars)
    if not passages:
        raise ValueError("No text to summarise. The upload produced no usable sections.")

    source_text = "\n\n".join(section.text for section in document.sections)
    coverage = part_coverage(document)
    system, user = build_prompt(document, profile, style, passages, tasks)

    tokens = 0
    report: Report | None = None
    feedback = ""
    process_notes: list[str] = []
    best: tuple[tuple[int, int, float], Report, Any, list[str], int] | None = None

    def finish(final: Report) -> Report:
        final.notes = process_notes + final.notes
        final.tokens = tokens
        return _finalise(final, document, style)

    for attempt in range(1, max_attempts + 1):
        prompt = f"{user}\n\n{feedback}".strip() if feedback else user

        try:
            reply = llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=REPORT_MAX_TOKENS,
                json_mode=True,
            )
        except LLMError as exc:
            raise LLMError(f"Report generation failed: {exc}") from exc

        tokens += getattr(getattr(reply, "usage", None), "total", 0) or 0

        try:
            parsed = extract_json(reply.text)
            report = _normalise(parsed, style=style, passages=passages)
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            report = None
            feedback = _error_feedback(exc)
            if on_event:
                on_event("report.invalid", {"attempt": attempt, "error": str(exc)[:200]})
            continue

        verdict = synthesis.check(report.plain_text(), source_text)
        issues = _quality_issues(report, document, style, coverage)
        missing = _missing_parts(report, coverage)
        report.coverage = {
            "parts": len(coverage),
            "covered": len(coverage) - len(missing),
            "missing": missing,
        }
        report.synthesis_ok = verdict.ok
        report.synthesis_ratio = verdict.copied_ratio
        report.attempts = attempt

        if on_event:
            on_event(
                "report.synthesis",
                {
                    "attempt": attempt,
                    "copied_ratio": round(verdict.copied_ratio, 3),
                    "ok": verdict.ok,
                    "issues": issues,
                },
            )

        if verdict.ok and not issues:
            return finish(report)

        # Keep the best draft seen so far, not merely the last one. Small models
        # often make the second attempt worse; shipping the worse of the two
        # because it happened later would be a needless regression.
        #
        # The ordering follows the app's promise: a genuine summary beats
        # everything, so a draft that passes the copy check is preferred over a
        # plagiarised one however many smaller problems it has. Among drafts that
        # both copy, the one that copies less wins.
        rank = (0, len(issues), verdict.copied_ratio) if verdict.ok else (1, 0, verdict.copied_ratio)
        if best is None or rank < best[0]:
            best = (rank, report, verdict, issues, attempt)

        if attempt < max_attempts:
            notes: list[str] = []
            if not verdict.ok:
                notes.append(synthesis.retry_instruction(verdict))
                process_notes.append(
                    f"The first draft reused source wording ({verdict.copied_ratio:.0%} of sentences); "
                    "it was rewritten."
                )
            if issues:
                notes.append(_issue_feedback(issues))
                process_notes.append(
                    "The first draft missed the brief (" + "; ".join(issues) + "); it was rewritten."
                )
            feedback = "\n\n".join(notes)
            continue

    # Nothing passed. Ship the least-bad draft, and say exactly what is wrong
    # with it: a report that admits its weaknesses is recoverable, a blank page
    # on the night before an exam is not.
    if best is not None:
        _, kept, verdict, issues, kept_at = best
        if kept_at < attempt:
            # We are shipping the earlier draft, so a note saying it "was
            # rewritten" would be a lie about the page in the reader's hands.
            process_notes = ["The rewrite came out worse, so the earlier draft is the one shown."]
        if not verdict.ok:
            kept.warnings.append(
                f"This report leans on the source wording ({verdict.copied_ratio:.0%} of sentences "
                "are close to verbatim). Treat it as a guided excerpt rather than a summary."
            )
        for issue in issues:
            kept.warnings.append(f"This report still: {issue}")
        kept.attempts = attempt
        return finish(kept)

    raise LLMError(f"Could not produce a valid report: {feedback or 'unknown error'}")


def _finalise(report: "Report", document: uploads.Document, style: str) -> "Report":
    """Last guarantee: a visual report always contains a visual."""
    if style == "V" and not report.diagrams:
        headings = _unique(section.get("heading", "") for section in report.sections)
        if len(headings) < 2:
            headings = _major_parts(document)
        diagram = fallback_diagram(document, headings)
        if diagram:
            report.diagrams = [diagram]
            # The chart drawn below is usable, so a warning printed above it
            # saying the report has no diagram would contradict the page.
            report.warnings = [
                warning for warning in report.warnings if "no usable diagram" not in warning
            ]
    return report


def _issue_feedback(issues: list[str]) -> str:
    return (
        "Your previous report was unusable for these reasons:\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\n\nFix exactly these. Keep everything that was already correct."
    )


def _error_feedback(error: Exception) -> str:
    return (
        "Your previous reply could not be used "
        f"({type(error).__name__}: {str(error)[:160]}).\n"
        "Reply with one JSON object that matches the schema exactly. Every key present, "
        "lists (not strings) for the list fields. `sections` must contain at least two "
        "entries, each with a heading, a body, and an exact source string copied from "
        "a `[source: ...]` line above."
    )


def as_json(report: Report) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
