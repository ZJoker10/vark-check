"""Declared vs observed comparison — deterministic on purpose.

This is arithmetic over rows in SQLite. Asking a language model to do it would
add latency, cost and nondeterminism for zero benefit, and it would make the
results screen impossible to unit-test. The agent explains; this module scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .questionnaire import DIMENSIONS, label, top_dimension

FORMATS = ("V", "A", "R", "K")


@dataclass
class FormatRow:
    format: str
    mean: float
    max_score: float
    attempts: int
    questions: int = 0

    @property
    def pct(self) -> float:
        return 0.0 if self.max_score <= 0 else round(self.mean / self.max_score, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "label": label(self.format),
            "mean": round(self.mean, 2),
            "max_score": self.max_score,
            "attempts": self.attempts,
            "pct": self.pct,
        }


def format_rows(attempts: Iterable[dict[str, Any]]) -> list[FormatRow]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        buckets.setdefault(str(attempt["format"]).upper(), []).append(attempt)

    rows: list[FormatRow] = []
    for fmt in FORMATS:
        items = buckets.get(fmt, [])
        if not items:
            continue
        rows.append(
            FormatRow(
                format=fmt,
                mean=sum(float(item["score"]) for item in items) / len(items),
                max_score=float(items[0].get("max_score") or 3),
                attempts=len(items),
            )
        )
    return rows


def best_format(rows: Iterable[FormatRow]) -> FormatRow | None:
    rows = list(rows)
    if not rows:
        return None
    return max(rows, key=lambda row: (row.pct, -row.attempts))


def comparison_message(declared: dict[str, float], rows: list[FormatRow], topic: str) -> str:
    """Neutral, non-labelling summary of the gap.

    Deliberately says "for this topic" and "worth trying", never "you are".
    """
    if not rows:
        return (
            f"No attempts recorded yet for \"{topic}\". Finish at least one format and "
            "the comparison will appear here."
        )

    declared_top = top_dimension(declared) if declared else None
    best = best_format(rows)
    assert best is not None

    parts = [
        f'For "{topic}", your highest score was in the **{label(best.format)}** format '
        f"({best.mean:.1f}/{best.max_score:.0f}, {best.attempts} attempt{'s' if best.attempts != 1 else ''})."
    ]

    if declared_top:
        parts.append(
            f"Your declared preference was **{label(declared_top)}** "
            f"(self-report score {declared.get(declared_top, 0):.2f} / 1.00)."
        )

    if declared_top and best.format == declared_top:
        parts.append(
            "Those agree here. That is one data point, not a rule — the next topic may disagree, "
            "and one topic is a small sample."
        )
    elif declared_top:
        parts.append(
            f"Worth trying {label(best.format)} first next time and seeing whether the result holds. "
            "The evidence for matching teaching to a preferred style is weak, so the useful question "
            "is not *what you are* but *what worked on this topic*."
        )

    weakest = min(rows, key=lambda row: row.pct)
    if weakest.format != best.format:
        parts.append(
            f"Lowest score here: {label(weakest.format)} ({weakest.mean:.1f}/{weakest.max_score:.0f}). "
            "That may be the format, or the specific questions — treat it as a hint, not a verdict."
        )

    parts.append("_Indicative only: a handful of short questions per format._")
    return "\n\n".join(parts)


def coverage_warning(rows: list[FormatRow]) -> str | None:
    missing = [label(fmt) for fmt in FORMATS if fmt not in {row.format for row in rows}]
    if not missing:
        return None
    return "Not attempted yet: " + ", ".join(missing) + ". The comparison is partial."


def suggested_next(rows: list[FormatRow]) -> str | None:
    best = best_format(rows)
    if best is None:
        return None
    return f"Start the next topic with the {label(best.format)} format."


def lecturer_digest(store, top_n: int = 3) -> list[dict[str, Any]]:
    """Top concepts the group has not grasped, with the course source."""
    digest = store.concept_digest(top_n=top_n)
    for entry in digest:
        entry["label"] = entry.get("concept", "")
    return digest


def declared_vector(declared: dict[str, Any]) -> dict[str, float]:
    """Normalise whatever is stored in SQLite into {dimension: score}."""
    return {dimension: float(declared.get(dimension, 0.0)) for dimension in DIMENSIONS}
