"""The VARK profile: scored once, remembered in the browser.

The brief for v2 changed on one point: the report *is* tuned to the learner's
dominant style. What did not change is that the score is arithmetic, not a model
call, and that it is presented as a starting hypothesis.

Persistence lives in the browser (``localStorage``), which is why this module is
the only place that knows how to serialise and validate a profile. The server
stores nothing about who the learner is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .questionnaire import DIMENSIONS, QUESTIONS, declared_scores, label, top_dimension


@dataclass
class Profile:
    scores: dict[str, float]
    responses: dict[str, int] = field(default_factory=dict)
    completed_at: str = ""
    version: int = 1

    @property
    def dominant(self) -> str:
        return top_dimension(self.scores)

    @property
    def secondary(self) -> str | None:
        """Second highest, if it is close enough to be worth mentioning."""
        ordered = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        if len(ordered) < 2:
            return None
        return ordered[1][0] if ordered[1][1] >= 0.5 else None

    @property
    def tied(self) -> bool:
        """True when the top two dimensions are level — the honest default case.

        Eight items cannot separate four dimensions finely, and an all-equal
        answer set is common. Pretending one of them "leads" would put a false
        claim at the top of the page and pick a report shape for no reason.
        """
        ordered = sorted(self.scores.values(), reverse=True)
        return len(ordered) > 1 and abs(ordered[0] - ordered[1]) < 0.01

    @property
    def label(self) -> str:
        return label(self.dominant)

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda item: item[1], reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": {k: round(float(v), 3) for k, v in self.scores.items()},
            "responses": self.responses,
            "completed_at": self.completed_at,
            "version": self.version,
            "dominant": self.dominant,
            "secondary": self.secondary,
            "tied": self.tied,
            "label": self.label,
        }

    @classmethod
    def from_responses(cls, responses: dict[str, int]) -> "Profile":
        scores = declared_scores(responses)
        return cls(
            scores=scores,
            responses={k: int(v) for k, v in responses.items()},
            completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Profile":
        """Rebuild from localStorage. Never trusts what it is handed."""
        raw_scores = payload.get("scores") or {}
        scores: dict[str, float] = {}
        for dimension in DIMENSIONS:
            try:
                value = float(raw_scores.get(dimension, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            scores[dimension] = min(1.0, max(0.0, value))

        responses: dict[str, int] = {}
        for question in QUESTIONS:
            value = (payload.get("responses") or {}).get(question.id)
            try:
                responses[question.id] = min(5, max(1, int(value)))
            except (TypeError, ValueError):
                continue

        return cls(
            scores=scores,
            responses=responses,
            completed_at=str(payload.get("completed_at") or ""),
            version=int(payload.get("version") or 1),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, blob: str) -> "Profile":
        return cls.from_dict(json.loads(blob))


def score(payload: Any) -> Profile:
    """Score a questionnaire submission, rejecting anything incomplete.

    Raises ``ValueError`` with a message the UI can show, rather than silently
    inventing a zero profile — a wrong profile would quietly produce the wrong
    report, which is the worst failure mode this app has.
    """
    if not isinstance(payload, dict):
        raise ValueError("Expected an object of question id -> 1..5.")

    responses: dict[str, int] = {}
    missing: list[str] = []
    for question in QUESTIONS:
        raw = payload.get(question.id)
        if raw is None:
            missing.append(question.id)
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{question.id}: expected an integer 1-5, got {raw!r}.") from None
        if not 1 <= value <= 5:
            raise ValueError(f"{question.id}: expected 1-5, got {value}.")
        responses[question.id] = value

    if missing:
        raise ValueError(f"Missing answers for: {', '.join(missing)}.")

    return Profile.from_responses(responses)


def describe(profile: Profile) -> str:
    """One neutral sentence the UI can show next to the result."""
    ranked = profile.ranked()
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    if profile.tied:
        text = (
            f"Declared profile: no single preference stands out "
            f"({label(top[0])} and {label(second[0])} are both at {top[1]:.2f}/1.00)."
            if second
            else f"Declared profile: {label(top[0])} at {top[1]:.2f}/1.00."
        )
        text += " Pick whichever report shape you want to read."
    else:
        text = (
            f"Declared profile: {label(top[0])} leads at {top[1]:.2f}/1.00"
            + (f", then {label(second[0])} at {second[1]:.2f}." if second else ".")
        )
    return text + " This is a starting hypothesis that the reports are tuned to, not a diagnosis."


STYLE_INTENT: dict[str, str] = {
    "V": "diagrams, concept maps, comparison tables and short captions; the prose explains the pictures",
    "A": "a spoken-style explanation the student can read aloud; dialogue, signposting, mnemonics; no tables or diagrams",
    "R": "a structured written outline: numbered headings, precise definitions, dense paragraphs, a glossary",
    "K": "worked steps, hands-on exercises and self-checks; theory arrives only when a step needs it",
}
