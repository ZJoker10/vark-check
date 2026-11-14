"""The declared-preference questionnaire.

Eight neutral self-report items, two per VARK dimension, 1–5 Likert scale.
Written from scratch on purpose: the official VARK questionnaire is
copyrighted (vark-learn.com), and the brief forbids copying it.

The point of this questionnaire is to capture a *belief* that the rest of the
prototype will test. It is never used to restrict what a student receives.
"""

from __future__ import annotations

from dataclasses import dataclass

DIMENSIONS = ("V", "A", "R", "K")

DIMENSION_LABELS = {
    "V": "Visual",
    "A": "Aural",
    "R": "Read/write",
    "K": "Kinesthetic",
}

SCALE = {
    1: "Not like me at all",
    2: "Slightly like me",
    3: "Neutral",
    4: "Mostly like me",
    5: "Very much like me",
}


@dataclass(frozen=True)
class Question:
    id: str
    dimension: str
    text: str


QUESTIONS: list[Question] = [
    Question("q1", "V", "When I need to understand a process, a diagram or flowchart helps me see how the parts connect."),
    Question("q2", "A", "Saying an explanation out loud, or hearing someone explain it, helps it stick."),
    Question("q3", "R", "Writing my own summary, or re-reading structured notes, is how I usually organise a topic."),
    Question("q4", "K", "I understand something best once I have performed the steps or built it myself."),
    Question("q5", "V", "I remember layouts, charts and spatial arrangements better than lists of words."),
    Question("q6", "A", "I often repeat a concept aloud or explain it to someone else to check that I understand it."),
    Question("q7", "R", "I prefer precise definitions and written steps over pictures or hands-on trial and error."),
    Question("q8", "K", "Working through a concrete example by hand (or in a simulation) beats reading about it."),
]


def declared_scores(responses: dict[str, int]) -> dict[str, float]:
    """Turn item responses into a 0–1 score per dimension.

    Each dimension has two items scored 1–5, so the raw range is 2–10; we
    rescale to 0–1 to keep it comparable with quiz percentages.
    """
    totals = {dimension: 0.0 for dimension in DIMENSIONS}
    counts = {dimension: 0 for dimension in DIMENSIONS}

    for question in QUESTIONS:
        value = responses.get(question.id)
        if value is None:
            continue
        totals[question.dimension] += float(value)
        counts[question.dimension] += 1

    scores: dict[str, float] = {}
    for dimension in DIMENSIONS:
        if counts[dimension] == 0:
            scores[dimension] = 0.0
            continue
        minimum = 1.0 * counts[dimension]
        maximum = 5.0 * counts[dimension]
        scores[dimension] = round((totals[dimension] - minimum) / (maximum - minimum), 3)
    return scores


def top_dimension(scores: dict[str, float]) -> str:
    return max(DIMENSIONS, key=lambda dimension: scores.get(dimension, 0.0))


def is_complete(responses: dict[str, int]) -> bool:
    return all(question.id in responses for question in QUESTIONS)


def label(dimension: str) -> str:
    return DIMENSION_LABELS.get(dimension.upper(), dimension)


def as_row(scores: dict[str, float]) -> list[dict[str, object]]:
    """Human-readable table for the UI."""
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {"dimension": dimension, "label": label(dimension), "score": round(value, 2)}
        for dimension, value in ordered
    ]
