"""Guardrails enforced in CODE, not in the prompt.

A system prompt can be argued with. These checks cannot: they run on every
student message before the model is called, and on every model reply before the
student sees it. Each check returns a verdict with a safe replacement, so the
UI always has something to show.

Three output rules from the brief:
  1. no answer-key leak        -> answers live in SQLite only
  2. no ungrounded content     -> a reply that generates course content must
                                  cite a passage that was actually retrieved
  3. no learner labelling      -> never "you are a visual learner"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Hand-written, safe replies. Keeping them here means the fallback text is
# reviewed once, in one place, and never improvised by the model.
# --------------------------------------------------------------------------- #
ANSWER_REQUEST_REPLY = (
    "I won't hand over the answers or do the exercise for you — that is precisely "
    "the part that builds understanding, and it is also the part I am measuring.\n\n"
    "Here is how to get unstuck:\n"
    "1. Say out loud what the question is asking you to find.\n"
    "2. Point at the exact step where you got unsure.\n"
    "3. Ask me for a hint about *that step*, and I will point you to the relevant "
    "section of the course material."
)

HANDOVER_REPLY = (
    "I've flagged this for your lecturer. I could not take this further safely on my "
    "own, so a human should look at it — that is a feature, not a failure."
)

UNGROUNDED_REPLY = (
    "I could not find this in the course material, so I am not going to invent an "
    "answer. I've noted the gap for your lecturer.\n\n"
    "Try rephrasing with a term that appears in the lecture notes, or ask about a "
    "different concept from the same lecture."
)

NEUTRAL_REWRITE_REPLY = (
    "Preference is a starting hypothesis, not a verdict. Let the score decide: try a "
    "different format on the same topic and compare your results."
)

# --------------------------------------------------------------------------- #
# Input patterns: "give me the answers", "do my exercise", ...
# --------------------------------------------------------------------------- #
_ANSWER_REQUEST_PATTERNS = [
    r"\b(give|show|tell|send|reveal|share|paste)\s+(me\s+)?(the\s+|your\s+)?(correct\s+|right\s+)?"
    r"(answers?|answer\s*keys?|solutions?|solution\s*key)\b",
    r"\banswer\s*key\b",
    r"\bwhat(?:'s|\s+is)\s+the\s+(correct\s+)?answer\s+(to|for)\b",
    r"\bjust\s+(tell|give)\s+me\s+(the\s+)?(answer|solution)\b",
    r"\bwhich\s+(option|one|answer)\s+is\s+(correct|right)\b",
    r"\bdo\s+(my|the|this)\s+(exercise|homework|assignment|problem|quiz|task)\b",
    r"\b(solve|complete|finish|write)\s+(it|this|that|the\s+(problem|exercise|assignment))\s+for\s+me\b",
    r"\bfor\s+me\s+(please\s+)?(solve|answer)\b",
    # Italian — the challenge is run in Italy, students will type Italian.
    r"\b(dammi|dimmi|mostrami|mandami|passami)\s+(le\s+|la\s+)?(risposte?|soluzion[ei])\b",
    r"\bsoluzion[ei]\s+(del|della|dei)\s+(quiz|esercizio|compito)\b",
    r"\b(fai|svolgi|risolvi)\s+(tu\s+)?(il\s+mio\s+)?(esercizio|compito|quiz)\b",
    r"\bqual[e']?\s+(è|e')\s+la\s+risposta\s+(giusta|corretta)\b",
]

# --------------------------------------------------------------------------- #
# Output patterns: learner labelling
# --------------------------------------------------------------------------- #
_LABELLING_PATTERNS = [
    r"\byou\s+(are|'re)\s+(a|an)\s+[\w\s/-]{0,20}?(visual|aural|auditory|read/write|reading/writing|"
    r"kinesthetic|kinaesthetic|tactile)\s+(learner|type|person)\b",
    r"\byour\s+learning\s+(style|type)\s+is\b",
    r"\byou\s+learn\s+(best|only)\s+(by|through|with)\s+(seeing|hearing|reading|doing)\b",
    r"\bsince\s+you\s+are\s+(a|an)\s+[\w\s/-]{0,20}?learner\b",
    r"\b(visual|aural|auditory|read/write|kinesthetic|kinaesthetic)\s+learners?\s+"
    r"(should|must|can\s+only|only)\b",
    r"\bas\s+a\s+(visual|aural|auditory|read/write|kinesthetic|kinaesthetic)\s+learner\b",
]

_CITATION_RE = re.compile(r"source\s*:\s*([^\]\)\n,;]+)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class GuardVerdict:
    allowed: bool
    kind: str | None = None  # "answer_request" | "labelling" | "answer_key_leak" | "missing_citation" | "fabricated_citation"
    detail: str = ""
    safe_reply: str = ""

    @classmethod
    def ok(cls) -> "GuardVerdict":
        return cls(allowed=True)

    @classmethod
    def block(cls, kind: str, detail: str, safe_reply: str) -> "GuardVerdict":
        return cls(allowed=False, kind=kind, detail=detail, safe_reply=safe_reply)

    def to_dict(self) -> dict[str, object]:
        return {"allowed": self.allowed, "kind": self.kind, "detail": self.detail}


# --------------------------------------------------------------------------- #
# Public checks
# --------------------------------------------------------------------------- #
def check_student_input(text: str) -> GuardVerdict:
    """Runs BEFORE any model call. Costs zero tokens."""
    if not text or not text.strip():
        return GuardVerdict.ok()
    for pattern in _ANSWER_REQUEST_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return GuardVerdict.block(
                "answer_request",
                f"matched /{pattern}/ on {match.group(0)!r}",
                ANSWER_REQUEST_REPLY,
            )
    return GuardVerdict.ok()


def check_agent_output(
    text: str,
    *,
    answer_keys: list[str] | None = None,
    require_citation: bool = False,
    allowed_docs: list[str] | None = None,
) -> GuardVerdict:
    """Runs on every model reply before it reaches the student."""
    if not text or not text.strip():
        return GuardVerdict.block("empty", "empty reply", UNGROUNDED_REPLY)

    # 1. learner labelling
    for pattern in _LABELLING_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return GuardVerdict.block(
                "labelling",
                f"matched /{pattern}/ on {match.group(0)!r}",
                NEUTRAL_REWRITE_REPLY,
            )

    # 2. answer-key leak
    normalised_output = _normalise(text)
    for key in answer_keys or []:
        normalised_key = _normalise(key)
        if len(normalised_key) >= 8 and normalised_key in normalised_output:
            return GuardVerdict.block(
                "answer_key_leak",
                "reply contains a stored answer key",
                "I can't show you the answer key before you have submitted an answer. "
                "Try the question first, then I will explain what was right or wrong.",
            )

    # 3. citations
    if require_citation:
        citations = [citation.strip() for citation in _CITATION_RE.findall(text)]
        if not citations:
            return GuardVerdict.block(
                "missing_citation",
                "course content generated without a source",
                UNGROUNDED_REPLY,
            )
        if allowed_docs:
            known = tuple(allowed_docs)
            for citation in citations:
                if not any(doc in citation for doc in known):
                    return GuardVerdict.block(
                        "fabricated_citation",
                        f"citation {citation!r} is not one of the retrieved passages",
                        UNGROUNDED_REPLY,
                    )

    return GuardVerdict.ok()


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_citations(text: str) -> list[str]:
    """Every `source: ...` marker in a reply, in order."""
    return [citation.strip() for citation in _CITATION_RE.findall(text)]


def leak_score(text: str, answer_key: str) -> bool:
    """Convenience wrapper used by the tests to assert the leak check works."""
    verdict = check_agent_output(text, answer_keys=[answer_key])
    return not verdict.allowed and verdict.kind == "answer_key_leak"
