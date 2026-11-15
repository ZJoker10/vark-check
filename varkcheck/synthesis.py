"""Is the report a summary, or the source wearing a new font?

A summariser that pastes source sentences back at the learner is the single most
likely failure of this app, and a prompt cannot prevent it. This module measures
it, so the orchestrator can reject a report and ask for a real synthesis.

How it works
------------
For each substantive sentence in the report, compute the fraction of its 6-word
shingles that also occur somewhere in the source text.

* A **paraphrase** keeps the ideas and changes the words, so its shingles are
  mostly new -> low containment.
* A **copy** keeps whole phrases, so nearly every shingle already exists in the
  source -> high containment.

Using shingles rather than edit distance means this is O(n) in the source size,
which matters: a 600 kB upload would make pairwise sentence comparison take
minutes.

The threshold is deliberately forgiving. Technical text legitimately reuses
terminology ("learning rate", "chain rule"), and flagging every shared phrase
would punish correct domain vocabulary. We only care about *runs* of six words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SHINGLE = 6
MIN_WORDS = 8
COPY_THRESHOLD = 0.55     # >= this share of shingles found in source -> copied
MAX_COPIED_RATIO = 0.30   # reject a report where >30% of sentences are copied

_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NOISE = re.compile(r"^(source\s*:|<!--|```|\||\-{3,})", re.IGNORECASE)


@dataclass
class Verdict:
    ok: bool
    copied_ratio: float
    sentences: int
    offenders: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "copied_ratio": round(self.copied_ratio, 3),
            "sentences": self.sentences,
            "offenders": self.offenders[:5],
            "reason": self.reason,
        }


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def shingles(tokens: list[str], size: int = SHINGLE) -> set[str]:
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def sentences(text: str) -> list[str]:
    """Paragraph-aware sentence split that skips markup lines."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or _NOISE.match(block):
            continue
        if block.startswith(("#", ">", "*", "-", "1.", "2.", "3.")) and len(block) < 90:
            continue  # headings and bullets are not prose to be judged
        for candidate in _SENTENCE.split(block.replace("\n", " ")):
            candidate = candidate.strip(" #*-")
            if len(_tokens(candidate)) >= MIN_WORDS:
                out.append(candidate)
    return out


class SourceFingerprint:
    """Pre-computed shingle set for the uploaded material."""

    def __init__(self, text: str):
        self._shingles = shingles(_tokens(text))

    def containment(self, sentence: str) -> float:
        grams = shingles(_tokens(sentence))
        if not grams:
            return 0.0
        return sum(1 for gram in grams if gram in self._shingles) / len(grams)


def check(report_text: str, source_text: str, *, max_ratio: float = MAX_COPIED_RATIO) -> Verdict:
    """Return whether the report synthesises rather than copies."""
    fingerprint = SourceFingerprint(source_text)
    candidates = sentences(report_text)
    if not candidates:
        return Verdict(ok=True, copied_ratio=0.0, sentences=0, reason="no prose to judge")

    copied: list[str] = []
    for sentence in candidates:
        if float(fingerprint.containment(sentence)) >= COPY_THRESHOLD:
            copied.append(sentence)

    ratio = len(copied) / len(candidates)
    ok = ratio <= max_ratio

    # Longest offenders first: those are the ones worth quoting back at the model.
    copied.sort(key=len, reverse=True)
    reason = (
        "Report reads as synthesis."
        if ok
        else (
            f"{len(copied)} of {len(candidates)} sentences reuse source wording verbatim "
            f"({ratio:.0%}, limit {max_ratio:.0%})."
        )
    )
    return Verdict(ok=ok, copied_ratio=ratio, sentences=len(candidates), offenders=copied, reason=reason)


def retry_instruction(verdict: Verdict) -> str:
    """Feedback appended to the prompt on a rejected attempt."""
    examples = "\n".join(f'- Your sentence: "{sentence[:160]}"' for sentence in verdict.offenders[:3])
    return (
        "Your previous report was rejected: it reused the source's wording instead of summarising it.\n"
        f"{verdict.reason}\n\n"
        f"{examples}\n\n"
        "Rewrite those passages. Keep the factual claims and their citations, but express them in "
        "your own sentences. Merge points from several passages into one explanation, add the "
        "'why it matters' connection the source leaves implicit, and shorten. Do not replace one "
        "copied sentence with another."
    )


def compression(report_text: str, source_text: str) -> dict[str, object]:
    """Informational: how much shorter the report is than the material."""
    source_words = len(_tokens(source_text)) or 1
    report_words = len(_tokens(report_text))
    return {
        "source_words": source_words,
        "report_words": report_words,
        "ratio": round(report_words / source_words, 3),
    }
