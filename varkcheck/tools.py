"""The agent's tools.

A tool is a typed function with a name and a description. The model chooses
*when* to call one; this module decides *what it is allowed to do*. Tools return
a short `summary` (the observation the model sees, hard-capped at 600 chars) and
a richer `data` payload that only the application reads. That split is the main
token-control mechanism in the whole prototype: the model never accumulates
artefacts in its context.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

from . import compare, guardrails, rag
from .config import Settings
from .llm import LLMError, extract_json
from .store import Store
from .trace import Trace

FORMATS = ("V", "A", "R", "K")
FORMAT_LABELS = {"V": "Visual", "A": "Aural", "R": "Read/write", "K": "Kinesthetic"}
OBSERVATION_LIMIT = 600


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
@dataclass
class ToolContext:
    settings: Settings
    store: Store
    index: rag.Index
    llm: Any  # ChatClient or MockLLM (duck-typed: .chat(messages, **kwargs))
    tasks: dict[str, str] = field(default_factory=dict)
    trace: Trace | None = None


@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    error: str | None = None

    def to_observation(self) -> str:
        if self.ok:
            return f"OK: {self.summary}"
        return f"ERROR ({self.error}): {self.summary}"


def _clip(text: str, limit: int = OBSERVATION_LIMIT) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def normalise_format(value: str) -> str:
    fmt = str(value or "").strip().upper()
    if fmt in FORMATS:
        return fmt
    for key, label in FORMAT_LABELS.items():
        if label.lower() == str(value).strip().lower():
            return key
    raise ValueError(f"format must be one of V, A, R, K (got {value!r})")


def _cache_key(kind: str, topic: str, fmt: str, material_hash: str) -> str:
    payload = f"{kind}|{topic.strip().lower()}|{fmt}|{material_hash}|v1"
    return hashlib.sha256(payload.encode()).hexdigest()


def _allowed_docs(results: list[rag.Retrieved]) -> list[str]:
    return [result.chunk.doc for result in results]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def search_notes(ctx: ToolContext, query: str, k: int | None = None) -> ToolResult:
    """Retrieve passages from the real course material, with citations."""
    if not str(query or "").strip():
        return ToolResult(False, "query must not be empty", error="bad_arguments")

    top_k = int(k or ctx.settings.top_k)
    results = ctx.index.search(str(query), k=top_k, min_score=ctx.settings.min_similarity)
    if not results:
        return ToolResult(
            False,
            f"No passage in the course material matches {query!r}. The material does not cover it.",
            error="no_material",
        )

    lines = []
    for result in results[:3]:
        snippet = re.sub(r"\s+", " ", result.chunk.text)[:170]
        lines.append(f"[{result.chunk.citation}] {snippet}")
    summary = "\n".join(lines)

    if ctx.trace:
        ctx.trace.log(
            "tool.search_notes",
            query=query,
            hits=len(results),
            citations=[result.chunk.citation for result in results],
        )
    return ToolResult(
        True,
        _clip(summary),
        {"passages": [result.to_dict() for result in results]},
        [result.chunk.citation for result in results],
    )


def generate_format(ctx: ToolContext, topic: str, format: str) -> ToolResult:  # noqa: A002 - tool arg name
    """Produce the topic in V, A, R or K, grounded in retrieved passages."""
    try:
        fmt = normalise_format(format)
    except ValueError as exc:
        return ToolResult(False, str(exc), error="bad_arguments")

    results = ctx.index.search(topic, k=max(ctx.settings.top_k, 5), min_score=ctx.settings.min_similarity)
    if not results:
        return ToolResult(
            False,
            f"The course material does not cover {topic!r}. Say so instead of inventing content.",
            error="no_material",
        )

    cache_key = _cache_key("format", topic, fmt, ctx.index.material_hash)
    cached = ctx.store.cache_get(cache_key)
    if cached:
        if ctx.trace:
            ctx.trace.log("tool.generate_format.cache_hit", topic=topic, format=fmt)
        return ToolResult(True, cached["summary"], cached, cached.get("citations", []))

    task = ctx.tasks.get(f"FORMAT_{fmt}", "Render the topic in the requested format.")
    passages = rag.format_passages(results)
    system = (
        "You are a course-content renderer. Use ONLY the passages provided. "
        "If something is not in the passages, omit it. Never add outside knowledge. "
        "Never give quiz answers or solutions."
    )
    base_prompt = (
        f"TOPIC: {topic}\n"
        f"FORMAT: {FORMAT_LABELS[fmt]}\n\n"
        f"PASSAGES:\n{passages}\n\n"
        f"TASK:\n{task}\n\n"
        "Finish with one `source: <file> §<section>` line per passage you used."
    )

    text = ""
    verdict = guardrails.GuardVerdict.ok()
    for attempt in range(2):
        prompt = base_prompt if attempt == 0 else (
            base_prompt + "\n\nIMPORTANT: your previous answer was rejected because it was not "
            "grounded. Repeat the task and include at least one `source:` line."
        )
        try:
            text = ctx.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=900,
            ).text.strip()
        except LLMError as exc:
            return ToolResult(False, f"Model unavailable: {exc}", error="llm_error")

        verdict = guardrails.check_agent_output(
            text,
            require_citation=True,
            allowed_docs=_allowed_docs(results),
        )
        if verdict.allowed:
            break
        if verdict.kind != "missing_citation":
            break  # retrying will not fix a fabrication or a label

    if not verdict.allowed:
        if ctx.trace:
            ctx.trace.log("tool.generate_format.blocked", topic=topic, format=fmt, guard=verdict.to_dict())
        return ToolResult(False, f"Output rejected by guardrail ({verdict.kind}).", error=verdict.kind or "guardrail")

    diagram_ok = True
    if fmt == "V":
        diagram_ok = bool(re.search(r"```mermaid", text)) and bool(re.search(r"(graph|flowchart)\s+(TD|LR|TB)", text))
        if not diagram_ok:
            text += (
                "\n\n> The diagram could not be rendered reliably. The bullet points above are the "
                "authoritative version; ask for the Read/write format for the same content in prose."
            )

    payload = {
        "topic": topic,
        "format": fmt,
        "text": text,
        "citations": guardrails.extract_citations(text),
        "diagram_ok": diagram_ok,
        "summary": _clip(
            f"Generated the {FORMAT_LABELS[fmt]} version of {topic!r} "
            f"({len(text)} chars, {len(guardrails.extract_citations(text))} citation(s))."
        ),
    }
    ctx.store.cache_set(cache_key, payload)
    if ctx.trace:
        ctx.trace.log("tool.generate_format", topic=topic, format=fmt, chars=len(text), diagram_ok=diagram_ok)
    return ToolResult(True, payload["summary"], payload, payload["citations"])


def generate_quiz(ctx: ToolContext, topic: str, format: str, student_id: str = "anon") -> ToolResult:  # noqa: A002
    """Create 3 grounded questions. Answer keys never leave the database."""
    try:
        fmt = normalise_format(format)
    except ValueError as exc:
        return ToolResult(False, str(exc), error="bad_arguments")

    results = ctx.index.search(topic, k=max(ctx.settings.top_k, 5), min_score=ctx.settings.min_similarity)
    if not results:
        return ToolResult(
            False,
            f"The course material does not cover {topic!r}, so no quiz can be generated.",
            error="no_material",
        )

    # Shared across students: same topic + format => same questions. That makes
    # the lecturer digest comparable and costs nothing on repeat runs.
    cache_key = _cache_key("quiz", topic, fmt, ctx.index.material_hash)
    payload = ctx.store.cache_get(cache_key)

    if payload is None:
        task = ctx.tasks.get("QUIZ", "Write three questions.")
        passages = rag.format_passages(results)
        system = (
            "You write short diagnostic quizzes grounded strictly in the supplied passages. "
            "Reply with JSON only. Never put the answer inside the question."
        )
        prompt = (
            f"TOPIC: {topic}\n"
            f"STUDENT FORMAT: {FORMAT_LABELS[fmt]}\n\n"
            f"PASSAGES:\n{passages}\n\n"
            f"TASK:\n{task}"
        )
        try:
            raw = ctx.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=800,
                json_mode=True,
            ).text
            parsed = extract_json(raw)
        except (LLMError, ValueError) as exc:
            return ToolResult(False, f"Quiz generation failed: {exc}", error="llm_error")

        docs = _allowed_docs(results)
        citations = [result.chunk.citation for result in results]
        items: list[dict[str, str]] = []
        for item in (parsed.get("items") or [])[:3]:
            question = str(item.get("question", "")).strip()
            answer_key = str(item.get("answer_key", "")).strip()
            if not question or not answer_key:
                continue
            # A question that contains its own answer is a broken item.
            if guardrails.leak_score(question, answer_key):
                continue
            source = str(item.get("source", "") or "").strip()
            if not any(doc in source for doc in docs):
                source = citations[0]
            items.append(
                {
                    "question": question,
                    "answer_key": answer_key,
                    "concept": str(item.get("concept", "") or topic).strip(),
                    "source": source,
                    "rubric": str(item.get("rubric", "") or "").strip(),
                }
            )

        if len(items) < 2:
            return ToolResult(
                False,
                "The model could not produce usable questions from this material.",
                error="quiz_rejected",
            )
        payload = {"items": items}

    item_ids: list[str] = []
    for item in payload["items"]:
        item_ids.append(
            ctx.store.add_quiz_item(
                student_id=student_id,
                topic=topic,
                fmt=fmt,
                question=item["question"],
                answer_key=item["answer_key"],
                concept=item["concept"],
                source=item["source"],
                rubric=item["rubric"],
                max_score=2,
            )
        )
    ctx.store.cache_set(cache_key, payload)

    public_items = [
        {
            "item_id": item_id,
            "question": item["question"],
            "concept": item["concept"],
            "source": item["source"],
        }
        for item_id, item in zip(item_ids, payload["items"])
    ]
    summary = " ".join(f"({index + 1}) {item['question']}" for index, item in enumerate(public_items))
    if ctx.trace:
        ctx.trace.log("tool.generate_quiz", topic=topic, format=fmt, items=len(public_items))
    return ToolResult(
        True,
        _clip(f"Created {len(public_items)} questions. Answer keys are server-side only. {summary}"),
        {"items": public_items},
        [item["source"] for item in public_items],
    )


def grade_answer(ctx: ToolContext, answers: list[dict[str, str]]) -> ToolResult:
    """Grade one or more answers in a single call (fewer tokens, one round trip)."""
    if not isinstance(answers, list) or not answers:
        return ToolResult(
            False,
            "answers must be a non-empty list of {question_id, answer}",
            error="bad_arguments",
        )

    results: list[dict[str, Any]] = []
    for entry in answers:
        item_id = str(entry.get("question_id") or entry.get("item_id") or "").strip()
        answer = str(entry.get("answer") or "").strip()
        item = ctx.store.get_quiz_item(item_id)
        if item is None:
            results.append({"item_id": item_id, "error": "unknown_question", "score": 0.0, "max_score": 2})
            continue

        max_score = float(item.get("max_score") or 2)
        if not answer:
            why = (
                "No answer submitted. Re-read the section and try again: guessing costs you nothing, "
                "skipping costs you the data point."
            )
            ctx.store.mark_graded(item_id, answer, 0.0, why)
            results.append(
                {
                    "item_id": item_id,
                    "score": 0.0,
                    "max_score": max_score,
                    "why": why,
                    "concept": item.get("concept"),
                    "source": item.get("source"),
                }
            )
            continue

        task = ctx.tasks.get("GRADE", "Grade the answer. Return JSON.")
        system = (
            "You grade one student answer against a stored key and rubric. "
            "Reply with JSON only. Do not restate the key verbatim and do not rewrite the student's work."
        )
        prompt = (
            f"QUESTION: {item['question']}\n"
            f"EXPECTED ANSWER (server-side, do not quote): {item['answer_key']}\n"
            f"RUBRIC: {item.get('rubric') or 'n/a'}\n"
            f"MAX SCORE: {int(max_score)}\n\n"
            f"STUDENT ANSWER: {answer}\n\n"
            f"TASK:\n{task}"
        )
        try:
            raw = ctx.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=260,
                json_mode=True,
            ).text
            parsed = extract_json(raw)
            score = float(parsed.get("score", 0))
            why = str(parsed.get("why", "")).strip()
        except (LLMError, ValueError, TypeError) as exc:
            results.append(
                {"item_id": item_id, "error": f"grading_failed: {exc}", "score": 0.0, "max_score": max_score}
            )
            continue

        score = max(0.0, min(max_score, score))
        verdict = guardrails.check_agent_output(why, answer_keys=[item["answer_key"]])
        if not verdict.allowed:
            why = (
                f"Feedback withheld by the answer-key guardrail. Re-read "
                f"{item.get('source') or 'the relevant section'} and compare your answer to the rubric."
            )
            if ctx.trace:
                ctx.trace.log("tool.grade_answer.blocked", item_id=item_id, guard=verdict.to_dict())

        ctx.store.mark_graded(item_id, answer, score, why)
        results.append(
            {
                "item_id": item_id,
                "score": score,
                "max_score": max_score,
                "why": why,
                "concept": item.get("concept"),
                "source": item.get("source"),
            }
        )

    total = sum(float(entry.get("score", 0)) for entry in results)
    maximum = sum(float(entry.get("max_score", 2)) for entry in results) or 1.0
    per_item = "; ".join(
        f"{entry['item_id']}: {entry.get('score', 0):g}/{entry.get('max_score', 2):g}" for entry in results
    )
    if ctx.trace:
        ctx.trace.log("tool.grade_answer", graded=len(results), score=total, max_score=maximum)
    return ToolResult(
        True,
        _clip(f"Graded {len(results)} answer(s): {total:g}/{maximum:g}. {per_item}"),
        {"results": results, "score": total, "max_score": maximum},
    )


def update_profile(
    ctx: ToolContext,
    student_id: str,
    topic: str,
    format: str,  # noqa: A002
    score: float,
    max_score: float = 3,
) -> ToolResult:
    """Write the attempt to memory and report the per-topic standing."""
    try:
        fmt = normalise_format(format)
    except ValueError as exc:
        return ToolResult(False, str(exc), error="bad_arguments")

    ctx.store.add_attempt(student_id, topic, fmt, float(score), float(max_score))
    rows = compare.format_rows(ctx.store.attempts_for(student_id, topic))
    standing = ", ".join(f"{row.format} {row.mean:.1f}/{row.max_score:.0f}" for row in rows)
    if ctx.trace:
        ctx.trace.log("tool.update_profile", student_id=student_id, topic=topic, format=fmt, score=score)
    return ToolResult(
        True,
        _clip(f"Recorded {fmt} {float(score):g}/{float(max_score):g} for {topic!r}. Topic standing: {standing}"),
        {"standing": [row.to_dict() for row in rows]},
    )


def flag_for_lecturer(
    ctx: ToolContext,
    concept: str,
    reason: str = "",
    student_id: str | None = None,
    topic: str | None = None,
) -> ToolResult:
    """Add a concept to the lecturer digest. This is the human hand-over."""
    concept = str(concept or "").strip()
    if not concept:
        return ToolResult(False, "concept must not be empty", error="bad_arguments")
    ctx.store.add_flag(concept=concept, reason=str(reason or ""), student_id=student_id, topic=topic)
    if ctx.trace:
        ctx.trace.log("tool.flag_for_lecturer", concept=concept, reason=reason, topic=topic)
    return ToolResult(
        True,
        _clip(f"Flagged {concept!r} for the lecturer digest. Reason: {reason or 'not specified'}"),
        {"concept": concept},
    )


def schedule_review(ctx: ToolContext, student_id: str, topic: str, days: int = 3) -> ToolResult:
    """Optional spaced repetition: book a review date."""
    days = max(1, int(days or 3))
    due = (date.today() + timedelta(days=days)).isoformat()
    ctx.store.add_review(student_id, topic, due)
    if ctx.trace:
        ctx.trace.log("tool.schedule_review", student_id=student_id, topic=topic, due=due)
    return ToolResult(True, f"Scheduled a review of {topic!r} on {due}.", {"due_date": due})


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
HANDLERS: dict[str, Callable[..., ToolResult]] = {
    "search_notes": search_notes,
    "generate_format": generate_format,
    "generate_quiz": generate_quiz,
    "grade_answer": grade_answer,
    "update_profile": update_profile,
    "flag_for_lecturer": flag_for_lecturer,
    "schedule_review": schedule_review,
}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_notes",
        "description": "Search the course material. Always call this before generating content on a topic.",
        "parameters": {"query": "string", "k": "integer (optional, default 4)"},
    },
    {
        "name": "generate_format",
        "description": "Render one topic in one format: V (Mermaid diagram), A (spoken script), R (structured notes), K (hands-on exercise). Grounded in retrieved passages.",
        "parameters": {"topic": "string (from search_notes)", "format": "V | A | R | K"},
    },
    {
        "name": "generate_quiz",
        "description": "Create 3 short questions on a topic. Answer keys stay server-side and are never returned.",
        "parameters": {"topic": "string", "format": "V | A | R | K", "student_id": "string"},
    },
    {
        "name": "grade_answer",
        "description": "Grade submitted answers in one batch. Returns score and why, never the answer key.",
        "parameters": {"answers": '[{"question_id": "q_...", "answer": "..."}]'},
    },
    {
        "name": "update_profile",
        "description": "Record a finished format attempt (score) in the learner profile.",
        "parameters": {"student_id": "string", "topic": "string", "format": "V|A|R|K", "score": "number", "max_score": "number"},
    },
    {
        "name": "flag_for_lecturer",
        "description": "Hand a concept to the lecturer digest when the material does not cover it, or when the student is stuck.",
        "parameters": {"concept": "string", "reason": "string", "student_id": "string (optional)", "topic": "string (optional)"},
    },
    {
        "name": "schedule_review",
        "description": "Optional: book a spaced-repetition review for this topic.",
        "parameters": {"student_id": "string", "topic": "string", "days": "integer (optional, default 3)"},
    },
]


def tool_menu() -> str:
    return "\n".join(
        f"- `{spec['name']}({', '.join(spec['parameters'])})` — {spec['description']}"
        for spec in TOOL_SPECS
    )


def call_tool(ctx: ToolContext, name: str, args: dict[str, Any] | None = None) -> ToolResult:
    """Single entry point used by the orchestrator. Never raises."""
    handler = HANDLERS.get(name)
    if handler is None:
        return ToolResult(
            False,
            f"Unknown tool {name!r}. Available: {', '.join(HANDLERS)}",
            error="unknown_tool",
        )
    try:
        return handler(ctx, **(args or {}))
    except TypeError as exc:
        return ToolResult(False, f"Bad arguments for {name}: {exc}", error="bad_arguments")
    except LLMError as exc:
        return ToolResult(False, f"{name} failed: {exc}", error="llm_error")
    except Exception as exc:  # noqa: BLE001 - a tool must never crash the session
        if ctx.trace:
            ctx.trace.log("tool.error", tool=name, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(False, f"{name} failed: {type(exc).__name__}: {exc}", error="tool_error")


def specs_json() -> str:
    return json.dumps(TOOL_SPECS, indent=2)
