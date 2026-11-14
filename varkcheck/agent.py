"""The orchestrator: an explicit plan → act → observe loop.

One turn of the loop is:

    model  ->  {"thought", "action", "args"}     (plan)
    code   ->  executes the tool                (act)
    code   ->  "OBSERVATION: ..." appended      (observe)

The loop stops when the model answers with `action: "final"`, when the step
budget is spent, or when the token budget is spent. In the last two cases the
case is handed to a human and the student is told so — the brief's "know your
limits" requirement, implemented as a control-flow branch rather than a hope.

Two rules keep the token bill sane:
  * tool observations are capped at 600 characters (tools.py);
  * artefacts travel in `ToolResult.data`, which the model never sees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import guardrails, rag, tools
from .config import Settings
from .llm import LLMError, extract_json
from .prompts import load_system_prompt, load_task_prompts
from .store import Store
from .trace import Trace

FINAL_ACTIONS = {"final", "answer", "done", "finish"}


@dataclass
class AgentStep:
    n: int
    thought: str
    action: str
    args: dict[str, Any]
    ok: bool
    observation: str
    latency_ms: float
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "thought": self.thought,
            "action": self.action,
            "args": self.args,
            "ok": self.ok,
            "observation": self.observation,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
        }


@dataclass
class AgentResult:
    message: str
    steps: list[AgentStep] = field(default_factory=list)
    tokens: int = 0
    handover: bool = False
    blocked: bool = False
    blocked_kind: str | None = None
    stopped_reason: str = "final"
    data: dict[str, Any] = field(default_factory=dict)
    fell_back: bool = False

    @property
    def called(self) -> list[str]:
        return [step.action for step in self.steps]


class Agent:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        index: rag.Index,
        llm: Any,
        tool_llm: Any | None = None,
        trace: Trace | None = None,
    ):
        self.settings = settings
        self.store = store
        self.index = index
        self.llm = llm
        self.trace = trace or Trace(settings.log_dir)
        self.system_prompt, self.prompt_version = load_system_prompt(settings.prompt_path)
        tasks_path = settings.prompt_path.parent / "tasks.md"
        self.tasks = load_task_prompts(tasks_path) if tasks_path.exists() else {}
        self.ctx = tools.ToolContext(
            settings=settings,
            store=store,
            index=index,
            llm=tool_llm or llm,
            tasks=self.tasks,
            trace=self.trace,
        )

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #
    def run(
        self,
        student_id: str,
        request: str,
        *,
        session_id: str | None = None,
        topic: str | None = None,
        require_citation: bool = False,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        # 1. Deterministic, zero-token input check. Runs before any model call.
        input_verdict = guardrails.check_student_input(request)
        if not input_verdict.allowed:
            self.trace.log("guardrail.input_blocked", kind=input_verdict.kind, detail=input_verdict.detail)
            message = input_verdict.safe_reply
            pointer = self._pointer_for(request)
            if pointer:
                message += f"\n\nRelevant section: `{pointer}`."
            return AgentResult(
                message=message,
                blocked=True,
                blocked_kind=input_verdict.kind,
                stopped_reason="input_guardrail",
            )

        messages = self._initial_messages(student_id, request, session_id=session_id, context=context)
        steps: list[AgentStep] = []
        collected: dict[str, Any] = {}
        tokens_used = 0
        parse_failures = 0

        for step_number in range(1, self.settings.max_steps + 1):
            if tokens_used >= self.settings.token_budget:
                return self._handover(
                    student_id, topic or request, steps, collected,
                    reason="token_budget", tokens=tokens_used,
                )

            try:
                reply = self.llm.chat(messages, max_tokens=self.settings.max_output_tokens)
            except LLMError as exc:
                self.trace.log("agent.llm_error", step=step_number, error=str(exc))
                return self._handover(
                    student_id, topic or request, steps, collected,
                    reason="llm_error", tokens=tokens_used, note=str(exc),
                )

            tokens_used += reply.usage.total

            try:
                action = extract_json(reply.text)
            except LLMError:
                parse_failures += 1
                self.trace.log("agent.bad_json", step=step_number, raw=reply.text[:200])
                if parse_failures >= 2:
                    return self._handover(
                        student_id, topic or request, steps, collected,
                        reason="protocol_error", tokens=tokens_used,
                    )
                messages.append({"role": "assistant", "content": reply.text[:400]})
                messages.append(
                    {
                        "role": "user",
                        "content": "INVALID: reply with exactly one JSON object "
                        '{"thought", "action", "args", "message"}. No prose, no code fences.',
                    }
                )
                continue

            name = str(action.get("action", "")).strip()
            thought = str(action.get("thought", "")).strip()[:300]

            if name.lower() in FINAL_ACTIONS:
                message = str(action.get("message") or "").strip()
                verdict = guardrails.check_agent_output(
                    message,
                    require_citation=require_citation,
                    allowed_docs=context.get("allowed_docs") if context else None,
                )
                if not verdict.allowed:
                    self.trace.log("guardrail.output_blocked", kind=verdict.kind, detail=verdict.detail)
                    message = verdict.safe_reply
                steps.append(
                    AgentStep(
                        n=step_number,
                        thought=thought,
                        action="final",
                        args={},
                        ok=verdict.allowed,
                        observation=message[:200],
                        latency_ms=reply.latency_ms,
                        tokens=reply.usage.total,
                    )
                )
                self.trace.log("agent.final", step=step_number, blocked=not verdict.allowed)
                return AgentResult(
                    message=message,
                    steps=steps,
                    tokens=tokens_used,
                    blocked=not verdict.allowed,
                    blocked_kind=verdict.kind,
                    stopped_reason="final",
                    data=collected,
                )

            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            result = tools.call_tool(self.ctx, name, args)

            observation = result.to_observation()
            steps.append(
                AgentStep(
                    n=step_number,
                    thought=thought,
                    action=name,
                    args=args,
                    ok=result.ok,
                    observation=observation[:300],
                    latency_ms=reply.latency_ms,
                    tokens=reply.usage.total,
                )
            )
            self.trace.log(
                "agent.step",
                step=step_number,
                thought=thought,
                tool=name,
                args=args,
                ok=result.ok,
                error=result.error,
                latency_ms=reply.latency_ms,
                tokens=reply.usage.total,
            )

            if result.ok:
                collected[name] = result.data
                collected.setdefault("citations", [])
                collected["citations"] = list(dict.fromkeys(collected["citations"] + result.citations))

            # Keep the transcript small: the action is re-serialised without
            # prose, and the observation is already capped at 600 chars.
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps({"thought": thought, "action": name, "args": args}),
                }
            )
            messages.append({"role": "user", "content": observation})

        return self._handover(
            student_id, topic or request, steps, collected,
            reason="step_budget", tokens=tokens_used,
        )

    # ------------------------------------------------------------------ #
    # High-level operations used by the UI
    # ------------------------------------------------------------------ #
    def deliver(self, student_id: str, topic: str, fmt: str, *, session_id: str | None = None) -> AgentResult:
        """Deliver one topic in one format. Falls back to a direct tool call."""
        fmt = tools.normalise_format(fmt)
        results = self.index.search(topic, k=5, min_score=self.settings.min_similarity)
        allowed_docs = [result.chunk.doc for result in results]
        request = (
            f"Deliver the topic {topic!r} to me in the {tools.FORMAT_LABELS[fmt]} format.\n"
            "Start with search_notes, then call generate_format, then answer me with two short "
            "sentences telling me what to look at first. Do not paste the whole artefact in your "
            "message; the interface shows it separately."
        )
        result = self.run(
            student_id,
            request,
            session_id=session_id,
            topic=topic,
            context={"allowed_docs": allowed_docs},
        )
        if "generate_format" not in result.data:
            fallback = tools.call_tool(self.ctx, "generate_format", {"topic": topic, "format": fmt})
            result.data["generate_format"] = fallback.data if fallback.ok else {}
            if not fallback.ok:
                result.data.setdefault("errors", []).append(fallback.error)
            result.fell_back = True
            self.trace.log("agent.fallback", operation="deliver", topic=topic, format=fmt, ok=fallback.ok)
        return result

    def quiz(self, student_id: str, topic: str, fmt: str, *, session_id: str | None = None) -> AgentResult:
        fmt = tools.normalise_format(fmt)
        request = (
            f"Quiz me on {topic!r} after the {tools.FORMAT_LABELS[fmt]} format. "
            "Call generate_quiz, then tell me in one sentence to answer without guessing. "
            "Never include the answers."
        )
        result = self.run(student_id, request, session_id=session_id, topic=topic)
        if "generate_quiz" not in result.data:
            fallback = tools.call_tool(
                self.ctx, "generate_quiz", {"topic": topic, "format": fmt, "student_id": student_id}
            )
            result.data["generate_quiz"] = fallback.data if fallback.ok else {}
            result.fell_back = True
            self.trace.log("agent.fallback", operation="quiz", topic=topic, format=fmt, ok=fallback.ok)
        return result

    def grade(
        self,
        student_id: str,
        topic: str,
        fmt: str,
        answers: list[dict[str, str]],
        *,
        session_id: str | None = None,
    ) -> AgentResult:
        fmt = tools.normalise_format(fmt)
        payload = json.dumps(answers, ensure_ascii=False)
        request = (
            f"Grade my answers for {topic!r} ({tools.FORMAT_LABELS[fmt]} format). "
            f"Call grade_answer with exactly this list, verbatim: {payload}\n"
            f"Then call update_profile with student_id={student_id!r}, topic={topic!r}, "
            f"format={fmt!r}, score=<the total you got>, max_score=<the maximum>. "
            "Finally summarise my score in one sentence."
        )
        result = self.run(student_id, request, session_id=session_id, topic=topic)

        graded = result.data.get("grade_answer")
        if not graded or len(graded.get("results", [])) < len(answers):
            fallback = tools.call_tool(self.ctx, "grade_answer", {"answers": answers})
            result.data["grade_answer"] = fallback.data if fallback.ok else {}
            result.fell_back = True
            self.trace.log("agent.fallback", operation="grade", ok=fallback.ok)

        recorded = self.store.attempts_for(student_id, topic)
        if not any(str(row["format"]).upper() == fmt for row in recorded):
            graded = result.data.get("grade_answer") or {}
            score = float(graded.get("score", 0) or 0)
            max_score = float(graded.get("max_score", 3) or 3)
            tools.call_tool(
                self.ctx,
                "update_profile",
                {
                    "student_id": student_id,
                    "topic": topic,
                    "format": fmt,
                    "score": score,
                    "max_score": max_score,
                },
            )
            result.fell_back = True
            self.trace.log("agent.fallback", operation="update_profile", ok=True)
        return result

    def ask(self, student_id: str, question: str, *, session_id: str | None = None, topic: str | None = None) -> AgentResult:
        """Free-form question, grounded in the course material."""
        return self.run(student_id, question, session_id=session_id, topic=topic, require_citation=True)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _system_message(self) -> str:
        return (
            self.system_prompt.replace("{max_steps}", str(self.settings.max_steps))
            .replace("{token_budget}", str(self.settings.token_budget))
            .replace("{tools}", tools.tool_menu())
        )

    def _initial_messages(
        self,
        student_id: str,
        request: str,
        *,
        session_id: str | None,
        context: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._system_message()}]
        state = self._state_block(student_id, context or {})
        messages.append({"role": "user", "content": f"STATE\n{state}\n\nREQUEST\n{request}"})
        if session_id:
            for row in self.store.history(session_id, limit=4):
                content = str(row["content"])[:300]
                messages.append({"role": row["role"], "content": content})
        return messages

    def _state_block(self, student_id: str, context: dict[str, Any]) -> str:
        student = self.store.get_student(student_id) or {}
        declared = student.get("declared") or {}
        attempts = self.store.attempts_for(student_id)

        lines = [f"student_id: {student_id}"]
        if declared:
            pretty = ", ".join(f"{dim}={float(value):.2f}" for dim, value in sorted(declared.items()))
            lines.append(f"declared self-report (0-1, belief only, never a restriction): {pretty}")
        if attempts:
            recent = attempts[-6:]
            lines.append(
                "recent attempts: "
                + "; ".join(f"{row['format']} {row['score']:g}/{row['max_score']:g}" for row in recent)
            )
        if context.get("allowed_docs"):
            lines.append("citable documents: " + ", ".join(sorted(set(context["allowed_docs"]))))
        lines.append(
            "course material: "
            + ", ".join(f"{entry['doc']} ({len(entry['sections'])} sections)" for entry in self.index.sources_digest())
        )
        return "\n".join(lines)

    def _pointer_for(self, text: str) -> str | None:
        """Cheap, model-free pointer to the best matching section."""
        results = self.index.search(text, k=1, min_score=self.settings.min_similarity)
        return results[0].chunk.citation if results else None

    def _handover(
        self,
        student_id: str,
        focus: str,
        steps: list[AgentStep],
        collected: dict[str, Any],
        *,
        reason: str,
        tokens: int,
        note: str = "",
    ) -> AgentResult:
        concept = (focus or "unresolved question").strip()[:120]
        try:
            self.store.add_flag(
                concept=concept,
                reason=f"agent handover: {reason}" + (f" ({note})" if note else ""),
                student_id=student_id,
                topic=focus,
            )
        except Exception:  # noqa: BLE001 - handover must not fail
            pass
        self.trace.log("agent.handover", reason=reason, concept=concept, steps=len(steps), tokens=tokens)
        return AgentResult(
            message=guardrails.HANDOVER_REPLY,
            steps=steps,
            tokens=tokens,
            handover=True,
            stopped_reason=reason,
            data=collected,
        )
