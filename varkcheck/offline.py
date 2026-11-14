"""Offline helpers: a deterministic stand-in for the tool LLM.

Used by the test suite and by `scripts/demo_flow.py`, so a reviewer can see the
whole pipeline (retrieval → artefact → quiz → grading → profile → digest) with
no API key, no network and no cost.

It is NOT a model. It produces well-formed, citation-correct output so the
plumbing can be verified; the quality of the prose is irrelevant here.
"""

from __future__ import annotations

import json
import re

from .llm import LLMReply, Usage

_CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]")


class CannedToolLLM:
    def __init__(self, default_score: int = 2):
        self.default_score = default_score
        self.calls: list[str] = []

    def chat(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> LLMReply:
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        citations = _CITATION_RE.findall(prompt) or ["course.md §Unknown"]
        primary = citations[0]

        # --- grading ---------------------------------------------------- #
        if "EXPECTED ANSWER" in prompt:
            return LLMReply(
                text=json.dumps(
                    {
                        "score": self.default_score,
                        "why": f"You covered the main idea; re-read {primary} to tighten the wording.",
                    }
                ),
                usage=Usage(120, 40),
                model="canned",
            )

        # --- quiz -------------------------------------------------------- #
        if "answer_key" in prompt or '"items"' in prompt:
            items = []
            for index, citation in enumerate(citations[:3], start=1):
                section = citation.split("§")[-1].strip() if "§" in citation else citation
                items.append(
                    {
                        "question": f"In your own words, explain the key mechanism of {section} "
                        f"(question {index}).",
                        "answer_key": f"A correct answer names the mechanism described in {citation}.",
                        "concept": section[:40],
                        "source": citation,
                        "rubric": "Names the mechanism and one condition under which it applies.",
                    }
                )
            while len(items) < 3:
                items.append(dict(items[-1]))
            return LLMReply(
                text=json.dumps({"items": items}),
                usage=Usage(300, 180),
                model="canned",
            )

        # --- format rendering ------------------------------------------- #
        match = re.search(r"FORMAT:\s*([A-Za-z/]+)", prompt)
        fmt = (match.group(1) if match else "Visual").strip().lower()
        second = citations[min(1, len(citations) - 1)]

        if fmt.startswith("visual"):
            body = (
                "```mermaid\nflowchart TD\n"
                "    A[Topic] --> B[Key idea]\n"
                "    B --> C[Consequence]\n"
                "    B --> D[Condition to check]\n"
                "```\n\n"
                f"This diagram traces the argument in {primary}: the key idea sits in the middle, "
                "and both consequences point back to it.\n\n"
                f"- Read the middle node first ({primary})\n"
                f"- Then the consequence branch ({second})"
            )
        elif fmt.startswith("aural"):
            body = (
                f"Picture the idea in {primary} as a sentence you would say out loud. "
                "First, name what the thing does. Then say what happens when it is set too high "
                "or too low. That second sentence is the one examiners look for. "
                f"If you can say it in your own words, you own it — and {second} is the detail to add next."
            )
        elif fmt.startswith("read"):
            body = (
                f"## 1. What the section says\n{primary}\n\n"
                "## 2. Key terms\n- **mechanism** — the step that produces the effect\n"
                "- **condition** — when the mechanism applies\n\n"
                f"## 3. Micro-example\nApply the mechanism once with a small number, then check it against {second}."
            )
        else:
            body = (
                f"## Exercise on {primary}\n"
                "1. Write the definition in one sentence, from memory.\n"
                "2. Draw the two failure cases side by side.\n"
                "3. Predict what changes if the key parameter doubles; write the prediction down.\n"
                "4. Check your prediction against the section.\n\n"
                f"Self-check: can you state the mechanism without looking at {second}?"
            )

        return LLMReply(
            text=f"{body}\n\nsource: {primary}",
            usage=Usage(400, 160),
            model="canned",
        )
