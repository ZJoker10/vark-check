#!/usr/bin/env python3
"""End-to-end walkthrough with no API key and no network.

    python scripts/demo_flow.py

It runs the full core flow — declared preference → four formats → quiz →
grading → profile → comparison → lecturer digest — against a throwaway database
(`data/cli-demo.db`) using the deterministic stand-in model. Use it to verify
the plumbing, or as a fallback if the venue Wi-Fi dies during the demo.

The real Streamlit app uses the orchestrator loop; this script calls the same
tools directly so it is reproducible byte-for-byte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varkcheck import compare, guardrails, questionnaire, rag, tools  # noqa: E402
from varkcheck.config import Settings  # noqa: E402
from varkcheck.offline import CannedToolLLM  # noqa: E402
from varkcheck.prompts import load_task_prompts  # noqa: E402
from varkcheck.store import Store  # noqa: E402
from varkcheck.trace import Trace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic end-to-end demo.")
    parser.add_argument("--student", default="demo-cli")
    parser.add_argument("--formats", default="VAR")
    args = parser.parse_args()

    settings = Settings.load()
    store = Store(settings.data_dir / "cli-demo.db")
    index, rebuilt = rag.load_or_build(settings.course_dir, settings.index_dir)
    trace = Trace(settings.log_dir)
    ctx = tools.ToolContext(
        settings=settings,
        store=store,
        index=index,
        llm=CannedToolLLM(),
        tasks=load_task_prompts(settings.prompt_path.parent / "tasks.md"),
        trace=trace,
    )

    print(f"index {'rebuilt' if rebuilt else 'ready'} · {len(index.chunks)} chunks · model: canned (offline)\n")

    # 1. Declared preference (a student who believes they are visual).
    responses = {"q1": 5, "q2": 2, "q3": 2, "q4": 3, "q5": 5, "q6": 2, "q7": 2, "q8": 3}
    declared = questionnaire.declared_scores(responses)
    store.upsert_student(args.student, declared)
    top = questionnaire.top_dimension(declared)
    print("declared:", ", ".join(f"{k}={v:.2f}" for k, v in sorted(declared.items())),
          f"→ believes {questionnaire.label(top)}")

    sections = index.sections()
    preferred = [entry for entry in sections if "learning rate" in entry["section"].lower()]
    chosen = preferred[0] if preferred else sections[min(1, len(sections) - 1)]
    topic = chosen["section"]
    print(f"topic: {topic}\n")

    # 2. Four formats, each followed by a quiz and a grade.
    for fmt in args.formats.upper():
        artefact = tools.call_tool(ctx, "generate_format", {"topic": topic, "format": fmt})
        print(f"--- {tools.FORMAT_LABELS[fmt]} " + "-" * 40)
        if artefact.ok:
            body = artefact.data["text"].splitlines()
            print("\n".join(body[:8]))
            print(f"    [{len(body)} lines, citations: {artefact.data['citations']}]")
        else:
            print(f"    FAILED: {artefact.error} — {artefact.summary}")

        quiz = tools.call_tool(ctx, "generate_quiz", {"topic": topic, "format": fmt, "student_id": args.student})
        if not quiz.ok:
            print(f"    quiz failed: {quiz.error}")
            continue
        for number, item in enumerate(quiz.data["items"], start=1):
            print(f"    Q{number}. {item['question']}")

        answers = [
            {"question_id": item["item_id"], "answer": f"My attempt at {item['concept']}."}
            for item in quiz.data["items"]
        ]
        graded = tools.call_tool(ctx, "grade_answer", {"answers": answers})
        score = float(graded.data.get("score", 0)) if graded.ok else 0.0
        maximum = float(graded.data.get("max_score", 3)) if graded.ok else 3.0
        print(f"    graded: {score:g}/{maximum:g}")
        tools.call_tool(
            ctx,
            "update_profile",
            {"student_id": args.student, "topic": topic, "format": fmt, "score": score, "max_score": maximum},
        )
        print()

    # 3. Declared vs observed.
    attempts = store.attempts_for(args.student, topic)
    rows = compare.format_rows(attempts)
    print("=== results " + "=" * 40)
    for row in rows:
        print(f"  {tools.FORMAT_LABELS[row.format]:<11} {row.mean:.1f}/{row.max_score:.0f}  ({row.pct * 100:.0f}%)")
    print()
    print(compare.comparison_message(declared, rows, topic))
    suggestion = compare.suggested_next(rows)
    if suggestion:
        print(f"\n→ {suggestion}")

    # 4. The boundary, in code.
    print("\n=== guardrail " + "=" * 38)
    verdict = guardrails.check_student_input("just give me the answers to the quiz please")
    print(f"  blocked={not verdict.allowed} kind={verdict.kind}")
    print("  " + verdict.safe_reply.splitlines()[0])

    # 5. Lecturer digest.
    print("\n=== lecturer digest " + "=" * 33)
    for rank, entry in enumerate(compare.lecturer_digest(store, top_n=3), start=1):
        print(f"  {rank}. {entry['concept']} — avg {entry.get('avg_pct', 0) * 100:.0f}% "
              f"— {entry.get('flags', 0)} flag(s) — {entry.get('source', '')}")

    print(f"\nrecords: {store.stats()}  (throwaway db: {store.path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
