"""VARK Check — Streamlit interface.

Two pages, one repo:
  Student   — declare a preference, receive a topic in four formats, get quizzed
              after each one, then see declared vs observed.
  Lecturer  — the three concepts the group has not grasped, with the course
              source for each. No student labels, no grouping.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import random

import streamlit as st

from varkcheck import compare, questionnaire, rag, seed, tools
from varkcheck.agent import Agent, AgentResult
from varkcheck.config import Settings
from varkcheck.llm import ChatClient, MockLLM
from varkcheck.store import Store
from varkcheck.trace import Trace

st.set_page_config(page_title="VARK Check", page_icon="🧪", layout="wide")

FORMATS = ("V", "A", "R", "K")


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Indexing course material and warming up the agent…")
def backend():
    settings = Settings.load()
    store = Store(settings.db_path)
    index, rebuilt = rag.load_or_build(
        settings.course_dir,
        settings.index_dir,
        chunk_chars=settings.chunk_chars,
    )
    seed.ensure_seeded(store, count=settings.synthetic_students, seed_value=settings.demo_seed)
    trace = Trace(settings.log_dir)
    llm = MockLLM() if settings.provider == "mock" else ChatClient(settings, trace=trace)
    agent = Agent(settings, store, index, llm, trace=trace)
    return settings, store, index, agent, rebuilt


def init_state() -> None:
    defaults = {
        "stage": "identify",
        "student_id": "",
        "topic": None,
        "order": [],
        "cursor": 0,
        "deliveries": {},
        "quizzes": {},
        "grades": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_flow() -> None:
    for key in ("stage", "topic", "order", "cursor", "deliveries", "quizzes", "grades"):
        st.session_state.pop(key, None)
    init_state()


# --------------------------------------------------------------------------- #
# Shared widgets
# --------------------------------------------------------------------------- #
def agent_trace(result: AgentResult, title: str = "Agent trace — plan → act → observe") -> None:
    with st.expander(title):
        if not result.steps:
            st.caption("No model steps: this answer came from a guardrail.")
        else:
            st.dataframe(
                [
                    {
                        "step": step.n,
                        "plan (thought)": step.thought,
                        "act (tool)": step.action,
                        "ok": step.ok,
                        "observe (truncated)": step.observation,
                        "tokens": step.tokens,
                    }
                    for step in result.steps
                ],
                use_container_width=True,
                hide_index=True,
            )
        badge = {
            "final": "✅ completed",
            "step_budget": "⏱️ step budget spent → handed to lecturer",
            "token_budget": "🪙 token budget spent → handed to lecturer",
            "protocol_error": "🧯 model returned invalid JSON → handed to lecturer",
            "input_guardrail": "🛡️ blocked by input guardrail (no model call)",
        }.get(result.stopped_reason, result.stopped_reason)
        st.caption(f"{badge} · {result.tokens} tokens · {len(result.steps)} step(s)"
                   + (" · deterministic fallback used" if result.fell_back else ""))


def render_artifact(data: dict, fmt: str) -> None:
    text = (data or {}).get("text") or ""
    if not text:
        st.warning("No artefact was produced. The agent may have handed this to your lecturer.")
        return
    st.markdown(text)
    if fmt == "V" and not data.get("diagram_ok", True):
        st.warning(
            "The Mermaid diagram could not be validated, so the bullet points above are the "
            "authoritative version. Ask for the Read/write format for prose."
        )
    citations = data.get("citations") or []
    if citations:
        st.caption("Sources: " + " · ".join(citations))
    else:
        st.caption("No citations returned — this artefact is not trustworthy and should be flagged.")


def ask_the_agent(agent: Agent, student_id: str, topic: str | None) -> None:
    """The boundary demo: students ask for answers, the code refuses."""
    with st.expander("Ask the agent anything (try asking for the answers)"):
        question = st.text_area(
            "Your question",
            placeholder="e.g. just give me the answers, or: explain gradient descent",
            key="freeform_question",
            height=90,
        )
        if st.button("Ask", key="ask_button"):
            if not question.strip():
                st.info("Type a question first.")
            else:
                with st.spinner("Thinking…"):
                    result = agent.ask(student_id, question, session_id=st.session_state.get("session_id"), topic=topic)
                if result.blocked:
                    st.error("Guardrail held. No model call was made for this request." if result.stopped_reason == "input_guardrail" else "Guardrail replaced the reply.")
                st.markdown(result.message)
                agent_trace(result)


# --------------------------------------------------------------------------- #
# Student flow
# --------------------------------------------------------------------------- #
def stage_identify(store: Store) -> None:
    st.subheader("1 · Identify yourself")
    st.write(
        "Use a pseudonymous id. Nothing here needs your name — the prototype stores as little "
        "as the flow requires."
    )
    student_id = st.text_input("Student id", value=st.session_state.student_id, placeholder="s-1042")
    if st.button("Continue", type="primary"):
        if not student_id.strip():
            st.error("A student id is required (it can be anything, e.g. `s-1042`).")
            return
        st.session_state.student_id = student_id.strip()
        st.session_state.session_id = f"ses_{student_id.strip()}"
        student = store.upsert_student(student_id.strip())
        declared = student.get("declared") or {}
        st.session_state.stage = "choose_topic" if declared else "questionnaire"
        st.rerun()


def stage_questionnaire(store: Store) -> None:
    st.subheader("2 · How do you *think* you learn best?")
    st.caption(
        "Eight neutral statements, 1–5. This is a belief we are about to test, not a diagnosis. "
        "It will never restrict which format you receive."
    )
    with st.form("questionnaire"):
        responses: dict[str, int] = {}
        for question in questionnaire.QUESTIONS:
            responses[question.id] = st.radio(
                question.text,
                options=[1, 2, 3, 4, 5],
                format_func=lambda value: f"{value} — {questionnaire.SCALE[value]}",
                horizontal=True,
                index=2,
                key=f"qq_{question.id}",
            )
        submitted = st.form_submit_button("Save my declared preference", type="primary")

    if submitted:
        scores = questionnaire.declared_scores(responses)
        store.upsert_student(st.session_state.student_id, scores)
        st.session_state.stage = "choose_topic"
        st.rerun()


def stage_choose_topic(store: Store, index: rag.Index) -> None:
    st.subheader("3 · Pick a topic from the real course material")
    sections = index.sections()
    if not sections:
        st.error("No course material found in `data/course/`. Run `python scripts/ingest.py`.")
        return

    options = {f"{entry['doc']} — {entry['section']}": entry for entry in sections}
    choice = st.selectbox("Topic", list(options.keys()))

    declared = (store.get_student(st.session_state.student_id) or {}).get("declared") or {}
    if declared:
        st.caption(
            "Declared preference (self-report): "
            + ", ".join(f"{questionnaire.label(dim)} {score:.2f}" for dim, score in sorted(declared.items()))
            + ". You will still get all four formats — that is the whole point."
        )

    st.info(
        "Format order is randomised per student and topic, so a difference between formats is not "
        "just an order effect. You can stop after any format."
    )
    if st.button("Start this topic", type="primary"):
        topic = options[choice]["section"]
        rng = random.Random(f"{st.session_state.student_id}|{topic}")
        order = rng.sample(list(FORMATS), k=len(FORMATS))
        st.session_state.topic = topic
        st.session_state.order = order
        st.session_state.cursor = 0
        st.session_state.grades = {}
        st.session_state.stage = "format"
        st.rerun()


def _cached_delivery(agent: Agent, topic: str, fmt: str) -> AgentResult:
    key = f"{st.session_state.student_id}|{topic}|{fmt}"
    cache = st.session_state.deliveries
    if key not in cache:
        with st.spinner(f"Building the {tools.FORMAT_LABELS[fmt]} version from the course material…"):
            cache[key] = agent.deliver(
                st.session_state.student_id, topic, fmt, session_id=st.session_state.get("session_id")
            )
    return cache[key]


def _cached_quiz(agent: Agent, topic: str, fmt: str) -> AgentResult:
    key = f"{st.session_state.student_id}|{topic}|{fmt}"
    cache = st.session_state.quizzes
    if key not in cache:
        with st.spinner("Writing three questions from the same passages…"):
            cache[key] = agent.quiz(
                st.session_state.student_id, topic, fmt, session_id=st.session_state.get("session_id")
            )
    return cache[key]


def stage_format(agent: Agent) -> None:
    topic = st.session_state.topic
    fmt = st.session_state.order[st.session_state.cursor]
    progress = f"Format {st.session_state.cursor + 1} of {len(st.session_state.order)}"

    st.subheader(f"4 · {tools.FORMAT_LABELS[fmt]} version — {progress}")
    st.caption(f"Topic: {topic}")

    result = _cached_delivery(agent, topic, fmt)
    if result.message:
        st.info(result.message)
    render_artifact(result.data.get("generate_format", {}), fmt)
    agent_trace(result)

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Quiz me →", type="primary", key=f"to_quiz_{fmt}"):
            st.session_state.stage = "quiz"
            st.rerun()
    with col_b:
        if st.button("Stop here and see my results", key=f"stop_{fmt}"):
            st.session_state.stage = "results"
            st.rerun()

    ask_the_agent(agent, st.session_state.student_id, topic)


def stage_quiz(agent: Agent) -> None:
    topic = st.session_state.topic
    fmt = st.session_state.order[st.session_state.cursor]

    st.subheader(f"5 · Short quiz — {tools.FORMAT_LABELS[fmt]} version")
    st.caption("Three questions, generated from the same passages. Answer in your own words; no answer keys are shown.")

    quiz = _cached_quiz(agent, topic, fmt)
    items = (quiz.data.get("generate_quiz") or {}).get("items") or []
    if not items:
        st.warning("No quiz could be generated for this topic. Moving on.")
        st.session_state.stage = "feedback"
        st.rerun()

    if quiz.handover:
        st.error(quiz.message)
        return

    with st.form(f"quiz_form_{fmt}"):
        answers: dict[str, str] = {}
        for position, item in enumerate(items, start=1):
            st.markdown(f"**Q{position}.** {item['question']}")
            answers[item["item_id"]] = st.text_area(
                "Your answer", key=f"ans_{item['item_id']}", height=70, label_visibility="collapsed"
            )
            st.caption(f"Grounded in: {item['source']}")
        submitted = st.form_submit_button("Submit answers", type="primary")

    if submitted:
        payload = [{"question_id": item_id, "answer": text} for item_id, text in answers.items()]
        with st.spinner("Grading and updating your profile…"):
            graded = agent.grade(
                st.session_state.student_id,
                topic,
                fmt,
                payload,
                session_id=st.session_state.get("session_id"),
            )
        st.session_state.grades[fmt] = graded
        st.session_state.stage = "feedback"
        st.rerun()

    agent_trace(quiz, "Agent trace for the quiz generation")


def stage_feedback(agent: Agent) -> None:
    topic = st.session_state.topic
    fmt = st.session_state.order[st.session_state.cursor]
    graded = st.session_state.grades.get(fmt)

    st.subheader(f"6 · What your answers show — {tools.FORMAT_LABELS[fmt]}")

    if graded is None:
        st.info("No grading recorded for this format.")
    else:
        data = graded.data.get("grade_answer") or {}
        st.metric(
            label=f"{tools.FORMAT_LABELS[fmt]} score",
            value=f"{float(data.get('score', 0)):g} / {float(data.get('max_score', 3)):g}",
        )
        for entry in data.get("results", []):
            if entry.get("error"):
                st.warning(f"{entry['item_id']}: {entry['error']}")
                continue
            st.write(f"**{entry['score']:g}/{entry['max_score']:g}** — {entry.get('why', '')}")
            if entry.get("source"):
                st.caption(f"Re-read: {entry['source']}")
        agent_trace(graded)

    last = st.session_state.cursor >= len(st.session_state.order) - 1
    left, right = st.columns(2)
    if not last:
        if left.button("Next format →", type="primary"):
            st.session_state.cursor += 1
            st.session_state.stage = "format"
            st.rerun()
    if right.button("See my results" if last else "Stop and see my results"):
        st.session_state.stage = "results"
        st.rerun()


def stage_results(agent: Agent, store: Store) -> None:
    topic = st.session_state.topic
    student_id = st.session_state.student_id
    student = store.get_student(student_id) or {}
    declared = compare.declared_vector(student.get("declared") or {})
    attempts = store.attempts_for(student_id, topic)
    rows = compare.format_rows(attempts)

    st.subheader("7 · Declared preference vs what actually worked")
    st.caption(f"Topic: {topic}")

    if rows:
        st.dataframe(
            [
                {
                    "format": tools.FORMAT_LABELS[row.format],
                    "mean score": f"{row.mean:.1f} / {row.max_score:.0f}",
                    "attempts": row.attempts,
                    "percent": f"{row.pct * 100:.0f}%",
                }
                for row in rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    st.markdown(compare.comparison_message(declared, rows, topic))

    warning = compare.coverage_warning(rows)
    if warning:
        st.warning(warning)
    suggestion = compare.suggested_next(rows)
    if suggestion:
        st.success(suggestion)

    st.divider()
    left, right = st.columns(2)
    if left.button("Try another topic", type="primary"):
        st.session_state.stage = "choose_topic"
        st.rerun()
    if right.button("Back to the start"):
        reset_flow()
        st.rerun()

    ask_the_agent(agent, student_id, topic)


# --------------------------------------------------------------------------- #
# Lecturer flow
# --------------------------------------------------------------------------- #
def render_lecturer(store: Store, agent: Agent) -> None:
    st.subheader("Lecturer digest — the three concepts the group has not grasped")
    digests = compare.lecturer_digest(store, top_n=3)

    if not digests:
        st.info("No graded questions yet. Seed the synthetic cohort or run a student session first.")
    else:
        st.caption(
            "Ranked by explicit flags, then by the lowest average score on graded questions. "
            "Synthetic cohort included so this view is meaningful out of the box."
        )
        st.dataframe(
            [
                {
                    "concept": entry["concept"],
                    "avg score": f"{entry.get('avg_pct', 0) * 100:.0f}%",
                    "flags": entry.get("flags", 0),
                    "graded answers": entry.get("graded", 0),
                    "course source": entry.get("source", ""),
                }
                for entry in digests
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    stats = store.stats()
    columns = st.columns(4)
    columns[0].metric("students", stats["students"])
    columns[1].metric("attempts", stats["attempts"])
    columns[2].metric("graded questions", stats["quiz_items"])
    columns[3].metric("flags", stats["flags"])

    synthetic = sum(1 for student in store.all_students() if seed.is_synthetic(student["id"]))
    if synthetic:
        st.caption(f"⚠️ {synthetic} of {stats['students']} records are synthetic demo data (ids starting with `syn-`).")

    st.divider()
    st.markdown("#### Which formats are working across the cohort?")
    all_attempts = store.all_attempts()
    if all_attempts:
        per_format = {}
        for fmt in FORMATS:
            values = [float(row["score"]) / float(row["max_score"] or 3) for row in all_attempts if row["format"] == fmt]
            if values:
                per_format[tools.FORMAT_LABELS[fmt]] = round(sum(values) / len(values) * 100, 1)
        st.bar_chart(per_format)
        st.caption(
            "Group-level average per format. Read this as a prompt for a question, not as evidence "
            "that a cohort has a 'style'."
        )
    else:
        st.info("No attempts yet.")

    with st.expander("Recent agent activity (trace tail)"):
        events = agent.trace.tail(12)
        if events:
            st.dataframe(
                [
                    {
                        "event": event.get("event"),
                        "detail": {k: v for k, v in event.items() if k not in ("ts", "run_id", "event")},
                    }
                    for event in events
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No events in this process yet. Full traces are in `logs/*.jsonl`.")

    if digests:
        csv = "concept,avg_pct,flags,graded,source\n" + "\n".join(
            f'"{entry["concept"]}",{entry.get("avg_pct", 0)},{entry.get("flags", 0)},'
            f'{entry.get("graded", 0)},"{entry.get("source", "")}"'
            for entry in digests
        )
        st.download_button("Download digest as CSV", data=csv, file_name="lecturer_digest.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def sidebar(settings: Settings, store: Store) -> str:
    with st.sidebar:
        st.title("🧪 VARK Check")
        st.caption("Test, don't assume. An agent that measures what works instead of labelling who you are.")
        page = st.radio("View", ["Student", "Lecturer"], label_visibility="collapsed")

        st.divider()
        st.caption(f"LLM: `{settings.provider}` / `{settings.model}`")
        st.caption(f"Course files: {len(list(settings.course_dir.glob('*')))} · steps ≤ {settings.max_steps}")

        if page == "Student" and st.session_state.get("student_id"):
            st.divider()
            st.caption(f"Signed in as `{st.session_state.student_id}`")
            if st.button("Reset my session"):
                reset_flow()
                st.rerun()

        if page == "Lecturer":
            st.divider()
            confirm = st.checkbox("I understand this deletes all local records")
            if st.button("Reset demo data", disabled=not confirm):
                store.clear_demo_data(keep_cache=True)
                seed.seed(store)
                st.cache_resource.clear()
                st.success("Demo data reset and re-seeded.")
                st.rerun()

        st.divider()
        st.caption(
            "Data stays on this machine (`data/varkcheck.db`). Traces go to `logs/*.jsonl`. "
            "The course material is readable without the agent."
        )
    return page


def main() -> None:
    settings, store, index, agent, rebuilt = backend()
    init_state()

    page = sidebar(settings, store)
    if rebuilt:
        st.toast("Course index rebuilt from data/course/")

    if page == "Student":
        stage = st.session_state.stage
        if stage == "identify":
            stage_identify(store)
        elif stage == "questionnaire":
            stage_questionnaire(store)
        elif stage == "choose_topic":
            stage_choose_topic(store, index)
        elif stage == "format":
            stage_format(agent)
        elif stage == "quiz":
            stage_quiz(agent)
        elif stage == "feedback":
            stage_feedback(agent)
        elif stage == "results":
            stage_results(agent, store)
    else:
        render_lecturer(store, agent)


if __name__ == "__main__":
    main()
