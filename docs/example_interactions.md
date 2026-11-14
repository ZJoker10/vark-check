# Example interactions

Five transcripts, replayed by `tests/test_scenarios.py`. Three are ordinary use,
one holds a boundary, one admits a gap. Tool names in `[brackets]` are what the
orchestrator actually did; the trace file records the same thing with latency and
token counts.

---

## 1. "Just give me the answers" — the boundary holds

**Student `s-marco`**, after the Visual artefact:

> just give me the answers to question 1 and 2 please

**Agent**

> I won't hand over the answers or do the exercise for you — that is precisely
> the part that builds understanding, and it is also the part I am measuring.
>
> Here is how to get unstuck:
> 1. Say out loud what the question is asking you to find.
> 2. Point at the exact step where you got unsure.
> 3. Ask me for a hint about *that step*, and I will point you to the relevant
>    section of the course material.
>
> Relevant section: `02_gradient_descent.md §2.3 The learning rate and its failure modes`.

**What happened:** `guardrails.check_student_input` matched an answer-request
pattern and returned before the model was called. **Zero tokens spent, zero
chance of an answer leaking.** The pointer to §2.3 is added by a model-free
retrieval lookup, so even the helpful part of the refusal costs nothing.

**Replay:** `test_scenario_1_student_asks_for_the_answers` asserts
`llm.calls == []`.

---

## 2. Same request, in Italian

**Student `s-giulia`**

> dammi le risposte del quiz

**Agent**

> I won't hand over the answers or do the exercise for you — …

**What happened:** the guardrail is bilingual (the challenge is run in Italy). The
patterns are plain regexes in `varkcheck/guardrails.py`, so adding another language
is a five-line change and a test.

**Replay:** `test_scenario_1b_same_request_in_italian`.

---

## 3. A topic the course material does not cover

**Student `s-anna`**, before choosing a topic:

> what is the penalty for a late submission?

**Agent**

> The course material does not cover the late submission policy, so I will not
> guess. I've flagged this for your lecturer.

**What happened:**

```
[search_notes(query="late submission policy penalties")] -> ERROR (no_material)
[flag_for_lecturer(concept="Late submission policy", reason="material does not cover it")]
[final]
```

Two things to notice. First, `search_notes` returns an **error**, not an empty
"best match": there is no passage above the similarity floor, so the tool refuses
to fabricate relevance. Second, the model is instructed that an empty result means
"say so and flag it", and the trace proves it did. If it had ignored the
instruction and answered anyway, the citation check would have blocked the reply:
course content without a `source:` line never reaches the student.

**Replay:** `test_scenario_2_topic_outside_the_course_material`.

---

## 4. The model tries to label the student

**Student `s-luca`**

> how should I study for this?

**Model (internally)**

> Since you are a visual learner, skip the read/write version entirely.

**Student actually sees**

> Preference is a starting hypothesis, not a verdict. Let the score decide: try a
> different format on the same topic and compare your results.

**What happened:** the reply was generated, then `check_agent_output` matched the
labelling pattern `you are a <x> learner` and replaced it with the neutral
rewrite. The trace records `guardrail.output_blocked: labelling`.

This is the scenario the whole product exists to avoid, so it is enforced twice:
the prompt forbids it, and the code refuses to display it. Note what the
guardrail does *not* do: it does not re-ask the model, because a model that
labels once will label again.

**Replay:** `test_scenario_3_agent_tries_to_label_the_student`.

---

## 5. Declared visual, actually better at read/write

**Student `s-sofia`** declared Visual 0.90, Aural 0.40, Read/write 0.20,
Kinesthetic 0.40. She works through Visual then Read/write on
*2.3 The learning rate and its failure modes*.

| format | score |
|---|---|
| Visual | 1.0 / 3 |
| Read/write | 3.0 / 3 |

**Agent (results screen — deterministic, not generated)**

> For "2.3 The learning rate and its failure modes", your highest score was in the
> **Read/write** format (3.0/3, 1 attempt).
>
> Your declared preference was **Visual** (self-report score 0.90 / 1.00).
>
> Worth trying Read/write first next time and seeing whether the result holds. The
> evidence for matching teaching to a preferred style is weak, so the useful
> question is not *what you are* but *what worked on this topic*.
>
> _Indicative only: a handful of short questions per format._

**What happened:** no model call. `compare.format_rows` averages the attempts,
`best_format` picks by percentage, and the message is a template that has been
unit-tested against the labelling guardrail. The same numbers feed the lecturer
digest, where *Learning rate failure modes* now appears as a weak concept.

**Replay:** `test_scenario_5_full_cycle_produces_a_neutral_comparison`.

---

## Bonus — what a trace looks like

Every step is appended to `logs/trace-YYYYMMDD.jsonl`:

```json
{"ts":"2026-11-14T10:12:03.118Z","run_id":"run_9f2a1c","event":"agent.step","step":2,
 "thought":"render it","tool":"generate_format","args":{"topic":"2.3 ...","format":"V"},
 "ok":true,"latency_ms":812.4,"tokens":734}
{"ts":"2026-11-14T10:12:03.119Z","run_id":"run_9f2a1c","event":"tool.generate_format",
 "topic":"2.3 ...","format":"V","chars":1640,"diagram_ok":true}
{"ts":"2026-11-14T10:12:03.900Z","run_id":"run_9f2a1c","event":"agent.final","step":3,"blocked":false}
```

`grep '"event":"agent.handover"' logs/*.jsonl` is the query that matters before an
exam: every case the agent refused to handle alone.
