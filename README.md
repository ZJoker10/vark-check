# 🧪 VARK Check

**An AI study agent that tests learning preferences instead of assuming them.**

Most students believe they are "visual" or "auditory" learners, and study
accordingly. The evidence for matching teaching to a preferred style is weak
(Pashler et al., 2008). So instead of asking students what they are and feeding
them that, this agent takes one topic from the *real* course material, renders it
in all four formats (Visual, Aural, Read/write, Kinesthetic), quizzes the student
after each one, and shows them which format actually produced understanding —
compared with what they declared at the start.

Built for the **Reply Student Tech Clash 2026 — "AI on Campus" (Level 2 · Build,
Path B · Coding)**.

```
declared preference  ──►  4 formats  ──►  4 quizzes  ──►  observed scores
        │                                                       │
        └──────────────────  the gap  ◄─────────────────────────┘
                                 │
                    lecturer digest: 3 concepts the group
                    has not grasped yet, with the source
```

**v2** flips it around: instead of measuring which format works on *our* course,
it summarises *your* material in the format your profile points to —
`python -m varkcheck.web`, upload, one questionnaire, download. See
[v2 — bring your own material](#v2--bring-your-own-material).

---

## Contents

- [v2 — bring your own material](#v2--bring-your-own-material)
- [What makes this an agent, not a chatbot](#what-makes-this-an-agent-not-a-chatbot)
- [Run it in 60 seconds](#run-it-in-60-seconds)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [The core flow](#the-core-flow)
- [Architecture](#architecture)
- [Tools](#tools)
- [Guardrails — in code, not in the prompt](#guardrails--in-code-not-in-the-prompt)
- [Memory](#memory)
- [Determinism, tokens and cost](#determinism-tokens-and-cost)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Limits & risks](#limits--risks)
- [Team](#team)
- [References](#references)

---

## v2 — bring your own material

v1 is a fixed experiment: one course, four formats, measure which one worked.
v2 answers the question students actually ask next — *"can you do that to **my**
notes?"*

Upload your material, take the questionnaire once, pick a shape, download the
report. Same VARK framing, same caveat, but the content is yours instead of
`data/course/`.

```bash
pip install -r requirements.txt
python -m varkcheck.web                 # http://127.0.0.1:8765
python -m varkcheck.web --provider ollama --model llama3.1
```

A local server rather than a `file://` page, for three reasons: the API key stays
on the machine it belongs to, a PDF can actually be assembled, and there is no
CORS dance to reach a local model. It is stdlib `ThreadingHTTPServer` — v2 adds
no web framework.

### The flow

1. **Read** — drop PDFs, DOCX, Markdown, plain text, pasted text or images. Up to
   20 files, 25 MB each. The page tells you what it actually extracted (sections,
   characters, headings, images) *before* anything is generated, because the
   commonest failure is a scanned PDF with no text layer.
2. **Declare** — the same eight items as v1, scored server-side, then kept in the
   browser's `localStorage`. Taking it once means never taking it again in that
   browser, and the emphasis *"a starting hypothesis, not a diagnosis"* is stated
   in the quiz, the profile card, the report and the exported file.
3. **Choose a shape** — the same material becomes four genuinely different
   documents. Not the same text with a different accent colour: the block order,
   the prose register, the units and what is present all change.

| | Visual | Aural | Read / write | Kinesthetic |
|---|---|---|---|---|
| Order | diagrams → terms → prose | summary → spoken explanation → glossary | definitions → numbered analysis → tables | numbered steps → self-checks |
| Diagrams | 1–3, required | **never** | never | at most 1 |
| Tables | yes | no | yes | no |
| Prose register | captions and labels | "say it out loud" narration | dense, formal, numbered | imperative, one action per step |

The shape is enforced in code, not requested in the prompt — `_normalise()`
strips diagrams from an aural report even when the model sends them, because a
report with a flowchart in it is not an aural report.

### What makes it a summary and not an excerpt

This is the part worth reading. Four checks run on every draft, and any of them
can send it back to the model once:

| Guardrail | Rule | Where |
|---|---|---|
| **Synthesis** | 6-word shingle containment against the source. A sentence is "copied" at ≥ 0.55 similarity; a draft over 30 % copied is rejected. | `varkcheck/synthesis.py` |
| **Coverage** | The document is grouped into parts with their own sources; the report must cite each part. Missing parts are named in the retry prompt. | `summarize.part_coverage` / `_missing_parts` |
| **Shape** | Summary must be 90–420 words, must not describe itself ("This report covers…"), and a visual report must contain a usable diagram. | `summarize._quality_issues` |
| **Grounding** | Every `source: file §section` is resolved against the passages actually sent. An invented citation is replaced with the best real match, or the item is dropped. | `summarize._normalise` |

If the second draft still falls short, the report ships **with the failure stated
on the page** — "the summary is 60 words", "nothing in the report comes from
Lecture 1" — rather than silently pretending. The better of the two drafts is
the one kept: a genuine summary with one flaw beats a rewrite that copies its
source, and among drafts that both copy, the one that copies less wins. When a
weaker model returns an unusable diagram, a visual report falls back to a
hierarchy chart built from the document's own headings, and says so in the
caption. The page also tells you when a draft was rewritten; the exported file
reports only what is still wrong with the document you are holding.

### Export

PDF (WeasyPrint), DOCX (`python-docx`), or Markdown. Diagrams are drawn in the
browser — Mermaid runs in the parent page over the preview iframe's DOM, so no
Node and no headless Chrome are needed — and the resulting SVG is posted back and
embedded in the file. In the DOCX it is rasterised with `cairosvg`. In the
Markdown the Mermaid source is kept, which is what you want in a repo.

### Privacy

No account, no cookie, no analytics, and the server does not log IP addresses.
Uploads live in an in-memory session keyed by a random token, capped at 8
sessions, discarded after two hours, and their temp directories deleted. Your
answers never leave the browser except as part of a report request. The only
outbound traffic is the prompt to the model you configured — run
`--provider ollama` and there is none at all.

Images are **embedded as figures, not interpreted**. There is no vision model in
this build, so a picture of a diagram is placed in the report with its caption
and nothing is claimed about its contents. The UI says this in as many words.

### What v2 does not do

- **Small local models struggle.** `llama3.1` (8B) frequently produces 60-word
  summaries that describe the report instead of the material, and covers one
  lecture out of three even after a retry. `gemma4:12b` obeys the schema and the
  coverage rule reliably at ~25 s per call. The guardrails surface the difference
  honestly instead of hiding it; a frontier model is the obvious setting for
  marking, and `--provider deepseek` is one flag away.
- **No OCR.** A scanned PDF yields a warning, not a report.
- **No recognition.** Nothing learns across sessions; two uploads of the same
  file are two unrelated documents.
- WeasyPrint's flexbox support is partial, so wide layouts stack. Legible, not
  pixel-perfect.

### Where it lives

```
varkcheck/web/          server.py, __main__.py, static/{index.html,app.js,style.css,mermaid.min.js}
varkcheck/uploads.py    PDF/DOCX/MD/TXT/image extraction -> Document/Section/Figure
varkcheck/profile.py    Profile, score(), describe(), localStorage serialisation
varkcheck/synthesis.py  the copy detector and its retry instruction
varkcheck/summarize.py  passage selection, coverage, quality gate, retry loop
varkcheck/render.py     HTML / PDF / DOCX / Markdown, prompt-injection sanitiser
varkcheck/templates/    report.html — one template, four styles
prompts/report.md       SHARED_RULES + REPORT_V/A/R/K + schema (version 2.0.0)
tests/test_v2.py        54 tests: extraction, coverage, profiling, copy check, renderers
```

v1 is untouched: `app.py` and its 86 tests still run exactly as before.

---

## What makes this an agent, not a chatbot

| Challenge requirement | Where it is implemented |
|---|---|
| **Works with real material** | Only `data/course/` is indexed and only retrieved passages are used. `search_notes` refuses when nothing clears the relevance floor, and every generated artefact must carry a `source: file §section` that the output guardrail verifies against the passages actually retrieved. |
| **Does something, not just says something** | Seven tools with side effects: artefacts are cached, quizzes are stored with server-side keys, grades are written, the learner profile is updated, concepts are queued for the lecturer, reviews can be scheduled. |
| **Knows its limits** | Answer requests are refused *before* the model is called. Uncited course content never reaches the student. When the model fails twice, or a budget runs out, the case is handed to a human and a flag is written. |

Everything else follows from those three.

---

## Run it in 60 seconds

No API key, no network, no cost — this runs the entire pipeline against a
throwaway database with a deterministic stand-in model:

```bash
python scripts/demo_flow.py
```

You will see: a declared preference → four artefacts with citations → three
questions per format → grading → per-format comparison → a guardrail refusing
"give me the answers" → a lecturer digest.

For the real thing (a real model, the Streamlit UI), keep reading.

---

## Quickstart

```bash
git clone <this-repo> && cd vark-check

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # streamlit only
cp .env.example .env                     # then paste your API key

python scripts/ingest.py                 # build the retrieval index
streamlit run app.py                     # open http://localhost:8501
```

Optional extras (the code detects them and degrades gracefully):

```bash
pip install pytest                       # for the test suite
pip install pypdf                        # to ingest PDFs as well as .md/.txt
pip install sentence-transformers        # better embeddings than the built-in TF-IDF
```

### Models, libraries and frameworks

| Layer | Choice | Why |
|---|---|---|
| LLM | **DeepSeek** by default; any OpenAI-compatible endpoint; **Ollama** for a fully local model | One JSON contract, three providers, no SDK pinning |
| UI | **Streamlit** | One file, two pages, fastest path to a working demo |
| Retrieval | Built-in TF-IDF + cosine (pure Python); optional **sentence-transformers / MiniLM** | Zero downloads by default, no GPU, deterministic; upgrade path kept |
| Memory | **SQLite** (stdlib) | The brief asks for structured memory, not chat history |
| HTTP | `urllib` (stdlib) | No dependency for a 60-line transport |
| Tests | `unittest` classes (pytest-compatible) | Runs with or without pytest installed |

The v1 agent core has **one** third-party dependency: Streamlit. Everything else
is standard library, which is why `scripts/demo_flow.py` and the 86 v1 tests run
on a bare interpreter. v2 is where the real dependencies live — WeasyPrint,
`python-docx`, `pypdf`, `cairosvg` — because turning model prose into a PDF (and
a diagram into a DOCX image) is not something to hand-roll.

---

## Configuration

Copy `.env.example` → `.env`. Real environment variables always win over the file.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `deepseek` | `deepseek`, `openai`, `ollama`, `mock` |
| `LLM_API_KEY` | — | your key (any non-empty string for Ollama) |
| `LLM_BASE_URL` | per provider | override for a university gateway |
| `LLM_MODEL` | `deepseek-chat` | any compatible model id |
| `LLM_TEMPERATURE` | `0.2` | low: this is a study tool, not a creativity tool |
| `AGENT_MAX_STEPS` | `6` | the loop's step budget |
| `AGENT_TOKEN_BUDGET` | `25000` | the loop's token budget |
| `RAG_TOP_K` | `4` | passages per retrieval |
| `DB_PATH` | `data/varkcheck.db` | memory |
| `LOG_DIR` | `logs` | JSONL traces |

Provider presets are in `varkcheck/config.py`. For Ollama:

```bash
LLM_PROVIDER=ollama LLM_MODEL=llama3.1 streamlit run app.py
```

---

## The core flow

1. **Identify** — a pseudonymous id (`s-1042`). No email, no name.
2. **Declare** — eight neutral self-report statements, two per VARK dimension,
   1–5. These are *our own* items; the official VARK questionnaire is
   copyrighted and is not reproduced here (see [vark-learn.com](https://vark-learn.com)).
   The result is a hypothesis, and the UI says so.
3. **Choose a topic** — the menu is generated from the headings of
   `data/course/`, so a topic is always a real section with a citation.
4. **Four formats** — the order is *randomised per student and topic*, so a
   difference between formats is not just a fatigue/order effect. The student can
   stop after any format.
   - **V** — a Mermaid diagram + caption
   - **A** — a spoken script (offline TTS is optional, not on by default)
   - **R** — a structured written summary
   - **K** — a hands-on exercise with a self-check, never the solution
5. **Quiz → grade → profile** — three questions from the same passages; keys stay
   in SQLite; grading explains *why*, never restates the key.
6. **Results** — declared vs observed, per format, for this topic, with a neutral
   message. Deterministic arithmetic, no model call.
7. **Lecturer view** — the top three concepts by flags then by lowest average
   score, with the course source for each. Ten synthetic students are seeded so
   the view is meaningful on first launch (ids start with `syn-`, and the UI says
   so).

---

## Architecture

Full description and diagram: [`docs/architecture.md`](docs/architecture.md).
Two decisions are worth stating up front.

**The model never holds the artefacts.** Tools return a 600-character
`summary` (the observation the model sees) and a rich `data` payload that only
the application reads. Without that split, a six-step loop re-sends every
generated diagram on every step and the token bill triples for no benefit.

**The loop is verified, not trusted.** `Agent.deliver() / .quiz() / .grade()`
run the plan → act → observe loop, then check that the expected tool actually
ran and that a grade was written. If not, the same tool is invoked directly and
`agent.fallback` is logged. A live demo should not hinge on a model choosing
correctly on the first attempt.

Example loop with a real trace: [`docs/example_interactions.md`](docs/example_interactions.md).

---

## Tools

| Tool | Input | What it does | Guardrail inside |
|---|---|---|---|
| `search_notes` | `query`, `k` | Retrieves passages with `file §section` citations | Returns `no_material` instead of a weak match |
| `generate_format` | `topic`, `format` | Renders V/A/R/K from retrieved passages; caches the result | Rejects uncited or mis-cited output |
| `generate_quiz` | `topic`, `format`, `student_id` | 3 questions + keys, stored server-side | A question may not contain its own answer |
| `grade_answer` | `answers[]` | Batch grades, updates the item | Feedback containing the key is replaced |
| `update_profile` | `student_id`, `topic`, `format`, `score` | Writes the attempt | Score clamped to `[0, max]` |
| `flag_for_lecturer` | `concept`, `reason` | Queues a concept for the digest | — |
| `schedule_review` | `student_id`, `topic`, `days` | Optional spaced repetition | — |

Tool schemas live in `varkcheck/tools.py` and are injected into the system prompt;
handlers are the only code that touches memory or the index.

---

## Guardrails — in code, not in the prompt

| Check | When | Blocks | Type |
|---|---|---|---|
| `check_student_input` | **before** the model call | "give me the answers", "do my exercise", "soluzioni del quiz"… | regex, EN + IT |
| labelling | before display | "you are a visual learner", "visual learners should…" | regex |
| answer-key leak | before display | any stored key appearing in feedback | normalised substring match |
| citation | before display, for generated content | missing or fabricated `source:` | regex + allow-list of retrieved docs |

All four cost zero tokens. The prompt states the same rules — but a prompt can be
talked around and a string check cannot. Each has a test; see
`tests/test_guardrails.py` and `tests/test_scenarios.py`.

Refusals are designed to be *useful*: a blocked answer request appends a
model-free pointer to the most relevant section.

---

## Memory

SQLite (`data/varkcheck.db`), created automatically:

```
students    id, declared (JSON), created_at
attempts    student, topic, format, score, max_score, ts
flags       concept, reason, student, topic, ts
quiz_items  question, ANSWER_KEY, concept, source, rubric, graded, awarded
reviews     student, topic, due_date
cache       sha256(topic|format|material_hash) → artefact
messages    session_id, role, content
```

Answer keys exist only in `quiz_items`, and no prompt that produces
student-facing text receives them.

---

## Determinism, tokens and cost

Everything that must be trustworthy is deterministic: questionnaire scoring,
grading arithmetic (not grading *judgement*), the comparison, the lecturer digest,
the guardrails. The synthetic cohort uses a fixed seed, so the digest is identical
on every machine — which matters when you record a demo twice.

Measured shape of one student-topic pass (all four formats):

| Stage | Calls | Input | Output |
|---|---|---|---|
| `search_notes` | 4 | local embeddings, **0 API tokens** | — |
| `generate_format` | 4 | ~2 k | ~500 |
| `generate_quiz` | 4 | ~2.2 k | ~450 |
| `grade_answer` (3 answers batched) | 4 | ~1.5 k | ~700 |
| loop history overhead | — | +6–10 k (capped) | — |

≈ **2¢ per student-topic**, ≈ **6¢ per student** for three topics at
DeepSeek's list price (≈ $4 for 100 students × 2 topics). Three levers keep it
there: artefacts out of context, content-addressed caching (re-runs are free),
and no model call at all for refusals, comparison or digest.

`AGENT_MAX_STEPS` and `AGENT_TOKEN_BUDGET` are enforced in the loop; exceeding
either produces the same "I've flagged this for your lecturer" path.

---

## Tests

```bash
pytest -q                      # or: python -m unittest discover -s tests
```

**140 tests, ~15 seconds, no network, no API key.** They run against the real
course material in `data/course/`, so they fail if the demo material stops being
indexable.

| File | Covers |
|---|---|
| `test_rag.py` | heading-aware chunking, citations, retrieval relevance, off-material queries, index persistence |
| `test_guardrails.py` | answer requests (EN + IT), labelling, key leaks, missing/fabricated citations, false-positive control |
| `test_compare.py` | questionnaire scoring, per-format aggregation, neutral wording passes the guardrail, digest ranking |
| `test_store.py` | upsert, attempts, the answer-key vault, cache, digest ordering, history |
| `test_tools.py` | every tool, including cache hits, hidden keys, empty answers costing no tokens, leak replacement, bad arguments |
| `test_agent.py` | the loop, step/token budgets, protocol failures, provider failure, both fallbacks, profile written even if the model forgets |
| `test_scenarios.py` | replay of the five documented interactions, including the two boundary cases |
| `test_v2.py` | upload extraction (order, bad files, image figures, caps), part coverage and sampling, profile clamping, the copy detector, blank-reply recovery, and each renderer — including that a PDF has no escaped HTML and that all four styles differ |

---

## Project layout

```
vark-check/
├── app.py                     Streamlit UI: student page + lecturer digest          (v1)
├── prompts/
│   ├── system.md              ROLE · GOAL · RULES · PROCEDURE · OUTPUT FORMAT (versioned)
│   ├── tasks.md               per-tool instructions (versioned)
│   └── report.md              v2: SHARED_RULES + per-style report briefs + schema
├── varkcheck/
│   ├── agent.py               plan → act → observe loop, budgets, hand-over, fallbacks
│   ├── tools.py               the seven tools + registry + observation caps
│   ├── guardrails.py          the four code-level checks
│   ├── rag.py                 chunking, TF-IDF/dense space, index persistence
│   ├── store.py               SQLite memory + the answer-key vault
│   ├── compare.py             declared vs observed (deterministic)
│   ├── questionnaire.py       eight self-report items (our own)
│   ├── config.py / env.py     settings, tiny .env loader
│   ├── llm.py                 provider-agnostic client, retries, JSON extraction
│   ├── prompts.py             prompt loader with version tags
│   ├── trace.py               JSONL tracing
│   ├── seed.py                synthetic cohort (fixed seed)
│   ├── offline.py             deterministic stand-in model for tests/demo
│   ├── uploads.py             v2: PDF/DOCX/MD/TXT/image → Document/Section/Figure
│   ├── profile.py             v2: scoring, describe(), localStorage round-trip
│   ├── synthesis.py           v2: the copy detector and its retry instruction
│   ├── summarize.py           v2: passages, coverage, quality gate, retry loop
│   ├── render.py              v2: HTML/PDF/DOCX/Markdown + sanitiser
│   ├── templates/report.html  v2: one template, four styles
│   └── web/                   v2: server.py, __main__.py, static/
├── data/course/               the real material the agent is allowed to use
├── scripts/
│   ├── ingest.py              build/rebuild the index
│   ├── seed_demo.py           refresh the synthetic cohort, print the digest
│   └── demo_flow.py           end-to-end walkthrough, no API key
├── tests/                     140 tests (86 v1 + 54 v2)
└── docs/
    ├── architecture.md
    └── example_interactions.md
```

---

## Limits & risks

**Student data.** Stored locally in one SQLite file, on the machine that runs the
app. Pseudonymous ids only; no names, emails, or accounts. The data owner is
whoever runs the instance (the course, not the vendor). There is a
"Reset demo data" button and `make clean`. Deleting `data/` is a complete delete —
nothing is sent anywhere except the text of the prompt to the configured LLM
provider, which is worth stating explicitly in a real deployment.

**Tool unavailable the night before an exam.** The course material is plain
Markdown in `data/course/` and stays readable with no agent, no index and no
network. `scripts/demo_flow.py` reproduces the whole pipeline offline with a
deterministic stand-in model, so a dead venue Wi-Fi does not kill the demo.
The index is a local file; a missing key degrades to "no new artefacts", not to
"no course material".

**Access.** Runs in a browser via Streamlit. No paid account is needed by
students: the app is served by the institution, and `LLM_PROVIDER=ollama` makes
the whole stack local and free. If a hosted model is used, cost is ≈ 6¢ per
student per three topics.

**Scientific honesty.** The tool does **not** claim learning styles exist, and it
never labels a student. It measures what worked on *this* topic with *these*
questions, and says so: three questions per format is indicative, not diagnostic.
The research position is stated in the UI and in `docs/example_interactions.md`.
"VARK" is the trademark of its authors; this project references
[vark-learn.com](https://vark-learn.com) and does not reproduce their
questionnaire.

**Small-sample noise.** One topic is one data point. The comparison screen
repeats this, and `coverage_warning()` names the formats not yet attempted. A
student should be encouraged to try the recommended format on the *next* topic to
see whether it holds.

**What is deliberately not built.** No accounts, no LMS integration, no audio
generation by default, no PDF parsing unless `pypdf` is installed, no
multi-institution tenancy. One core flow, working end to end.

---

## Team

<!-- TEAM:START -->
| | Name | Contribution |
|---|---|---|
| 1 | Zeyad Wael | Agent core, retrieval, guardrails, UI |
| 2 | _(name)_ | _(role)_ |
| 3 | _(name)_ | _(role)_ |
| 4 | _(name)_ | _(role)_ |
<!-- TEAM:END -->

---

## References

- Pashler, H., McDaniel, M., Rohrer, D., & Bjork, R. (2008). *Learning Styles:
  Concepts and Evidence.* Psychological Science in the Public Interest, 9(3), 105–119.
- VARK — a guide to learning preferences: <https://vark-learn.com> (referenced,
  not reproduced).
- DeepSeek API: <https://api-docs.deepseek.com> · Ollama: <https://ollama.com>
