# Task prompts used by the tools

<!-- version: 1.0.0 -->

## FORMAT_V

Render the topic for a student who wants to SEE it.

- Start with one Mermaid diagram: fenced ```mermaid block, `flowchart TD` or
  `graph TD`, 6–12 nodes, node labels short (max 6 words), no parentheses or
  quotes inside labels.
- Then a caption of 3–4 sentences explaining what the diagram shows and which
  parts a student should look at first.
- Then at most 3 bullet points of "what to notice".
- End with one line per passage used: `source: <file> §<section>`.

## FORMAT_A

Render the topic for a student who wants to HEAR it.

- Write a spoken script of 150–220 words, exactly as a tutor would say it out
  loud, in the second person.
- Conversational, short sentences, no bullet points, no diagrams, no tables.
- Explain one idea at a time and signpost transitions ("First…", "Now the part
  that trips people up…").
- End with one line per passage used: `source: <file> §<section>`.

## FORMAT_R

Render the topic for a student who wants to READ it.

- A structured written summary with numbered headings.
- Each key term in **bold** with a one-line definition on first use.
- One short worked micro-example (3–5 lines), fully numeric if the topic allows.
- End with one line per passage used: `source: <file> §<section>`.

## FORMAT_K

Render the topic for a student who wants to DO it.

- A hands-on exercise in 4–6 numbered steps that the student performs on paper
  or in a notebook, using only what the passages support.
- For each step say what to write down or compute, and what to look for.
- End with a self-check question and the section to re-read if the answer is
  unclear.
- DO NOT include the solution or the final numeric answer.
- End with one line per passage used: `source: <file> §<section>`.

## QUIZ

Write a short quiz that measures understanding, not recall of a single word.

Return JSON only, exactly this shape:

```json
{"items": [
  {"question": "...",
   "answer_key": "...",
   "concept": "...",
   "source": "...",
   "rubric": "..."}
]}
```

- Exactly 3 items.
- Each question must be answerable from the passages alone, in 1–3 sentences.
- `answer_key` is a complete expected answer, 1–2 sentences, stored server-side.
- `concept` is a short canonical concept name (2–5 words) used for the lecturer
  digest. The same concept must get the same name every time.
- `source` is the exact citation of the passage the question is based on.
- `rubric` says what a full-credit answer must contain, in at most 15 words.
- Never put the answer inside the question.

## GRADE

Grade one student answer against the stored answer key and rubric.

Return JSON only:

```json
{"score": 0, "why": "..."}
```

- `score` is an integer: 2 = correct and complete, 1 = partially correct or
  right idea with a real gap, 0 = wrong or missing.
- `why` is at most 60 words, addressed to the student, in the second person.
- Explain WHAT was missing or right, not the full model answer: never restate
  the answer key verbatim, and never rewrite the student's reasoning for them.
- Point at the section to re-read when the score is below 2.
