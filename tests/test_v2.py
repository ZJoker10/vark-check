"""v2 tests: upload extraction, profiling, the copy check and the renderers.

These deliberately avoid the network. The one real-model path is exercised
manually; everything here is deterministic.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from varkcheck import profile as profile_mod  # noqa: E402
from varkcheck import render, summarize, synthesis, uploads  # noqa: E402
from varkcheck.questionnaire import QUESTIONS  # noqa: E402

COURSE = REPO / "data" / "course"

MARKDOWN = """# Photosynthesis

Plants convert light into chemical energy.

## Light reactions

Chlorophyll absorbs photons in the thylakoid membrane. Water is split and oxygen leaves as a by-product.

## Calvin cycle

Carbon dioxide is fixed into sugar using the energy stored by the light reactions.
"""


def course_files() -> list[tuple[str, bytes]]:
    return [(path.name, path.read_bytes()) for path in sorted(COURSE.glob("*.md"))]


def visual_profile() -> profile_mod.Profile:
    responses = {question.id: (5 if question.dimension == "V" else 2) for question in QUESTIONS}
    return profile_mod.score(responses)


class ScriptedLLM:
    """A model with a fixed script: one reply per call, in order."""

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.calls = 0

    def chat(self, messages, **kwargs):
        import json

        from varkcheck.llm import LLMReply, Usage

        self.calls += 1
        item = self.script.pop(0)
        return LLMReply(text=json.dumps(item), usage=Usage(10, 10), model="scripted")


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="vark-v2-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()


# --------------------------------------------------------------------------- #
class UploadTests(TempDirTest):
    def test_markdown_split_on_headings(self) -> None:
        document = uploads.extract([("photosynthesis.md", MARKDOWN.encode())], self.tmp)
        headings = [section.heading for section in document.sections]
        self.assertIn("Light reactions", headings)
        self.assertIn("Calvin cycle", headings)
        self.assertGreater(document.chars, uploads.MIN_USEFUL_CHARS)
        self.assertTrue(document.has_text)

    def test_sources_are_provenance_strings(self) -> None:
        document = uploads.extract([("photosynthesis.md", MARKDOWN.encode())], self.tmp)
        for section in document.sections:
            self.assertIn("photosynthesis.md", section.source)

    def test_sections_keep_file_order(self) -> None:
        """Two files must not interleave: the report's coverage depends on order."""
        first = "# Alpha\n\n" + ("Alpha paragraph. " * 30) + "\n\n## Alpha two\n\n" + ("More alpha. " * 30)
        second = "# Beta\n\n" + ("Beta paragraph. " * 30) + "\n\n## Beta two\n\n" + ("More beta. " * 30)
        document = uploads.extract(
            [("a_one.md", first.encode()), ("b_two.md", second.encode())], self.tmp
        )
        files = [section.source.split(" §")[0] for section in document.sections]
        self.assertEqual(files, sorted(files, key=lambda name: files.index(name)))
        beta_start = files.index("b_two.md")
        self.assertTrue(all(name == "b_two.md" for name in files[beta_start:]))

    def test_untitled_text_does_not_become_preamble(self) -> None:
        document = uploads.extract([("pasted-notes.md", ("A paragraph about loss. " * 20).encode())], self.tmp)
        self.assertNotIn("Preamble", document.outline())

    def test_bad_files_become_warnings(self) -> None:
        document = uploads.extract(
            [("malware.exe", b"MZ"), ("empty.txt", b""), ("broken.pdf", b"%PDF-1.4 nope")],
            self.tmp,
        )
        self.assertEqual(document.sections, [])
        self.assertFalse(document.has_text)
        joined = " ".join(document.warnings)
        self.assertIn("unsupported type", joined)
        self.assertIn("empty file", joined)
        self.assertIn("broken.pdf", joined)

    def test_image_is_kept_as_a_figure_with_a_visible_note(self) -> None:
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
            b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00"
            b"\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        document = uploads.extract([("diagram.png", png)], self.tmp)
        self.assertEqual(len(document.figures), 1)
        self.assertTrue(Path(document.figures[0].path).is_file())
        self.assertTrue(any("not interpreted" in warning for warning in document.warnings))

    def test_oversized_upload_is_refused_not_crashed(self) -> None:
        document = uploads.extract([("huge.md", b"x" * (uploads.MAX_FILE_BYTES + 10))], self.tmp)
        self.assertEqual(document.files, [])
        self.assertTrue(any("exceeds" in warning for warning in document.warnings))

    def test_text_cap_truncates_instead_of_failing(self) -> None:
        paragraph = ("word " * 200) + "\n\n"
        document = uploads.extract(
            [("long.md", (paragraph * 400).encode())], self.tmp
        )
        self.assertLessEqual(document.chars, uploads.MAX_TOTAL_CHARS)
        self.assertTrue(document.has_text)

    def test_chunk_splits_long_sections(self) -> None:
        document = uploads.extract(
            [("long.md", ("Sentence about gradients. " * 400).encode())], self.tmp
        )
        pieces = uploads.chunk(document, max_chars=500)
        self.assertGreater(len(pieces), len(document.sections))
        for piece in pieces:
            self.assertLessEqual(len(piece.text), 500)

    def test_major_parts_skips_page_and_subsection_headings(self) -> None:
        document = uploads.extract(course_files(), self.tmp)
        parts = summarize._major_parts(document)
        self.assertEqual(len(parts), 3)
        self.assertTrue(all("Lecture" in part for part in parts))


# --------------------------------------------------------------------------- #
class CoverageTests(TempDirTest):
    def test_parts_own_their_sources(self) -> None:
        document = uploads.extract(course_files(), self.tmp)
        coverage = summarize.part_coverage(document)
        self.assertEqual(len(coverage), 3)
        all_sources = [source for part in coverage for source in part["sources"]]
        self.assertEqual(len(all_sources), len(set(all_sources)), "a source belongs to one part")

    def test_missing_parts_are_detected(self) -> None:
        document = uploads.extract(course_files(), self.tmp)
        coverage = summarize.part_coverage(document)
        report = summarize.Report(
            title="t", one_line="o", summary="s" * 400, style="V",
            sections=[{"heading": "h", "body": "b", "source": coverage[0]["sources"][0]}],
            citations=[coverage[0]["sources"][0]],
        )
        missing = summarize._missing_parts(report, coverage)
        self.assertEqual(len(missing), 2)

    def test_many_parts_are_sampled_not_crowded_out(self) -> None:
        document = uploads.Document(title="Big")
        for index in range(20):
            document.sections.append(
                uploads.Section(
                    heading=f"Part {index}", text="Body text here. " * 20, source=f"f.md §Part {index}", order=index
                )
            )
        coverage = summarize.part_coverage(document, limit=4)
        self.assertEqual(len(coverage), 4)
        labels = [part["label"] for part in coverage]
        self.assertIn("Part 0", labels)
        self.assertIn("Part 19", labels, "sampling must still reach the end of the document")


# --------------------------------------------------------------------------- #
class ProfileTests(unittest.TestCase):
    def test_score_is_arithmetic_and_bounded(self) -> None:
        scores = profile_mod.score({question.id: 5 for question in QUESTIONS}).scores
        self.assertTrue(all(0.0 <= value <= 1.0 for value in scores.values()))
        self.assertTrue(all(value == 1.0 for value in scores.values()))

    def test_dominant_and_secondary(self) -> None:
        built = visual_profile()
        self.assertEqual(built.dominant, "V")
        self.assertIsNone(built.secondary, "a 0.75 point gap is too wide for a secondary")

    def test_round_trip_through_localstorage_json(self) -> None:
        original = visual_profile()
        restored = profile_mod.Profile.from_json(original.to_json())
        self.assertEqual(original.scores, restored.scores)
        self.assertEqual(original.dominant, restored.dominant)

    def test_from_dict_clamps_hostile_values(self) -> None:
        restored = profile_mod.Profile.from_dict(
            {"scores": {"V": "99", "A": -4, "R": "abc"}, "responses": {"q1": 900}}
        )
        self.assertEqual(restored.scores["V"], 1.0)
        self.assertEqual(restored.scores["A"], 0.0)
        self.assertEqual(restored.scores["R"], 0.0)
        self.assertEqual(restored.responses["q1"], 5)

    def test_incomplete_and_out_of_range_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            profile_mod.score({"q1": 3})
        payload = {question.id: 3 for question in QUESTIONS}
        payload["q2"] = 9
        with self.assertRaises(ValueError):
            profile_mod.score(payload)

    def test_description_keeps_the_hypothesis_caveat(self) -> None:
        self.assertIn("not a diagnosis", profile_mod.describe(visual_profile()))

    def test_an_all_equal_answer_set_is_reported_as_a_tie(self) -> None:
        """Eight items cannot separate four dimensions; claiming a winner would be
        a false claim at the top of the page."""
        flat = profile_mod.score({question.id: 3 for question in QUESTIONS})
        self.assertTrue(flat.tied)
        text = profile_mod.describe(flat)
        self.assertIn("no single preference", text)
        self.assertIn("not a diagnosis", text)
        self.assertTrue(flat.to_dict()["tied"])

    def test_a_clear_winner_is_not_reported_as_a_tie(self) -> None:
        self.assertFalse(visual_profile().tied)


# --------------------------------------------------------------------------- #
SOURCE = (
    "Gradient descent minimises a loss function by stepping against its gradient. "
    "The learning rate controls how large each step is, and a rate that is too high "
    "makes the loss oscillate or diverge instead of converging. Feature scaling keeps "
    "the loss surface roughly spherical so a single rate works for every parameter."
)


class SynthesisTests(unittest.TestCase):
    def test_verbatim_copy_is_rejected(self) -> None:
        verdict = synthesis.check(SOURCE, SOURCE)
        self.assertFalse(verdict.ok)
        self.assertGreater(verdict.copied_ratio, 0.5)

    def test_paraphrase_is_accepted(self) -> None:
        rewritten = (
            "Training moves parameters downhill along the error surface. Choose too big a step "
            "and the process bounces or explodes; normalise the inputs first so one step size "
            "behaves predictably everywhere."
        )
        verdict = synthesis.check(rewritten, SOURCE)
        self.assertTrue(verdict.ok, verdict.reason)

    def test_short_text_is_not_judged(self) -> None:
        verdict = synthesis.check("Too short.", SOURCE)
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.sentences, 0)

    def test_markup_is_skipped(self) -> None:
        report = "# Heading\n\n- a bullet that is fairly long but not prose\n\n```\ncode here\n```"
        self.assertEqual(synthesis.sentences(report), [])

    def test_retry_instruction_quotes_the_offender(self) -> None:
        verdict = synthesis.check(SOURCE, SOURCE)
        instruction = synthesis.retry_instruction(verdict)
        self.assertIn("rejected", instruction)
        self.assertIn("Gradient descent minimises", instruction)

    def test_compression_reports_a_ratio(self) -> None:
        result = synthesis.compression("short summary here", SOURCE)
        self.assertLess(result["ratio"], 1.0)
        self.assertGreater(result["source_words"], 0)


# --------------------------------------------------------------------------- #
def sample_report(style: str = "V") -> summarize.Report:
    return summarize.Report(
        title="Regression and Learning, Condensed",
        one_line="How the three lectures fit together.",
        summary="A short synthesis of the uploaded material. " * 6,
        style=style,
        sections=[
            {"heading": "The loop", "body": "A claim with **emphasis**.", "source": "a.md §1"},
            {"heading": "The rate", "body": "Another claim.", "source": "b.md §2"},
        ],
        key_terms=[{"term": "Gradient", "meaning": "Direction of steepest increase.", "source": "a.md §1"}],
        diagrams=[
            {
                "title": "The loop",
                "caption": "Predict, measure, correct.",
                "mermaid": "flowchart LR\n  A[Predict] --> B[Measure]\n  B --> C[Update]\n  C --> A",
            }
        ],
        tables=[{"title": "Where things live", "columns": ["Idea", "File"], "rows": [["Loss", "a.md"]]}],
        checks=[{"question": "What does the rate control?", "hint": "Think about step size."}],
        study_plan=["Draw the loop from memory"],
        citations=["a.md §1", "b.md §2"],
    )


SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80">'
    '<rect width="200" height="80" fill="#eef"/>'
    '<text x="10" y="45" font-family="DejaVu Sans" font-size="16">Predict then Update</text></svg>'
)


class RenderTests(TempDirTest):
    def test_pdf_is_a_pdf_and_contains_the_real_text(self) -> None:
        payload, mimetype, filename = render.render(sample_report(), visual_profile(), "pdf")
        self.assertEqual(mimetype, "application/pdf")
        self.assertTrue(filename.endswith(".pdf"))
        self.assertEqual(payload[:4], b"%PDF")

        from pypdf import PdfReader

        target = self.tmp / "report.pdf"
        target.write_bytes(payload)
        text = "".join(page.extract_text() or "" for page in PdfReader(str(target)).pages)
        self.assertIn("Regression and Learning", text)
        self.assertIn("The loop", text)
        self.assertNotIn("<h2>", text, "HTML must be rendered, not escaped")

    def test_pdf_embeds_the_diagram_as_vector(self) -> None:
        with_svg, _, _ = render.render(sample_report(), visual_profile(), "pdf", svgs={0: SVG})
        without, _, _ = render.render(sample_report(), visual_profile(), "pdf", svgs={})
        self.assertNotEqual(len(with_svg), len(without))
        self.assertGreater(len(with_svg), 1000)

    def test_preview_mode_leaves_diagrams_for_the_browser(self) -> None:
        html = render.to_html(sample_report(), visual_profile(), preview=True)
        self.assertIn('class="mermaid"', html)
        self.assertIn("Predict", html)

    def test_non_preview_degrades_to_the_mermaid_source(self) -> None:
        html = render.to_html(sample_report(), visual_profile())
        self.assertIn('class="pending"', html)
        self.assertIn("flowchart LR", html)

    def test_docx_opens_and_carries_the_sections(self) -> None:
        payload, mimetype, filename = render.render(
            sample_report("K"), visual_profile(), "docx", svgs={0: SVG}
        )
        self.assertEqual(payload[:2], b"PK")
        self.assertTrue(filename.endswith(".docx"))

        import docx

        document = docx.Document(io.BytesIO(payload))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        self.assertIn("Regression and Learning", text)
        self.assertIn("The loop", text)
        self.assertGreaterEqual(len(document.inline_shapes), 2, "chart plus diagram image")

    def test_markdown_export_is_complete(self) -> None:
        text = render.to_markdown(sample_report(), visual_profile())
        for expected in ("# Regression", "## Summary", "```mermaid", "## Terms to know", "## Sources"):
            self.assertIn(expected, text)

    def test_html_export_is_sanitised(self) -> None:
        report = sample_report()
        report.sections[0]["body"] = '<p>ok</p><script>alert(1)</script><img src=x onerror="alert(2)">'
        html = render.to_html(report, visual_profile())
        self.assertNotIn("<script", html.lower())
        self.assertNotIn("onerror", html.lower())
        self.assertIn("ok", html)

    def test_unknown_format_is_a_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            render.render(sample_report(), visual_profile(), "exe")

    def test_each_style_produces_a_different_document(self) -> None:
        payloads = {style: render.render(sample_report(style), visual_profile(), "html")[0] for style in "VARK"}
        self.assertEqual(len(set(payloads.values())), 4)
        self.assertNotIn(b'class="pending"', payloads["A"], "aural must not carry diagrams")
        self.assertNotIn(b'class="pending"', payloads["R"], "read/write must not carry diagrams")

    def test_profile_chart_is_a_png(self) -> None:
        import base64

        payload = base64.b64decode(render.profile_chart(visual_profile()))
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")

    def test_mermaid_svg_gets_an_intrinsic_size(self) -> None:
        """width="100%" gives an <img> no aspect ratio, so WeasyPrint stretches it."""
        mermaid = (
            '<svg id="d" width="100%" xmlns="http://www.w3.org/2000/svg" '
            'style="max-width: 131px;" viewBox="0 0 131 212"><rect width="131" height="212"/></svg>'
        )
        fitted = render._fit_svg(mermaid)
        self.assertIn('width="131"', fitted)
        self.assertIn('height="212"', fitted)
        self.assertNotIn('width="100%"', fitted)
        self.assertNotIn("max-width", fitted)
        self.assertIn("<rect", fitted, "the drawing itself must survive untouched")
        # An SVG with no viewBox is left alone rather than given a guessed size.
        self.assertEqual(render._fit_svg("<svg></svg>"), "<svg></svg>")

    def test_process_notes_stay_out_of_the_export(self) -> None:
        """The PDF states what is wrong with it, not how many drafts it took."""
        report = sample_report()
        report.notes = ["The first draft reused source wording (60% of sentences); it was rewritten."]
        report.warnings = ["This report still: The summary is 51 words."]

        html = render.to_html(report, visual_profile())
        markdown = render.to_markdown(report, visual_profile())
        for artifact in (html, markdown):
            self.assertIn("This report still: The summary is 51 words.", artifact)
            self.assertNotIn("was rewritten", artifact)


# --------------------------------------------------------------------------- #
class EmptyReplyTests(unittest.TestCase):
    """A blank reply is a model problem, not a report problem — retry cheaply."""

    def setUp(self) -> None:
        from varkcheck.config import Settings
        from varkcheck.llm import ChatClient

        self.client = ChatClient(
            Settings(provider="ollama", base_url="http://localhost:11434/v1", model="test")
        )

    def _script(self, replies: list[tuple[str, str]]) -> list[dict]:
        seen: list[dict] = []
        queue = list(replies)

        def fake_post(payload: dict) -> dict:
            seen.append(dict(payload))
            text, finish = queue.pop(0) if queue else ("", "stop")
            return {
                "choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        self.client._post = fake_post  # type: ignore[method-assign]
        return seen

    def test_a_truncated_reply_is_retried_with_a_bigger_ceiling(self) -> None:
        seen = self._script([("", "length"), ('{"ok": true}', "stop")])
        reply = self.client.chat(
            [{"role": "user", "content": "hi"}], max_tokens=100, json_mode=True
        )
        self.assertEqual(reply.text, '{"ok": true}')
        self.assertEqual(seen[0]["max_tokens"], 100)
        self.assertEqual(seen[1]["max_tokens"], 200, "the retry must raise the ceiling")

    def test_json_mode_is_dropped_when_the_reply_is_blank(self) -> None:
        seen = self._script([("", "stop"), ('{"ok": true}', "stop")])
        reply = self.client.chat(
            [{"role": "user", "content": "hi"}], max_tokens=100, json_mode=True
        )
        self.assertEqual(reply.text, '{"ok": true}')
        self.assertIn("response_format", seen[0])
        self.assertNotIn("response_format", seen[1])

    def test_a_persistently_blank_model_raises_a_useful_error(self) -> None:
        from varkcheck.llm import LLMError

        self._script([("", "stop"), ("", "stop")])
        with self.assertRaises(LLMError) as caught:
            self.client.chat([{"role": "user", "content": "hi"}], max_tokens=100, json_mode=True)
        self.assertIn("empty reply", str(caught.exception))

    def test_the_ceiling_is_capped(self) -> None:
        seen = self._script([("", "length"), ("", "length")])
        with self.assertRaises(Exception):
            self.client.chat(
                [{"role": "user", "content": "hi"}], max_tokens=12000, json_mode=False
            )
        self.assertTrue(all(payload["max_tokens"] <= 16000 for payload in seen))


# --------------------------------------------------------------------------- #
class MetadataTests(TempDirTest):
    def test_model_output_is_normalised_and_cited(self) -> None:
        document = uploads.extract([("photosynthesis.md", MARKDOWN.encode())], self.tmp)
        passages = summarize.select_passages(document)
        real_source = passages[0].source
        raw = {
            "title": "Plants and Light",
            "one_line": "How a leaf turns light into sugar.",
            "summary": "A summary that is long enough to be plausible. " * 6,
            "sections": [{"heading": "Light", "body": "Chlorophyll captures photons.", "source": real_source}],
            "key_terms": [{"term": "Chlorophyll", "meaning": "The pigment.", "source": real_source}],
            "diagrams": [],
            "tables": [],
            "checks": [],
            "study_plan": ["Draw the chloroplast"],
        }
        report = summarize._normalise(raw, style="V", passages=passages)
        self.assertEqual(report.citations, [real_source])
        self.assertEqual(report.sections[0]["source"], real_source)

    def test_invented_citation_is_replaced_by_a_real_one(self) -> None:
        document = uploads.extract([("photosynthesis.md", MARKDOWN.encode())], self.tmp)
        passages = summarize.select_passages(document)
        raw = {
            "title": "t",
            "one_line": "o",
            "summary": "s" * 300,
            "sections": [
                {
                    "heading": "Invented",
                    "body": "Chlorophyll absorbs photons inside the thylakoid membrane.",
                    "source": "nowhere.pdf p.99",
                },
            ],
            "key_terms": [],
            "diagrams": [],
            "tables": [],
            "checks": [],
            "study_plan": [],
        }
        report = summarize._normalise(raw, style="V", passages=passages)
        allowed = {passage.source for passage in passages}
        self.assertIn(report.sections[0]["source"], allowed)

    def test_a_section_that_cannot_be_grounded_is_dropped(self) -> None:
        document = uploads.extract([("photosynthesis.md", MARKDOWN.encode())], self.tmp)
        passages = summarize.select_passages(document)
        raw = {
            "title": "t",
            "one_line": "o",
            "summary": "s" * 300,
            "sections": [{"heading": "Invented", "body": "Zzz.", "source": "nowhere.pdf p.99"}],
            "key_terms": [],
            "diagrams": [],
            "tables": [],
            "checks": [],
            "study_plan": [],
        }
        with self.assertRaises(ValueError):
            summarize._normalise(raw, style="V", passages=passages)

    def test_placeholder_diagram_is_rejected(self) -> None:
        self.assertFalse(summarize._valid_mermaid("flowchart TD\n  A --> B"))
        self.assertFalse(summarize._valid_mermaid("not a diagram at all, just words"))
        self.assertFalse(summarize._valid_mermaid("flowchart TD\n  A[One] --> B[Two]"))
        self.assertTrue(
            summarize._valid_mermaid("flowchart TD\n  A[One] --> B[Two]\n  B --> C[Three]")
        )

    def test_angle_brackets_in_labels_are_rejected(self) -> None:
        """Mermaid's lexer reads them as HTML and aborts the whole diagram."""
        broken = "flowchart LR\n  A[Input] -->|<Weighted Sum>| B[Hidden]\n  B --> C[Output]\n  C --> A"
        self.assertFalse(summarize._valid_mermaid(broken))
        # <br> inside a label is the one legal exception, and arrows are made of
        # angle brackets, so neither may trip the rule.
        self.assertTrue(
            summarize._valid_mermaid(
                "flowchart TD\n  A[Step one<br/>then two] --> B[Next]\n  B -.-> C[Last]"
            )
        )

    def test_aural_report_drops_diagrams_even_if_the_model_sends_them(self) -> None:
        document = uploads.extract([("photosynthesis.md", MARKDOWN.encode())], self.tmp)
        passages = summarize.select_passages(document)
        raw = {
            "title": "t",
            "one_line": "o",
            "summary": "s" * 300,
            "sections": [{"heading": "h", "body": "b", "source": passages[0].source}],
            "key_terms": [],
            "diagrams": [{"title": "d", "caption": "", "mermaid": "flowchart TD\n A[One] --> B[Two]\n B --> C[Three]"}],
            "tables": [{"title": "t", "columns": ["a"], "rows": [["b"]]}],
            "checks": [],
            "study_plan": [],
        }
        report = summarize._normalise(raw, style="A", passages=passages)
        self.assertEqual(report.diagrams, [])
        self.assertEqual(report.tables, [])

    def test_the_fallback_diagram_belies_the_no_diagram_warning(self) -> None:
        """A visual report gains a chart, so it must stop claiming it has none."""
        document = uploads.extract([("photosynthesis.md", MARKDOWN.encode())], self.tmp)
        report = summarize.Report(
            title="t",
            one_line="o",
            summary="s" * 300,
            style="V",
            sections=[
                {"heading": "Light reactions", "body": "b", "source": "photosynthesis.md §Light reactions"},
                {"heading": "Calvin cycle", "body": "b", "source": "photosynthesis.md §Calvin cycle"},
            ],
            warnings=["This report still: There is no usable diagram. Produce at least one flowchart."],
        )
        finished = summarize._finalise(report, document, "V")
        self.assertEqual(len(finished.diagrams), 1)
        self.assertIn("headings", finished.diagrams[0]["caption"])
        self.assertEqual(finished.warnings, [])

    def test_quality_gate_catches_meta_prose_and_short_summaries(self) -> None:
        document = uploads.extract(course_files(), self.tmp)
        coverage = summarize.part_coverage(document)
        report = summarize.Report(
            title="t",
            one_line="o",
            summary="This report covers the basics. It is short.",
            style="V",
            sections=[],
            citations=[],
            diagrams=[{"title": "d", "caption": "", "mermaid": "flowchart TD\n A[One] --> B[Two]\n B --> C[Three]"}],
        )
        issues = summarize._quality_issues(report, document, "V", coverage)
        joined = " ".join(issues)
        self.assertIn("words", joined)
        self.assertIn("describes the report", joined)
        self.assertIn("nothing in the report comes from", joined.lower())


# --------------------------------------------------------------------------- #
class GenerationLoopTests(TempDirTest):
    def test_a_copying_draft_is_rewritten(self) -> None:
        document = uploads.extract([("photosynthesis.md", MARKDOWN.encode())], self.tmp)
        passages = summarize.select_passages(document)
        source_sentences = synthesis.sentences(" ".join(passage.text for passage in document.sections))

        copied = {
            "title": "Copied",
            "one_line": "o",
            "summary": " ".join(source_sentences)[:1200],
            "sections": [
                {"heading": "A", "body": " ".join(source_sentences)[:400], "source": passages[0].source},
                {"heading": "B", "body": " ".join(source_sentences)[:400], "source": passages[-1].source},
            ],
            "key_terms": [],
            "diagrams": [],
            "tables": [],
            "checks": [],
            "study_plan": [],
        }
        honest = {
            "title": "Plants and Light",
            "one_line": "How leaves store sunlight as sugar.",
            "summary": (
                "Leaves capture photons with chlorophyll and use that energy to split water, "
                "releasing oxygen as a side product. The captured energy then drives a cycle that "
                "fixes carbon dioxide into sugar, so light becomes stored chemical fuel. Together "
                "the two stages form one chain: capture, then build. "
            )
            * 3,
            "sections": [
                {"heading": "Capture", "body": "Pigments absorb light and split water.", "source": passages[0].source},
                {"heading": "Build", "body": "The stored energy fixes carbon into sugar.", "source": passages[-1].source},
            ],
            "key_terms": [{"term": "Thylakoid", "meaning": "The membrane where light is caught.", "source": passages[0].source}],
            "diagrams": [{"title": "Capture then build", "caption": "", "mermaid": "flowchart LR\n A[Light] --> B[Split water]\n B --> C[Fix carbon]"}],
            "tables": [],
            "checks": [],
            "study_plan": ["Draw the two stages"],
        }

        class Scripted:
            def __init__(self, script):
                self.script = list(script)
                self.calls = 0

            def chat(self, messages, **kwargs):
                self.calls += 1
                from varkcheck.llm import LLMReply, Usage

                item = self.script.pop(0)
                import json

                return LLMReply(text=json.dumps(item), usage=Usage(10, 10), model="scripted")

        llm = Scripted([copied, honest])
        report = summarize.generate(document, visual_profile(), "V", llm=llm, max_attempts=2)
        self.assertEqual(report.attempts, 2)
        self.assertTrue(report.synthesis_ok)
        self.assertTrue(any("rewritten" in note for note in report.notes))

    def test_the_better_of_two_bad_drafts_is_kept(self) -> None:
        """A rewrite that copies must not displace a genuine but imperfect draft."""
        document = uploads.extract([("photosynthesis.md", MARKDOWN.encode())], self.tmp)
        passages = summarize.select_passages(document)
        source_sentences = synthesis.sentences(" ".join(passage.text for passage in document.sections))

        # Draft one: our own words, but too short to be a summary of the material.
        thin = {
            "title": "Plants and Light",
            "one_line": "How leaves store sunlight as sugar.",
            "summary": "Leaves turn light into sugar through two linked stages.",
            "sections": [
                {"heading": "Capture", "body": "Pigments absorb light and split water.", "source": passages[0].source},
                {"heading": "Build", "body": "The stored energy fixes carbon into sugar.", "source": passages[-1].source},
            ],
            "key_terms": [],
            "diagrams": [{"title": "Capture then build", "caption": "", "mermaid": "flowchart LR\n A[Light] --> B[Split water]\n B --> C[Fix carbon]"}],
            "tables": [],
            "checks": [],
            "study_plan": [],
        }
        # Draft two: the retry, which simply pastes the source back.
        copied = {
            "title": "Copied",
            "one_line": "o",
            "summary": " ".join(source_sentences)[:1200],
            "sections": [
                {"heading": "A", "body": " ".join(source_sentences)[:400], "source": passages[0].source},
                {"heading": "B", "body": " ".join(source_sentences)[:400], "source": passages[-1].source},
            ],
            "key_terms": [],
            "diagrams": [],
            "tables": [],
            "checks": [],
            "study_plan": [],
        }

        llm = ScriptedLLM([thin, copied])
        report = summarize.generate(document, visual_profile(), "V", llm=llm, max_attempts=2)

        self.assertEqual(llm.calls, 2)
        self.assertEqual(report.title, "Plants and Light", "the honest draft should be the one shipped")
        self.assertTrue(report.synthesis_ok)
        self.assertLess(report.synthesis_ratio, synthesis.MAX_COPIED_RATIO)
        self.assertEqual(report.attempts, 2, "two drafts were written")
        process = " ".join(report.notes)
        self.assertIn("earlier draft", process)
        self.assertNotIn("was rewritten", process, "a draft that was kept was not rewritten")
        self.assertTrue(any("still" in warning for warning in report.warnings))

    def test_no_text_raises_a_clear_error(self) -> None:
        document = uploads.extract([("empty.md", b"")], self.tmp)
        with self.assertRaises(ValueError):
            summarize.generate(document, visual_profile(), "V", llm=object())


if __name__ == "__main__":
    unittest.main()
