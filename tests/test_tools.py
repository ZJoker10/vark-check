"""Tool contracts: grounded output, server-side keys, graceful failure."""

from __future__ import annotations

import unittest

from tests.support import TOPIC_BACKPROP, TOPIC_RATE, EnvTestCase

from varkcheck import tools
from varkcheck.llm import LLMReply, Usage
from varkcheck.offline import CannedToolLLM
from varkcheck.prompts import load_task_prompts


class LeakyGrader:
    """A deliberately bad model that quotes the answer key back at the student."""

    def __init__(self, key: str):
        self.key = key

    def chat(self, messages, **kwargs) -> LLMReply:
        return LLMReply(
            text=f'{{"score": 1, "why": "The correct answer is: {self.key}"}}',
            usage=Usage(10, 10),
            model="leaky",
        )


class ToolTestCase(EnvTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.llm = CannedToolLLM()
        self.ctx = tools.ToolContext(
            settings=self.settings,
            store=self.store,
            index=self.index,
            llm=self.llm,
            tasks=load_task_prompts(self.settings.prompt_path.parent / "tasks.md"),
            trace=self.env.trace,
        )


class SearchToolTests(ToolTestCase):
    def test_search_returns_citable_passages(self):
        result = tools.call_tool(self.ctx, "search_notes", {"query": "learning rate too large"})
        self.assertTrue(result.ok)
        self.assertTrue(result.citations)
        self.assertIn(result.citations[0].split(" §")[0], result.summary)

    def test_observation_is_capped(self):
        result = tools.call_tool(self.ctx, "search_notes", {"query": "gradient descent learning rate"})
        self.assertLessEqual(len(result.summary), tools.OBSERVATION_LIMIT)

    def test_unknown_topic_is_reported_not_invented(self):
        result = tools.call_tool(self.ctx, "search_notes", {"query": "lattice gauge renormalisation"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "no_material")


class FormatToolTests(ToolTestCase):
    def test_format_is_grounded_and_cited(self):
        result = tools.call_tool(self.ctx, "generate_format", {"topic": TOPIC_RATE, "format": "R"})
        self.assertTrue(result.ok, result.summary)
        self.assertIn("source:", result.data["text"])
        self.assertTrue(result.data["citations"])

    def test_visual_format_contains_a_diagram(self):
        result = tools.call_tool(self.ctx, "generate_format", {"topic": TOPIC_RATE, "format": "V"})
        self.assertTrue(result.ok)
        self.assertIn("```mermaid", result.data["text"])
        self.assertTrue(result.data["diagram_ok"])

    def test_second_call_is_served_from_cache(self):
        tools.call_tool(self.ctx, "generate_format", {"topic": TOPIC_RATE, "format": "A"})
        calls_after_first = len(self.llm.calls)
        tools.call_tool(self.ctx, "generate_format", {"topic": TOPIC_RATE, "format": "A"})
        self.assertEqual(len(self.llm.calls), calls_after_first)

    def test_missing_material_fails_loudly(self):
        result = tools.call_tool(self.ctx, "generate_format", {"topic": "quantum chromodynamics", "format": "V"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "no_material")

    def test_bad_format_is_rejected(self):
        result = tools.call_tool(self.ctx, "generate_format", {"topic": TOPIC_RATE, "format": "Z"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "bad_arguments")


class QuizToolTests(ToolTestCase):
    def test_quiz_items_hide_the_answer_key(self):
        result = tools.call_tool(
            self.ctx, "generate_quiz", {"topic": TOPIC_BACKPROP, "format": "R", "student_id": "s1"}
        )
        self.assertTrue(result.ok, result.summary)
        self.assertEqual(len(result.data["items"]), 3)
        for item in result.data["items"]:
            self.assertNotIn("answer_key", item)
            self.assertTrue(item["item_id"].startswith("q_"))

        stored = self.store.get_quiz_item(result.data["items"][0]["item_id"])
        self.assertTrue(stored["answer_key"])

    def test_question_never_contains_its_own_answer(self):
        result = tools.call_tool(
            self.ctx, "generate_quiz", {"topic": TOPIC_RATE, "format": "V", "student_id": "s1"}
        )
        for item in result.data["items"]:
            stored = self.store.get_quiz_item(item["item_id"])
            self.assertNotIn(stored["answer_key"].lower(), item["question"].lower())

    def test_quizzes_are_shared_across_students_but_items_are_per_student(self):
        first = tools.call_tool(
            self.ctx, "generate_quiz", {"topic": TOPIC_RATE, "format": "R", "student_id": "s1"}
        )
        second = tools.call_tool(
            self.ctx, "generate_quiz", {"topic": TOPIC_RATE, "format": "R", "student_id": "s2"}
        )
        self.assertEqual(
            [item["question"] for item in first.data["items"]],
            [item["question"] for item in second.data["items"]],
        )
        self.assertNotEqual(first.data["items"][0]["item_id"], second.data["items"][0]["item_id"])
        self.assertEqual(len(self.store.quiz_items_for("s1", TOPIC_RATE, "R")), 3)
        self.assertEqual(len(self.store.quiz_items_for("s2", TOPIC_RATE, "R")), 3)


class GradeToolTests(ToolTestCase):
    def _item(self, key: str = "The loss oscillates and then diverges instead of decreasing.") -> str:
        return self.store.add_quiz_item(
            student_id="s1",
            topic=TOPIC_RATE,
            fmt="V",
            question="What happens when the learning rate is too large?",
            answer_key=key,
            concept="Learning rate failure modes",
            source="02_gradient_descent.md §2.3",
            max_score=2,
        )

    def test_grading_scores_and_marks_the_item(self):
        item_id = self._item()
        result = tools.call_tool(
            self.ctx, "grade_answer", {"answers": [{"question_id": item_id, "answer": "It diverges."}]}
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.data["score"], 2.0)
        self.assertEqual(result.data["max_score"], 2.0)
        self.assertEqual(self.store.get_quiz_item(item_id)["graded"], 1)

    def test_empty_answer_costs_no_model_call(self):
        item_id = self._item()
        calls_before = len(self.llm.calls)
        result = tools.call_tool(
            self.ctx, "grade_answer", {"answers": [{"question_id": item_id, "answer": "   "}]}
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.data["score"], 0.0)
        self.assertEqual(len(self.llm.calls), calls_before)

    def test_answer_key_leak_in_feedback_is_replaced(self):
        key = "It oscillates and then diverges instead of decreasing."
        item_id = self._item(key)
        self.ctx.llm = LeakyGrader(key)

        result = tools.call_tool(
            self.ctx, "grade_answer", {"answers": [{"question_id": item_id, "answer": "It diverges."}]}
        )
        why = result.data["results"][0]["why"]
        self.assertIn("withheld", why.lower())
        self.assertNotIn("oscillates and then diverges", why.lower())

    def test_unknown_question_is_reported_not_guessed(self):
        result = tools.call_tool(
            self.ctx, "grade_answer", {"answers": [{"question_id": "q_missing", "answer": "x"}]}
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.data["results"][0]["error"], "unknown_question")

    def test_bad_payload_is_rejected(self):
        result = tools.call_tool(self.ctx, "grade_answer", {"answers": "not a list"})
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "bad_arguments")


class ProfileToolTests(ToolTestCase):
    def test_update_profile_records_and_summarises(self):
        result = tools.call_tool(
            self.ctx,
            "update_profile",
            {"student_id": "s1", "topic": TOPIC_RATE, "format": "V", "score": 2, "max_score": 3},
        )
        self.assertTrue(result.ok)
        self.assertIn("V 2.0/3", result.summary)
        self.assertEqual(len(self.store.attempts_for("s1", TOPIC_RATE)), 1)

    def test_flag_reaches_the_digest(self):
        tools.call_tool(
            self.ctx,
            "flag_for_lecturer",
            {"concept": "Feature scaling", "reason": "three students asked", "student_id": "s1", "topic": TOPIC_RATE},
        )
        concepts = [entry["concept"] for entry in self.store.concept_digest(top_n=3)]
        self.assertIn("Feature scaling", concepts)

    def test_schedule_review(self):
        result = tools.call_tool(
            self.ctx, "schedule_review", {"student_id": "s1", "topic": TOPIC_RATE, "days": 5}
        )
        self.assertTrue(result.ok)
        self.assertEqual(len(self.store.reviews_for("s1")), 1)


class RegistryTests(ToolTestCase):
    def test_unknown_tool_returns_error_instead_of_raising(self):
        result = tools.call_tool(self.ctx, "delete_everything", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "unknown_tool")

    def test_wrong_arguments_return_error_instead_of_raising(self):
        result = tools.call_tool(self.ctx, "generate_format", {"nonsense": True})
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "bad_arguments")

    def test_every_spec_has_a_handler(self):
        for spec in tools.TOOL_SPECS:
            self.assertIn(spec["name"], tools.HANDLERS)

    def test_tool_menu_mentions_every_tool(self):
        menu = tools.tool_menu()
        for spec in tools.TOOL_SPECS:
            self.assertIn(spec["name"], menu)


if __name__ == "__main__":
    unittest.main()
