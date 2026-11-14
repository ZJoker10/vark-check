"""Questionnaire scoring, the comparison logic, and the lecturer digest."""

from __future__ import annotations

import unittest

from tests.support import TOPIC_RATE, EnvTestCase

from varkcheck import compare, guardrails, questionnaire, seed


class QuestionnaireTests(unittest.TestCase):
    def test_two_items_per_dimension(self):
        counts = {}
        for question in questionnaire.QUESTIONS:
            counts[question.dimension] = counts.get(question.dimension, 0) + 1
        self.assertEqual(counts, {"V": 2, "A": 2, "R": 2, "K": 2})

    def test_scoring_prefers_the_answered_dimension(self):
        responses = {"q1": 5, "q2": 1, "q3": 1, "q4": 1, "q5": 5, "q6": 1, "q7": 1, "q8": 1}
        scores = questionnaire.declared_scores(responses)
        self.assertEqual(questionnaire.top_dimension(scores), "V")
        self.assertEqual(scores["V"], 1.0)
        self.assertEqual(scores["A"], 0.0)

    def test_incomplete_responses_are_not_marked_complete(self):
        self.assertFalse(questionnaire.is_complete({"q1": 3}))
        self.assertTrue(questionnaire.is_complete({q.id: 3 for q in questionnaire.QUESTIONS}))

    def test_labels_are_human_readable(self):
        self.assertEqual(questionnaire.label("R"), "Read/write")


class ComparisonTests(EnvTestCase):
    def _declared(self) -> dict[str, float]:
        return {"V": 0.9, "A": 0.4, "R": 0.3, "K": 0.5}

    def test_rows_aggregate_attempts_per_format(self):
        self.store.add_attempt("s1", TOPIC_RATE, "V", 2, 3)
        self.store.add_attempt("s1", TOPIC_RATE, "V", 1, 3)
        self.store.add_attempt("s1", TOPIC_RATE, "R", 3, 3)

        rows = compare.format_rows(self.store.attempts_for("s1", TOPIC_RATE))
        by_format = {row.format: row for row in rows}
        self.assertEqual(by_format["V"].mean, 1.5)
        self.assertEqual(by_format["V"].attempts, 2)
        self.assertAlmostEqual(by_format["R"].pct, 1.0)

    def test_best_format_uses_percentage_not_raw_score(self):
        self.store.add_attempt("s1", TOPIC_RATE, "V", 2, 4)
        self.store.add_attempt("s1", TOPIC_RATE, "A", 2, 3)
        rows = compare.format_rows(self.store.attempts_for("s1", TOPIC_RATE))
        self.assertEqual(compare.best_format(rows).format, "A")

    def test_message_is_neutral_and_passes_the_guardrail(self):
        self.store.add_attempt("s1", TOPIC_RATE, "V", 1, 3)
        self.store.add_attempt("s1", TOPIC_RATE, "R", 3, 3)
        rows = compare.format_rows(self.store.attempts_for("s1", TOPIC_RATE))
        message = compare.comparison_message(self._declared(), rows, TOPIC_RATE)

        self.assertIn("Read/write", message)
        self.assertIn("declared preference", message.lower())
        self.assertTrue(guardrails.check_agent_output(message).allowed, message)
        self.assertNotIn("you are a", message.lower())

    def test_empty_rows_still_return_something_sane(self):
        message = compare.comparison_message(self._declared(), [], TOPIC_RATE)
        self.assertIn("No attempts", message)

    def test_coverage_warning_lists_missing_formats(self):
        self.store.add_attempt("s1", TOPIC_RATE, "V", 3, 3)
        rows = compare.format_rows(self.store.attempts_for("s1", TOPIC_RATE))
        warning = compare.coverage_warning(rows)
        self.assertIsNotNone(warning)
        self.assertIn("Aural", warning)


class DigestTests(EnvTestCase):
    def test_synthetic_cohort_produces_three_ranked_concepts(self):
        seed.seed(self.store, count=10, seed_value=7)
        digest = compare.lecturer_digest(self.store, top_n=3)

        self.assertEqual(len(digest), 3)
        concepts = [entry["concept"] for entry in digest]
        self.assertEqual(concepts[0], "Chain rule in backpropagation")
        for entry in digest:
            self.assertIn(".md", entry["source"])
            self.assertGreaterEqual(entry["flags"], 0)

    def test_seeded_students_are_clearly_synthetic(self):
        seed.seed(self.store, count=4, seed_value=1)
        students = self.store.all_students()
        self.assertTrue(all(seed.is_synthetic(student["id"]) for student in students))

    def test_ensure_seeded_is_idempotent(self):
        self.assertTrue(seed.ensure_seeded(self.store, count=3))
        before = self.store.stats()["students"]
        self.assertFalse(seed.ensure_seeded(self.store, count=3))
        self.assertEqual(self.store.stats()["students"], before)


if __name__ == "__main__":
    unittest.main()
