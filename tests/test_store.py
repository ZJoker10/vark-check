"""SQLite memory: the tables the brief asks for, plus the answer-key vault."""

from __future__ import annotations

import unittest

from tests.support import TOPIC_RATE, EnvTestCase

from varkcheck import compare


class StudentAndAttemptTests(EnvTestCase):
    def test_upsert_is_idempotent_and_updates_declared(self):
        self.store.upsert_student("s1", {"V": 0.5})
        self.store.upsert_student("s1", {"V": 0.9})
        students = self.store.all_students()
        self.assertEqual(len(students), 1)
        self.assertEqual(self.store.get_student("s1")["declared"]["V"], 0.9)

    def test_attempts_are_scoped_by_student_and_topic(self):
        self.store.add_attempt("s1", TOPIC_RATE, "V", 2, 3)
        self.store.add_attempt("s1", "other topic", "A", 1, 3)
        self.store.add_attempt("s2", TOPIC_RATE, "R", 3, 3)
        self.assertEqual(len(self.store.attempts_for("s1")), 2)
        self.assertEqual(len(self.store.attempts_for("s1", TOPIC_RATE)), 1)

    def test_attempt_format_is_normalised(self):
        self.store.add_attempt("s1", TOPIC_RATE, "v", 2, 3)
        self.assertEqual(self.store.attempts_for("s1")[0]["format"], "V")

    def test_clear_demo_data_keeps_cache(self):
        self.store.upsert_student("s1")
        self.store.cache_set("k", {"value": 1})
        self.store.add_attempt("s1", TOPIC_RATE, "V", 1, 3)
        self.store.clear_demo_data(keep_cache=True)
        self.assertEqual(self.store.stats()["students"], 0)
        self.assertIsNotNone(self.store.cache_get("k"))


class QuizVaultTests(EnvTestCase):
    def test_answer_keys_are_stored_server_side(self):
        item_id = self.store.add_quiz_item(
            student_id="s1",
            topic=TOPIC_RATE,
            fmt="V",
            question="What happens when the learning rate is too large?",
            answer_key="The loss oscillates and then diverges.",
            concept="Learning rate failure modes",
            source="02_gradient_descent.md §2.3",
        )
        item = self.store.get_quiz_item(item_id)
        self.assertIsNotNone(item)
        self.assertIn("diverges", item["answer_key"])
        self.assertEqual(item["graded"], 0)

    def test_grading_marks_and_records(self):
        item_id = self.store.add_quiz_item(
            student_id="s1", topic=TOPIC_RATE, fmt="V", question="q", answer_key="a" * 30, concept="c"
        )
        self.store.mark_graded(item_id, "my answer", 1.5, "close but incomplete")
        item = self.store.get_quiz_item(item_id)
        self.assertEqual(item["graded"], 1)
        self.assertEqual(item["awarded"], 1.5)
        self.assertEqual(item["student_answer"], "my answer")

    def test_unknown_item_returns_none(self):
        self.assertIsNone(self.store.get_quiz_item("q_missing"))


class DigestQueryTests(EnvTestCase):
    def test_weakest_concept_ranks_first(self):
        for concept, awarded in (("Easy concept", 2.0), ("Hard concept", 0.5), ("Medium concept", 1.0)):
            item_id = self.store.add_quiz_item(
                student_id="s1",
                topic=TOPIC_RATE,
                fmt="R",
                question=f"q about {concept}",
                answer_key="something long enough to be a key",
                concept=concept,
                source="02_gradient_descent.md §2.3",
                max_score=2,
            )
            self.store.mark_graded(item_id, "answer", awarded, "feedback")

        digest = self.store.concept_digest(top_n=3)
        self.assertEqual(digest[0]["concept"], "Hard concept")
        self.assertGreater(digest[0]["weakness"], digest[-1]["weakness"])

    def test_flags_outrank_scores(self):
        self.store.add_flag("Frequently asked", "asked twice", "s1", TOPIC_RATE)
        self.store.add_flag("Frequently asked", "asked twice", "s2", TOPIC_RATE)
        item_id = self.store.add_quiz_item(
            student_id="s1", topic=TOPIC_RATE, fmt="R", question="q", answer_key="k" * 20,
            concept="Low score but no flags", max_score=2,
        )
        self.store.mark_graded(item_id, "a", 0.0, "f")

        digest = self.store.concept_digest(top_n=3)
        self.assertEqual(digest[0]["concept"], "Frequently asked")
        self.assertEqual(digest[0]["flags"], 2)


class HistoryTests(EnvTestCase):
    def test_history_is_chronological_and_limited(self):
        for index in range(6):
            self.store.add_message("ses1", "s1", "user", f"message {index}")
        history = self.store.history("ses1", limit=4)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[-1]["content"], "message 5")
        self.assertEqual(history[0]["content"], "message 2")


if __name__ == "__main__":
    unittest.main()
