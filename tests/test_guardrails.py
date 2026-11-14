"""The guardrails are code, so they get tests, not trust."""

from __future__ import annotations

import unittest

from tests.support import EnvTestCase  # noqa: F401 - ensures the repo is importable

from varkcheck import guardrails

ANSWER_REQUESTS = [
    "just give me the answers",
    "show me the answer key",
    "what is the answer to question 2?",
    "can you do my exercise for me",
    "solve this for me please",
    "which option is correct",
    "dammi le risposte del quiz",
    "fai tu il mio esercizio",
]

BENIGN = [
    "explain gradient descent to me",
    "I do not understand the learning rate, can you help",
    "what should I re-read before the exam?",
]


class InputGuardrailTests(unittest.TestCase):
    def test_answer_requests_are_blocked(self):
        for text in ANSWER_REQUESTS:
            with self.subTest(text=text):
                verdict = guardrails.check_student_input(text)
                self.assertFalse(verdict.allowed)
                self.assertEqual(verdict.kind, "answer_request")
                self.assertTrue(verdict.safe_reply)

    def test_benign_messages_pass(self):
        for text in BENIGN:
            with self.subTest(text=text):
                self.assertTrue(guardrails.check_student_input(text).allowed)

    def test_blocked_reply_does_not_contain_the_word_answer_key(self):
        verdict = guardrails.check_student_input("give me the answers")
        self.assertNotIn("answer key", verdict.safe_reply.lower())


class OutputGuardrailTests(unittest.TestCase):
    def test_labelling_is_blocked(self):
        phrases = [
            "You are a visual learner, so diagrams are your thing.",
            "Your learning style is kinesthetic.",
            "Since you are an aural learner, listen to the recording.",
            "Visual learners should stick to diagrams.",
        ]
        for text in phrases:
            with self.subTest(text=text):
                verdict = guardrails.check_agent_output(text)
                self.assertFalse(verdict.allowed)
                self.assertEqual(verdict.kind, "labelling")

    def test_neutral_comparison_wording_is_allowed(self):
        text = (
            "For this topic, your highest score was in the Read/write format (2.5/3). "
            "Your declared preference was Visual. Worth trying Read/write first next time."
        )
        self.assertTrue(guardrails.check_agent_output(text).allowed)

    def test_answer_key_leak_is_blocked(self):
        key = "It oscillates and then diverges instead of decreasing."
        leaky = f"The correct answer is: {key}"
        self.assertTrue(guardrails.leak_score(leaky, key))

        clean = "You identified oscillation but not divergence; re-read section 2.3."
        self.assertFalse(guardrails.leak_score(clean, key))

    def test_short_keys_do_not_trigger_false_positives(self):
        self.assertFalse(guardrails.leak_score("the answer is 4", "4"))

    def test_citation_required_when_asked(self):
        verdict = guardrails.check_agent_output(
            "Gradient descent repeats the update rule.", require_citation=True
        )
        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.kind, "missing_citation")

    def test_fabricated_citation_is_blocked(self):
        verdict = guardrails.check_agent_output(
            "Something invented. source: secret_notes.md §9.9",
            require_citation=True,
            allowed_docs=["02_gradient_descent.md"],
        )
        self.assertFalse(verdict.allowed)
        self.assertEqual(verdict.kind, "fabricated_citation")

    def test_valid_citation_passes(self):
        verdict = guardrails.check_agent_output(
            "The update rule is in 2.2. source: 02_gradient_descent.md §2.2 The update rule",
            require_citation=True,
            allowed_docs=["02_gradient_descent.md"],
        )
        self.assertTrue(verdict.allowed)

    def test_extract_citations(self):
        citations = guardrails.extract_citations(
            "a source: 02_gradient_descent.md §2.3 The learning rate\nb source: 03_neural_networks.md §3.4"
        )
        self.assertEqual(len(citations), 2)
        self.assertIn("02_gradient_descent.md", citations[0])


if __name__ == "__main__":
    unittest.main()
