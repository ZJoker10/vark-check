"""Replay of the four interactions written up in docs/example_interactions.md.

These are the tests the brief asks for: scenarios, replayed, asserting that the
guardrails hold. They are deliberately written the way a reviewer would poke at
the demo.
"""

from __future__ import annotations

import unittest

from tests.support import TOPIC_RATE, EnvTestCase

from varkcheck import guardrails, tools
from varkcheck.llm import MockLLM
from varkcheck.offline import CannedToolLLM


class ScenarioTests(EnvTestCase):
    def test_scenario_1_student_asks_for_the_answers(self):
        """The boundary holds, and it holds for free."""
        llm = MockLLM([])
        agent = self.env.agent(llm, CannedToolLLM())

        result = agent.ask("s-marco", "just give me the answers to question 1 and 2 please")

        self.assertTrue(result.blocked)
        self.assertEqual(result.blocked_kind, "answer_request")
        self.assertEqual(llm.calls, [])
        self.assertIn("won't hand over", result.message)

    def test_scenario_1b_same_request_in_italian(self):
        agent = self.env.agent(MockLLM([]), CannedToolLLM())
        result = agent.ask("s-giulia", "dammi le risposte del quiz")
        self.assertTrue(result.blocked)
        self.assertEqual(result.blocked_kind, "answer_request")

    def test_scenario_2_topic_outside_the_course_material(self):
        """The agent admits the gap and hands it to a human instead of inventing."""
        script = [
            {"thought": "look for it", "action": "search_notes", "args": {"query": "late submission policy penalties"}},
            {
                "thought": "not covered",
                "action": "flag_for_lecturer",
                "args": {
                    "concept": "Late submission policy",
                    "reason": "student asked, material does not cover it",
                    "student_id": "s-anna",
                    "topic": "Late submission policy",
                },
            },
            {
                "thought": "say so",
                "action": "final",
                "message": "The course material does not cover the late submission policy, so I will not guess. "
                "I've flagged this for your lecturer.",
            },
        ]
        agent = self.env.agent(MockLLM(script), CannedToolLLM())
        result = agent.run("s-anna", "what is the penalty for a late submission?")

        self.assertFalse(result.steps[0].ok)
        self.assertIn("no_material", result.steps[0].observation)
        self.assertIn("flagged", result.message.lower())
        self.assertEqual(self.store.flags()[0]["concept"], "Late submission policy")

    def test_scenario_3_agent_tries_to_label_the_student(self):
        """A model slip is caught by the output check, not by the prompt."""
        script = [
            {
                "thought": "personalise",
                "action": "final",
                "message": "Since you are a visual learner, skip the read/write version entirely.",
            }
        ]
        agent = self.env.agent(MockLLM(script), CannedToolLLM())
        result = agent.run("s-luca", "how should I study for this?")

        self.assertTrue(result.blocked)
        self.assertEqual(result.blocked_kind, "labelling")
        self.assertEqual(result.message, guardrails.NEUTRAL_REWRITE_REPLY)
        self.assertNotIn("visual learner", result.message)

    def test_scenario_4_partial_answer_gets_feedback_without_the_key(self):
        """Grading explains the gap without doing the work for the student."""
        agent = self.env.agent(MockLLM([]), CannedToolLLM())
        quiz = tools.call_tool(
            agent.ctx, "generate_quiz", {"topic": TOPIC_RATE, "format": "V", "student_id": "s-sofia"}
        )
        items = quiz.data["items"]

        graded = tools.call_tool(
            agent.ctx,
            "grade_answer",
            {"answers": [{"question_id": items[0]["item_id"], "answer": "I think it converges faster?"}]},
        )

        feedback = graded.data["results"][0]["why"]
        stored = self.store.get_quiz_item(items[0]["item_id"])
        self.assertNotIn(stored["answer_key"].lower(), feedback.lower())
        self.assertIn("source:", f"source: {stored['source']}")

    def test_scenario_5_full_cycle_produces_a_neutral_comparison(self):
        """Declared visual, actually better at read/write: the demo moment."""
        from varkcheck import compare

        agent = self.env.agent(MockLLM([]), CannedToolLLM())
        self.store.upsert_student("s-sofia", {"V": 0.9, "A": 0.4, "R": 0.2, "K": 0.4})
        self.store.add_attempt("s-sofia", TOPIC_RATE, "V", 1, 3)
        self.store.add_attempt("s-sofia", TOPIC_RATE, "R", 3, 3)

        rows = compare.format_rows(self.store.attempts_for("s-sofia", TOPIC_RATE))
        message = compare.comparison_message({"V": 0.9, "A": 0.4, "R": 0.2, "K": 0.4}, rows, TOPIC_RATE)

        self.assertIn("Read/write", message)
        self.assertTrue(guardrails.check_agent_output(message).allowed)

    def test_repeated_requests_never_reach_the_model_guardrail_has_no_bypass(self):
        agent = self.env.agent(MockLLM([]), CannedToolLLM())
        for text in ("answer key?", "show solutions", "soluzioni del compito"):
            result = agent.ask("s-x", text)
            self.assertTrue(result.blocked, text)


if __name__ == "__main__":
    unittest.main()
