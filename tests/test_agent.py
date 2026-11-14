"""The orchestrator loop: budgets, protocol failures, guardrails, fallbacks."""

from __future__ import annotations

import unittest

from tests.support import TOPIC_BACKPROP, TOPIC_RATE, EnvTestCase

from varkcheck import guardrails, tools
from varkcheck.llm import LLMError, MockLLM, Usage
from varkcheck.offline import CannedToolLLM

SEARCH = {"thought": "find the passage", "action": "search_notes", "args": {"query": "learning rate too large"}}
FORMAT = {
    "thought": "render it",
    "action": "generate_format",
    "args": {"topic": TOPIC_RATE, "format": "V"},
}
FINAL = {
    "thought": "done",
    "action": "final",
    "message": "Start with the right-hand branch. source: 02_gradient_descent.md §2.3 The learning rate and its failure modes",
}


class BrokenLLM:
    usage = Usage()

    def chat(self, messages, **kwargs):
        raise LLMError("provider unreachable")


class LoopTests(EnvTestCase):
    def test_plan_act_observe_until_final(self):
        agent = self.env.agent(MockLLM([SEARCH, FORMAT, FINAL]), CannedToolLLM())
        result = agent.run(
            "s1",
            "teach me the learning rate",
            require_citation=True,
            context={"allowed_docs": ["02_gradient_descent.md"]},
        )

        self.assertEqual(result.stopped_reason, "final")
        self.assertFalse(result.blocked)
        self.assertEqual(result.called, ["search_notes", "generate_format", "final"])
        self.assertEqual(len(result.steps), 3)
        self.assertTrue(result.steps[0].ok)
        self.assertIn("generate_format", result.data)
        self.assertTrue(result.data["generate_format"]["text"])

    def test_step_budget_hands_over_to_a_human(self):
        self.settings.max_steps = 3
        loop_forever = [SEARCH, SEARCH, SEARCH]
        agent = self.env.agent(MockLLM(loop_forever), CannedToolLLM())
        result = agent.run("s1", "teach me")

        self.assertTrue(result.handover)
        self.assertEqual(result.stopped_reason, "step_budget")
        self.assertEqual(result.message, guardrails.HANDOVER_REPLY)
        concepts = [flag["concept"] for flag in self.store.flags()]
        self.assertTrue(concepts)

    def test_token_budget_hands_over(self):
        self.settings.token_budget = 0
        agent = self.env.agent(MockLLM([SEARCH]), CannedToolLLM())
        result = agent.run("s1", "teach me")
        self.assertEqual(result.stopped_reason, "token_budget")
        self.assertTrue(result.handover)

    def test_provider_failure_hands_over_instead_of_crashing(self):
        agent = self.env.agent(BrokenLLM(), CannedToolLLM())
        result = agent.run("s1", "teach me")
        self.assertEqual(result.stopped_reason, "llm_error")
        self.assertTrue(result.handover)

    def test_two_protocol_failures_hand_over(self):
        agent = self.env.agent(MockLLM(["this is prose", "so is this"]), CannedToolLLM())
        result = agent.run("s1", "teach me")
        self.assertEqual(result.stopped_reason, "protocol_error")
        self.assertTrue(result.handover)

    def test_correcting_json_after_one_failure(self):
        agent = self.env.agent(MockLLM(["oops, prose", FINAL]), CannedToolLLM())
        result = agent.run("s1", "teach me")
        self.assertEqual(result.stopped_reason, "final")


class InputGuardrailTests(EnvTestCase):
    def test_answer_request_is_blocked_before_any_model_call(self):
        llm = MockLLM([FINAL])
        agent = self.env.agent(llm, CannedToolLLM())
        result = agent.ask("s1", "just give me the answers to the quiz")

        self.assertTrue(result.blocked)
        self.assertEqual(result.blocked_kind, "answer_request")
        self.assertEqual(result.stopped_reason, "input_guardrail")
        self.assertEqual(llm.calls, [])  # zero tokens spent
        self.assertIn("won't hand over", result.message)

    def test_blocked_requests_are_not_flagged_as_lecturer_problems(self):
        agent = self.env.agent(MockLLM([FINAL]), CannedToolLLM())
        agent.ask("s1", "do my exercise for me")
        self.assertEqual(self.store.flags(), [])


class OutputGuardrailTests(EnvTestCase):
    def test_labelling_reply_is_replaced(self):
        script = [{"thought": "answer", "action": "final", "message": "You are a visual learner, so only use diagrams."}]
        agent = self.env.agent(MockLLM(script), CannedToolLLM())
        result = agent.run("s1", "how should I study?")

        self.assertTrue(result.blocked)
        self.assertEqual(result.blocked_kind, "labelling")
        self.assertEqual(result.message, guardrails.NEUTRAL_REWRITE_REPLY)

    def test_uncited_content_is_replaced_when_citation_is_required(self):
        script = [{"thought": "answer", "action": "final", "message": "Gradient descent just works."}]
        agent = self.env.agent(MockLLM(script), CannedToolLLM())
        result = agent.ask("s1", "explain gradient descent")

        self.assertTrue(result.blocked)
        self.assertEqual(result.blocked_kind, "missing_citation")
        self.assertEqual(result.message, guardrails.UNGROUNDED_REPLY)


class FallbackTests(EnvTestCase):
    def test_deliver_falls_back_to_a_direct_tool_call(self):
        agent = self.env.agent(
            MockLLM(default={"thought": "nothing", "action": "final", "message": "I did nothing useful."}),
            CannedToolLLM(),
        )
        result = agent.deliver("s1", TOPIC_RATE, "R")

        self.assertTrue(result.fell_back)
        text = result.data.get("generate_format", {}).get("text", "")
        self.assertIn("source:", text)

    def test_quiz_falls_back_to_a_direct_tool_call(self):
        agent = self.env.agent(
            MockLLM(default={"thought": "nothing", "action": "final", "message": "nope"}),
            CannedToolLLM(),
        )
        result = agent.quiz("s1", TOPIC_BACKPROP, "V")
        self.assertTrue(result.fell_back)
        self.assertEqual(len(result.data["generate_quiz"]["items"]), 3)


class GradingFlowTests(EnvTestCase):
    def _items(self, agent) -> list[dict]:
        quiz = tools.call_tool(
            agent.ctx, "generate_quiz", {"topic": TOPIC_RATE, "format": "R", "student_id": "s1"}
        )
        return quiz.data["items"]

    def test_grade_then_profile_is_recorded(self):
        agent = self.env.agent(
            MockLLM([{"thought": "pending", "action": "final", "message": "placeholder"}]),
            CannedToolLLM(),
        )
        items = self._items(agent)
        answers = [{"question_id": item["item_id"], "answer": "diverges"} for item in items]

        agent = self.env.agent(
            MockLLM(
                [
                    {"thought": "grade", "action": "grade_answer", "args": {"answers": answers}},
                    {
                        "thought": "save",
                        "action": "update_profile",
                        "args": {
                            "student_id": "s1",
                            "topic": TOPIC_RATE,
                            "format": "R",
                            "score": 6,
                            "max_score": 6,
                        },
                    },
                    {"thought": "done", "action": "final", "message": "You scored 6/6."},
                ]
            ),
            CannedToolLLM(),
        )
        result = agent.grade("s1", TOPIC_RATE, "R", answers)

        self.assertFalse(result.fell_back)
        attempts = self.store.attempts_for("s1", TOPIC_RATE)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["score"], 6.0)

    def test_profile_is_written_even_if_the_model_forgets(self):
        agent = self.env.agent(
            MockLLM(
                [
                    {"thought": "grade", "action": "grade_answer", "args": {"answers": []}},
                    {"thought": "done", "action": "final", "message": "done"},
                ]
            ),
            CannedToolLLM(),
        )
        items = self._items(agent)
        answers = [{"question_id": item["item_id"], "answer": "diverges"} for item in items]

        result = agent.grade("s1", TOPIC_RATE, "R", answers)
        self.assertTrue(result.fell_back)
        self.assertEqual(len(self.store.attempts_for("s1", TOPIC_RATE)), 1)


if __name__ == "__main__":
    unittest.main()
