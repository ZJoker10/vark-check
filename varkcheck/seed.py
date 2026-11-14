"""Synthetic demo cohort.

The lecturer view is meaningless with one student, so the demo ships with ten
fake ones. Everything here is clearly marked synthetic: the ids start with
`syn-`, and the UI shows a badge. No real student data is ever generated.

Scores are drawn from a fixed seed, so the lecturer digest is identical on every
machine and every run — which matters when you are recording a demo video.
"""

from __future__ import annotations

import random
from typing import Any

from .questionnaire import QUESTIONS, declared_scores
from .store import Store

SYNTHETIC_PREFIX = "syn-"

# Deliberately ordered: the first three are the ones the cohort "has not grasped".
CONCEPTS: list[dict[str, Any]] = [
    {
        "concept": "Chain rule in backpropagation",
        "source": "03_neural_networks.md §3.4 Loss and the chain rule",
        "weakness": 0.14,
        "flags": 5,
        "question": "In one sentence, why does backpropagation need the derivative of the loss?",
        "answer_key": "Because each parameter's gradient is the loss derivative multiplied along the path back to that parameter.",
    },
    {
        "concept": "Learning rate failure modes",
        "source": "02_gradient_descent.md §2.3 The learning rate and its failure modes",
        "weakness": 0.2,
        "flags": 4,
        "question": "What happens to the loss curve when the learning rate is too large?",
        "answer_key": "It oscillates and then diverges instead of decreasing.",
    },
    {
        "concept": "Feature scaling",
        "source": "02_gradient_descent.md §2.5 Why feature scaling matters",
        "weakness": 0.24,
        "flags": 3,
        "question": "Why does unscaled data make one learning rate a bad compromise?",
        "answer_key": "It makes the loss surface a long narrow valley, so one step size is either too small for one direction or too large for the other.",
    },
    {
        "concept": "Overfitting vs underfitting",
        "source": "01_linear_regression.md §1.6 Overfitting and underfitting",
        "weakness": 0.38,
        "flags": 0,
        "question": "Which error gap diagnoses overfitting?",
        "answer_key": "Low training error with high validation error.",
    },
    {
        "concept": "Activation function choice",
        "source": "03_neural_networks.md §3.2 Layers and activations",
        "weakness": 0.42,
        "flags": 0,
        "question": "What happens if you stack linear layers with no activation?",
        "answer_key": "The composition is still a single linear map, so depth buys nothing.",
    },
    {
        "concept": "Squared error rationale",
        "source": "01_linear_regression.md §1.4 Why squared error and not absolute error",
        "weakness": 0.46,
        "flags": 0,
        "question": "Give one reason MSE is preferred over MAE for gradient descent.",
        "answer_key": "MSE is smooth and differentiable everywhere, so the gradient is well defined.",
    },
]

TOPICS = [
    "1.6 Overfitting and underfitting",
    "2.3 The learning rate and its failure modes",
    "2.5 Why feature scaling matters",
    "3.2 Layers and activations",
    "3.5 The backpropagation algorithm",
]

FORMATS = ("V", "A", "R", "K")


def _rand_responses(rng: random.Random) -> dict[str, int]:
    return {question.id: rng.randint(2, 5) for question in QUESTIONS}


def is_synthetic(student_id: str) -> bool:
    return str(student_id).startswith(SYNTHETIC_PREFIX)


def seed(store: Store, *, count: int = 10, seed_value: int = 7) -> dict[str, int]:
    """Insert (or refresh) the synthetic cohort. Idempotent per student id."""
    rng = random.Random(seed_value)
    created = 0

    for index in range(1, count + 1):
        student_id = f"{SYNTHETIC_PREFIX}{index:02d}"
        responses = _rand_responses(rng)
        store.upsert_student(student_id, declared_scores(responses))
        created += 1

        # Attempts: 2–3 topics, a random subset of formats, scores around 0.45–0.85.
        topic_count = rng.randint(2, 3)
        for topic in rng.sample(TOPICS, k=topic_count):
            chosen = rng.sample(FORMATS, k=rng.randint(2, 4))
            for fmt in chosen:
                pct = min(0.95, max(0.1, rng.gauss(0.62, 0.18)))
                store.add_attempt(student_id, topic, fmt, round(3 * pct, 1), 3)

        # Concept-level quiz results drive the lecturer digest. Weakness values
        # come from CONCEPTS so the top 3 are stable and demonstrable.
        for concept in CONCEPTS:
            if rng.random() < 0.35:
                continue  # not every student answered every question
            max_score = 2.0
            noise = rng.gauss(0.0, 0.12)
            pct = min(1.0, max(0.0, concept["weakness"] + noise))
            item_id = store.add_quiz_item(
                student_id=student_id,
                topic=concept["source"],
                fmt="R",
                question=concept["question"],
                answer_key=concept["answer_key"],
                concept=concept["concept"],
                source=concept["source"],
                rubric="Mentions the mechanism described in the section.",
                max_score=max_score,
            )
            store.mark_graded(
                item_id,
                answer="(synthetic answer)",
                awarded=round(max_score * pct, 1),
                feedback="Synthetic feedback: see the source section.",
            )

    # Explicit flags, so the digest ranking has a human signal as well as scores.
    for concept in CONCEPTS:
        for _ in range(concept["flags"]):
            store.add_flag(
                concept=concept["concept"],
                reason="Students asked about this after the quiz (synthetic).",
                student_id=f"{SYNTHETIC_PREFIX}{rng.randint(1, count):02d}",
                topic=concept["source"],
            )

    return {"students": created, "concepts": len(CONCEPTS)}


def ensure_seeded(store: Store, count: int = 10, seed_value: int = 7) -> bool:
    """Seed only when there is no synthetic data yet."""
    existing = [student for student in store.all_students() if is_synthetic(student["id"])]
    if existing:
        return False
    seed(store, count=count, seed_value=seed_value)
    return True
