"""Shared test harness.

Everything is built in a temporary directory against the *real* course material,
so the tests fail if the demo material ever stops being indexable.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from varkcheck import rag  # noqa: E402
from varkcheck.agent import Agent  # noqa: E402
from varkcheck.config import Settings  # noqa: E402
from varkcheck.store import Store  # noqa: E402
from varkcheck.trace import Trace  # noqa: E402

TOPIC_RATE = "2.3 The learning rate and its failure modes"
TOPIC_BACKPROP = "3.5 The backpropagation algorithm"


def build_settings(root: Path) -> Settings:
    settings = Settings(
        provider="mock",
        base_url="mock://offline",
        api_key="",
        model="mock-1",
        temperature=0.0,
        max_output_tokens=400,
        max_steps=6,
        token_budget=25000,
        top_k=4,
        chunk_chars=1200,
        min_similarity=0.02,
        course_dir=REPO / "data" / "course",
        data_dir=root / "data",
        db_path=root / "data" / "test.db",
        index_dir=root / "data" / "index",
        log_dir=root / "logs",
        prompt_path=REPO / "prompts" / "system.md",
    )
    settings.ensure_dirs()
    return settings


class Env:
    """One throwaway environment: settings, store, index, trace, agent factory."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="varkcheck-test-"))
        self.settings = build_settings(self.root)
        self.store = Store(self.settings.db_path)
        self.index, _ = rag.load_or_build(
            self.settings.course_dir, self.settings.index_dir, chunk_chars=self.settings.chunk_chars
        )
        self.trace = Trace(self.settings.log_dir, run_id="test")

    def agent(self, llm, tool_llm=None) -> Agent:
        return Agent(
            self.settings,
            self.store,
            self.index,
            llm=llm,
            tool_llm=tool_llm,
            trace=self.trace,
        )

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class EnvTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.env = Env()
        self.settings = self.env.settings
        self.store = self.env.store
        self.index = self.env.index

    def tearDown(self) -> None:
        self.env.close()
