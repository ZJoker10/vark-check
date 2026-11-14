"""Retrieval must be grounded, citable and honest about what it does not know."""

from __future__ import annotations

import unittest

from tests.support import TOPIC_RATE, EnvTestCase

from varkcheck import rag


class ChunkingTests(EnvTestCase):
    def test_every_chunk_has_a_doc_and_a_section(self):
        self.assertGreater(len(self.index.chunks), 15)
        for chunk in self.index.chunks:
            self.assertTrue(chunk.doc.endswith(".md"), chunk.doc)
            self.assertTrue(chunk.section.strip())
            self.assertLessEqual(len(chunk.text), self.settings.chunk_chars + 200)

    def test_citations_name_the_file_and_section(self):
        citation = self.index.chunks[0].citation
        self.assertIn(".md", citation)
        self.assertIn("§", citation)

    def test_sections_are_usable_topics(self):
        sections = [entry["section"] for entry in self.index.sections()]
        self.assertGreaterEqual(len(sections), 10)
        self.assertIn(TOPIC_RATE, sections)


class SearchTests(EnvTestCase):
    def test_learning_rate_question_lands_in_lecture_two(self):
        results = self.index.search("what happens when the learning rate is too large", k=3)
        self.assertTrue(results)
        self.assertIn("02_gradient_descent.md", results[0].chunk.doc)
        self.assertGreater(results[0].score, 0)

    def test_backpropagation_question_lands_in_lecture_three(self):
        results = self.index.search("how does the chain rule work in backpropagation", k=3)
        self.assertTrue(results)
        self.assertIn("03_neural_networks.md", results[0].chunk.doc)

    def test_off_material_query_returns_nothing(self):
        self.assertEqual(self.index.search("lattice gauge renormalisation group", k=3), [])

    def test_passages_render_with_source_markers(self):
        rendered = rag.format_passages(self.index.search("mean squared error", k=2))
        self.assertIn("[source:", rendered)


class PersistenceTests(EnvTestCase):
    def test_index_round_trips_through_disk(self):
        saved = self.index.save(self.settings.index_dir)
        self.assertTrue(saved.exists())

        reloaded = rag.Index.load(self.settings.index_dir)
        self.assertEqual(len(reloaded.chunks), len(self.index.chunks))
        first = self.index.search("feature scaling", k=1)[0].chunk.id
        second = reloaded.search("feature scaling", k=1)[0].chunk.id
        self.assertEqual(first, second)

    def test_material_hash_changes_with_content(self):
        original = rag.material_hash(self.settings.course_dir)
        same = rag.material_hash(self.settings.course_dir)
        self.assertEqual(original, same)
        self.assertEqual(len(original), 16)


if __name__ == "__main__":
    unittest.main()
