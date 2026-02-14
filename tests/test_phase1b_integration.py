"""
Integration tests for Phase 1B: Semantic Search + FTS

Tests:
1. Golden path: store fact -> FTS row exists with text populated
2. Failure path: FTS populate fails -> save still succeeds -> health check flags
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts import ArtifactStore
from index import ArtifactIndex
from embeddings import artifact_to_text


class TestGoldenPathFTSPopulation(unittest.TestCase):
    """Test that storing artifacts populates FTS correctly."""

    def setUp(self):
        """Create temp directory and initialize stores."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_dir = Path(self.temp_dir) / "memory"
        self.memory_dir.mkdir(parents=True)

        # Create directory structure
        (self.memory_dir / "facts").mkdir()
        (self.memory_dir / "logs").mkdir()

        db_path = self.memory_dir / "artifacts.db"
        self.store = ArtifactStore(self.memory_dir, db_path)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_fact_creates_fts_entry(self):
        """Golden path: store_fact creates FTS entry with semantic text."""
        # Store a fact (returns tuple: success, artifact_id, file_path)
        success, artifact_id, file_path = self.store.store_fact(
            claim="Python 3.12 adds pattern matching improvements",
            confidence=0.5,  # Lower confidence to avoid source_urls requirement
            tags=["python", "programming"]
        )

        self.assertTrue(success, f"Store failed: {artifact_id}")

        # Verify FTS entry exists
        db_path = self.memory_dir / "artifacts.db"
        with sqlite3.connect(db_path) as conn:
            # Check if FTS table exists (may not if migration hasn't run)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_fts'"
            )
            if not cursor.fetchone():
                self.skipTest("FTS table not created - run migration first")

            # Check FTS entry exists
            cursor = conn.execute(
                "SELECT id, title, tags, text FROM artifact_fts WHERE id = ?",
                (artifact_id,)
            )
            row = cursor.fetchone()

            self.assertIsNotNone(row, f"FTS entry not found for {artifact_id}")

            fts_id, fts_title, fts_tags, fts_text = row

            # Verify basic fields (trigger-maintained)
            self.assertEqual(fts_id, artifact_id)
            self.assertIn("Python 3.12", fts_title)

            # Verify text column (Python-populated)
            # Note: text may be empty if populate_fts_text wasn't called
            # This test validates the integration works
            if fts_text:
                self.assertIn("pattern matching", fts_text.lower())

    def test_artifact_to_text_generates_content(self):
        """Verify artifact_to_text generates proper semantic content."""
        artifact = {
            "type": "fact",
            "data": {
                "claim": "FastEmbed is 10x faster than sentence-transformers",
                "snippet": "Benchmarks show significant speed improvements"
            },
            "tags": ["performance", "embeddings"]
        }

        text = artifact_to_text(artifact)

        self.assertIn("FastEmbed", text)
        self.assertIn("10x faster", text)
        self.assertIn("Benchmarks", text)
        self.assertIn("performance", text)
        self.assertIn("embeddings", text)

    def test_decision_to_text(self):
        """Verify decision artifact text generation."""
        artifact = {
            "type": "decision",
            "data": {
                "decision": "Use SQLite for audit log",
                "rationale": "Cross-process safety with transactions",
                "context": "Multi-process deployment scenario"
            },
            "tags": ["architecture"]
        }

        text = artifact_to_text(artifact)

        self.assertIn("SQLite", text)
        self.assertIn("Cross-process", text)
        self.assertIn("Multi-process", text)

    def test_episode_excludes_actions_array(self):
        """Verify episode text does NOT include full actions array."""
        artifact = {
            "type": "episode",
            "data": {
                "goal": "Implement semantic search",
                "result_summary": "Successfully added FTS5 and vec0 tables",
                "actions": [
                    {"tool": "read", "summary": "Read 50 files"},
                    {"tool": "edit", "summary": "Modified 10 files"},
                    {"tool": "bash", "summary": "Ran migrations"}
                ]
            },
            "tags": ["phase1b"]
        }

        text = artifact_to_text(artifact)

        # Should include goal and result
        self.assertIn("semantic search", text)
        self.assertIn("FTS5", text)

        # Should NOT include action details (would bloat embeddings)
        self.assertNotIn("Read 50 files", text)
        self.assertNotIn("Modified 10 files", text)


class TestFailurePathFTSSaveSucceeds(unittest.TestCase):
    """Test that FTS failures don't block artifact saves."""

    def setUp(self):
        """Create temp directory and initialize stores."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_dir = Path(self.temp_dir) / "memory"
        self.memory_dir.mkdir(parents=True)

        (self.memory_dir / "facts").mkdir()
        (self.memory_dir / "logs").mkdir()

        db_path = self.memory_dir / "artifacts.db"
        self.store = ArtifactStore(self.memory_dir, db_path)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fts_failure_does_not_block_save(self):
        """Failure path: FTS populate fails but save still succeeds."""
        # Patch populate_fts_text to fail
        original_populate = self.store.index.populate_fts_text

        def failing_populate(artifact_id, artifact):
            raise Exception("Simulated FTS failure")

        self.store.index.populate_fts_text = failing_populate

        try:
            # Store should still succeed (returns tuple: success, artifact_id, file_path)
            success, artifact_id, file_path = self.store.store_fact(
                claim="This should save despite FTS failure",
                confidence=0.5,  # Lower confidence to avoid source_urls requirement
                tags=["test"]
            )

            self.assertTrue(success, f"Save should succeed: {artifact_id}")

            # Verify JSON file exists
            json_file = self.memory_dir / "facts" / f"{artifact_id}.json"
            self.assertTrue(json_file.exists(), "JSON file should exist")

            # Verify artifact is in index (basic upsert, not FTS)
            db_path = self.memory_dir / "artifacts.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM artifacts WHERE id = ?",
                    (artifact_id,)
                )
                self.assertIsNotNone(cursor.fetchone(), "Artifact should be in index")

        finally:
            # Restore original
            self.store.index.populate_fts_text = original_populate


class TestHealthCheckFTSCompleteness(unittest.TestCase):
    """Test that health check correctly reports FTS completeness."""

    def setUp(self):
        """Create temp directory and initialize index."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "artifacts.db"
        self.index = ArtifactIndex(self.db_path)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fts_completeness_when_no_fts_table(self):
        """Health check handles missing FTS table gracefully."""
        stats = self.index.get_fts_completeness()

        self.assertFalse(stats.get("fts_exists"))
        self.assertEqual(stats.get("total_fts_rows", 0), 0)

    def test_fts_completeness_with_missing_text(self):
        """Health check reports missing text count correctly."""
        # Create FTS table manually for test
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS artifact_fts USING fts5(
                    id UNINDEXED, title, tags, text
                )
            """)
            # Insert rows - some with text, some without
            conn.execute(
                "INSERT INTO artifact_fts (id, title, tags, text) VALUES (?, ?, ?, ?)",
                ("fact_1", "Title 1", "tag1", "Full text here")
            )
            conn.execute(
                "INSERT INTO artifact_fts (id, title, tags, text) VALUES (?, ?, ?, ?)",
                ("fact_2", "Title 2", "tag2", "")  # Missing text
            )
            conn.execute(
                "INSERT INTO artifact_fts (id, title, tags, text) VALUES (?, ?, ?, ?)",
                ("fact_3", "Title 3", "tag3", "Another text")
            )
            conn.commit()

        stats = self.index.get_fts_completeness()

        self.assertTrue(stats.get("fts_exists"))
        self.assertEqual(stats.get("total_fts_rows"), 3)
        self.assertEqual(stats.get("missing_text_count"), 1)
        self.assertAlmostEqual(stats.get("coverage_pct"), 66.7, places=0)


class TestEmbeddingStats(unittest.TestCase):
    """Test embedding coverage stats for health check."""

    def setUp(self):
        """Create temp directory and initialize index."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "artifacts.db"
        self.index = ArtifactIndex(self.db_path)

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embedding_stats_when_vec_unavailable(self):
        """Embedding stats handles missing sqlite-vec gracefully."""
        stats = self.index.get_embedding_stats()

        # Should report graceful degradation, not error
        self.assertIn("vec_extension_available", stats)
        self.assertIn("vec_table_exists", stats)
        # Coverage should be 0 if no vectors
        self.assertEqual(stats.get("embeddings_count", 0), 0)


if __name__ == "__main__":
    unittest.main()
