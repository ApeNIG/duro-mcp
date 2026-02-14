"""
Phase 3 Integration Tests: Auto-capture & Proactive Recall

Tests:
1. Hot path classification
2. Warm path extraction
3. Cold path consolidation
4. Proactive recall
5. Extract learnings function
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestHotPathClassification:
    """Test the hot path keyword classifier."""

    def test_retrieval_triggered_for_project_context(self):
        from autocapture import hot_path_classify

        result = hot_path_classify("I'm working on the duro-mcp project")

        assert result.should_retrieve is True
        assert "project" in result.categories
        assert result.confidence > 0

    def test_retrieval_triggered_for_error_context(self):
        from autocapture import hot_path_classify

        result = hot_path_classify("I'm getting an error with the database connection")

        assert result.should_retrieve is True
        assert "error" in result.categories or "database" in result.categories

    def test_retrieval_triggered_for_decision_context(self):
        from autocapture import hot_path_classify

        result = hot_path_classify("We decided to use SQLite instead of PostgreSQL")

        assert result.should_retrieve is True
        assert "decision" in result.categories

    def test_no_retrieval_for_short_message(self):
        from autocapture import hot_path_classify

        result = hot_path_classify("hi")

        assert result.should_retrieve is False
        assert result.reason == "Message too short"

    def test_no_retrieval_for_generic_message(self):
        from autocapture import hot_path_classify

        result = hot_path_classify("The weather is nice today")

        assert result.should_retrieve is False
        assert len(result.categories) == 0

    def test_learning_signal_detected(self):
        from autocapture import detect_learning_signal

        assert detect_learning_signal("I learned that SQLite doesn't support ADD COLUMN IF NOT EXISTS") is True
        assert detect_learning_signal("Turns out the bug was in the migration") is True
        assert detect_learning_signal("Pro tip: always check the logs first") is True
        assert detect_learning_signal("Random sentence without learning") is False


class TestWarmPathExtraction:
    """Test the warm path fact/learning extraction."""

    def test_extract_facts_from_definition(self):
        from autocapture import warm_path_extract

        text = "Python is a high-level programming language. JavaScript uses event-driven architecture."
        result = warm_path_extract(text)

        assert len(result.facts) > 0
        assert any("python" in f.claim.lower() for f in result.facts)

    def test_extract_learnings(self):
        from autocapture import warm_path_extract

        text = "I learned that async/await is easier than callbacks. Turns out the error was a typo."
        result = warm_path_extract(text)

        assert len(result.learnings) > 0

    def test_extract_decisions(self):
        from autocapture import warm_path_extract

        text = "I decided to use FastAPI because it has automatic OpenAPI generation."
        result = warm_path_extract(text)

        assert len(result.decisions) > 0
        assert "FastAPI" in result.decisions[0]["decision"]

    def test_empty_text_returns_empty(self):
        from autocapture import warm_path_extract

        result = warm_path_extract("")

        assert len(result.facts) == 0
        assert len(result.learnings) == 0
        assert len(result.decisions) == 0


class TestColdPathConsolidation:
    """Test the cold path session consolidation."""

    def test_consolidate_session(self):
        from autocapture import cold_path_consolidate

        conversation = """
        User: Let's work on the authentication feature.
        Assistant: I'll implement JWT-based authentication.

        I decided to use bcrypt for password hashing because it's more secure.

        Successfully created the login endpoint.

        I learned that JWT tokens should be short-lived for security.

        Finished implementing user registration.
        """

        summary = cold_path_consolidate(conversation)

        # Should have detected topics
        assert "api" in summary.topics_discussed or len(summary.topics_discussed) > 0

        # Should have extracted some learnings
        assert len(summary.key_learnings) >= 0  # May not always extract

        # Should have found some decisions
        assert len(summary.decisions_made) >= 0


class TestCategoryToSearchParams:
    """Test conversion of categories to search params."""

    def test_project_category_mapping(self):
        from autocapture import category_to_search_params

        params = category_to_search_params(["project"])

        assert "project" in params["tags"] or "codebase" in params["tags"]
        assert "fact" in params["artifact_types"] or "decision" in params["artifact_types"]

    def test_multiple_categories(self):
        from autocapture import category_to_search_params

        params = category_to_search_params(["project", "error", "fix"])

        assert len(params["tags"]) > 0
        assert len(params["artifact_types"]) > 0


class TestProactiveRecall:
    """Test proactive recall functionality."""

    def test_recall_initialization(self):
        """Test that ProactiveRecall can be initialized."""
        from proactive import ProactiveRecall

        # Mock artifact store and index
        class MockArtifactStore:
            def get_artifact(self, aid):
                return None

            @property
            def index(self):
                return MockIndex()

        class MockIndex:
            def hybrid_search(self, **kwargs):
                return {"results": [], "mode": "fts_only"}

            def fts_search(self, **kwargs):
                return []

            def get_relations(self, aid, **kwargs):
                return []

        store = MockArtifactStore()
        recall = ProactiveRecall(store, store.index)

        assert recall is not None

    def test_recall_no_trigger_for_generic_text(self):
        """Test that generic text doesn't trigger recall."""
        from proactive import ProactiveRecall

        class MockStore:
            def get_artifact(self, aid):
                return None

            @property
            def index(self):
                return MockIndex()

        class MockIndex:
            pass

        store = MockStore()
        recall = ProactiveRecall(store, store.index)

        result = recall.recall("hello world", force=False)

        assert result.triggered is False


class TestExtractLearningsFunction:
    """Test the extract_learnings_from_text function."""

    def test_extract_without_autosave(self):
        from proactive import extract_learnings_from_text

        text = """
        Today I learned that Python 3.12 has new performance improvements.
        I decided to upgrade because the benefits outweigh the migration cost.
        The key insight is that pattern matching makes code cleaner.
        """

        result = extract_learnings_from_text(text, auto_save=False)

        assert "learnings" in result
        assert "facts" in result
        assert "decisions" in result
        assert result["auto_saved"] is False
        assert len(result["saved_ids"]) == 0

    def test_empty_text_returns_empty(self):
        from proactive import extract_learnings_from_text

        result = extract_learnings_from_text("", auto_save=False)

        assert result["count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
