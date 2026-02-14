"""
Quick Phase 3 smoke test without pytest dependency.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def test_autocapture_imports():
    """Test that autocapture module imports correctly."""
    print("Testing autocapture imports...")
    from autocapture import (
        hot_path_classify,
        warm_path_extract,
        cold_path_consolidate,
        detect_learning_signal,
        category_to_search_params
    )
    print("  [OK] All autocapture functions imported")
    return True


def test_proactive_imports():
    """Test that proactive module imports correctly."""
    print("Testing proactive imports...")
    from proactive import (
        ProactiveRecall,
        extract_learnings_from_text
    )
    print("  [OK] All proactive functions imported")
    return True


def test_hot_path_classification():
    """Test hot path classification."""
    print("Testing hot path classification...")
    from autocapture import hot_path_classify

    # Should trigger retrieval
    result = hot_path_classify("I'm working on the duro-mcp project")
    assert result.should_retrieve is True, "Expected retrieval trigger for project context"
    assert "project" in result.categories, "Expected 'project' category"
    print(f"  [OK] Project context: triggered={result.should_retrieve}, categories={result.categories}")

    # Should not trigger for short text
    result = hot_path_classify("hi")
    assert result.should_retrieve is False, "Expected no retrieval for short message"
    print(f"  [OK] Short message: triggered={result.should_retrieve}")

    # Should trigger for error context
    result = hot_path_classify("I'm getting an error with the database")
    assert result.should_retrieve is True, "Expected retrieval trigger for error context"
    print(f"  [OK] Error context: triggered={result.should_retrieve}, categories={result.categories}")

    return True


def test_learning_signal_detection():
    """Test learning signal detection."""
    print("Testing learning signal detection...")
    from autocapture import detect_learning_signal

    assert detect_learning_signal("I learned that X is Y") is True
    print("  [OK] 'learned' signal detected")

    assert detect_learning_signal("Turns out the bug was here") is True
    print("  [OK] 'turns out' signal detected")

    assert detect_learning_signal("Random text") is False
    print("  [OK] No signal in random text")

    return True


def test_warm_path_extraction():
    """Test warm path extraction."""
    print("Testing warm path extraction...")
    from autocapture import warm_path_extract

    text = """
    Python is a high-level programming language.
    I decided to use FastAPI because it has automatic OpenAPI generation.
    I learned that async/await simplifies code.
    """
    result = warm_path_extract(text)

    print(f"  [OK] Facts extracted: {len(result.facts)}")
    print(f"  [OK] Learnings extracted: {len(result.learnings)}")
    print(f"  [OK] Decisions extracted: {len(result.decisions)}")

    return True


def test_extract_learnings_function():
    """Test the extract_learnings_from_text function."""
    print("Testing extract_learnings_from_text...")
    from proactive import extract_learnings_from_text

    text = """
    Today I learned that Python 3.12 has new performance improvements.
    The key insight is that pattern matching makes code cleaner.
    """

    result = extract_learnings_from_text(text, auto_save=False)

    assert "learnings" in result
    assert "facts" in result
    assert "decisions" in result
    assert result["auto_saved"] is False

    print(f"  [OK] Result: {result['count']} items extracted")
    print(f"  [OK] Learnings: {result['learnings']}")

    return True


def test_category_mapping():
    """Test category to search params mapping."""
    print("Testing category to search params...")
    from autocapture import category_to_search_params

    params = category_to_search_params(["project", "error"])

    assert len(params["tags"]) > 0, "Expected tags"
    assert len(params["artifact_types"]) > 0, "Expected artifact types"

    print(f"  [OK] Tags: {params['tags']}")
    print(f"  [OK] Types: {params['artifact_types']}")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Phase 3 Quick Smoke Test")
    print("=" * 50)
    print()

    tests = [
        test_autocapture_imports,
        test_proactive_imports,
        test_hot_path_classification,
        test_learning_signal_detection,
        test_warm_path_extraction,
        test_extract_learnings_function,
        test_category_mapping,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  [FAIL] {test.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__} FAILED: {e}")
        print()

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
