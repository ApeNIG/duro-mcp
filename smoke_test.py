#!/usr/bin/env python3
"""
Duro Phase 1 Smoke Test Suite
Verifies the feedback loop is operational with both new and legacy payload formats.

Run: python smoke_test.py
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from artifacts import ArtifactStore

# Default paths (match duro_mcp_server.py)
DEFAULT_MEMORY_DIR = Path(os.path.expanduser("~/.agent/memory"))
DEFAULT_DB_PATH = DEFAULT_MEMORY_DIR / "index.db"

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "  ", "PASS": "[PASS] ", "FAIL": "[FAIL] ", "WARN": "[WARN] "}
    print(f"[{ts}] {prefix.get(level, '  ')}{msg}")

def assert_eq(actual, expected, msg: str):
    if actual != expected:
        log(f"{msg}: expected {expected}, got {actual}", "FAIL")
        return False
    log(f"{msg}: {actual}", "PASS")
    return True

def assert_close(actual, expected, tolerance: float, msg: str):
    if abs(actual - expected) > tolerance:
        log(f"{msg}: expected ~{expected}, got {actual}", "FAIL")
        return False
    log(f"{msg}: {actual}", "PASS")
    return True

def run_smoke_tests():
    """Run all smoke tests. Returns True if all pass."""
    store = ArtifactStore(DEFAULT_MEMORY_DIR, DEFAULT_DB_PATH)

    # Ensure index is up to date
    log("Rebuilding index...")
    indexed, errors = store.reindex()
    log(f"Indexed {indexed} artifacts ({errors} errors)")

    all_passed = True

    print("\n" + "="*60)
    print("DURO PHASE 1 SMOKE TEST SUITE")
    print("="*60 + "\n")

    # =========================================================
    # TEST 1: Full Episode Lifecycle (New Payload Format)
    # =========================================================
    print("\n--- TEST 1: Full Episode Lifecycle (New Format) ---\n")

    # Get baseline confidence
    baseline_artifact = store.get_artifact("ss_planning")
    if not baseline_artifact:
        log("ss_planning artifact not found - run seed first", "FAIL")
        return False
    baseline_conf = baseline_artifact["data"]["confidence"]
    log(f"Baseline confidence: {baseline_conf}")

    # Create episode
    success, ep_id, _ = store.store_episode(
        goal="Smoke test - new payload format",
        plan=["Create", "Close", "Evaluate", "Apply", "Verify"],
        tags=["smoke-test", "automated"],
        context={"test_type": "new_format"}
    )
    if not success:
        log(f"Failed to create episode: {ep_id}", "FAIL")
        return False
    log(f"Created episode: {ep_id}")

    # Add action
    success, msg = store.update_episode(ep_id, {
        "action": {"summary": "Automated smoke test action", "tool": "smoke_test.py"}
    })
    if not success:
        log(f"Failed to add action: {msg}", "FAIL")
        return False
    log("Added action to episode")

    # Close episode
    success, msg = store.update_episode(ep_id, {
        "status": "closed",
        "result": "success",
        "result_summary": "Smoke test completed successfully",
        "links": {"skills_used": ["planning"]}
    })
    if not success:
        log(f"Failed to close episode: {msg}", "FAIL")
        return False
    log("Closed episode")

    # Evaluate with NEW payload format (canonical)
    # Disable auto_skill_updates since we're explicitly providing updates
    success, eval_id, _ = store.store_evaluation(
        episode_id=ep_id,
        rubric={
            "outcome_quality": {"score": 5, "notes": "Automated smoke test"},
            "cost": {"tools_used": 1, "duration_mins": 0.1, "tokens_bucket": "XS"},
            "correctness_risk": {"score": 0, "notes": "Test only"},
            "reusability": {"score": 5, "notes": "Automated"},
            "reproducibility": {"score": 5, "notes": "Deterministic"}
        },
        grade="A",
        memory_updates={
            "reinforce": [{"type": "skill_stats", "id": "ss_planning", "delta": 0.01}],
            "decay": []
        },
        auto_skill_updates=False  # Explicit updates - disable auto
    )
    if not success:
        log(f"Failed to create evaluation: {eval_id}", "FAIL")
        return False
    log(f"Created evaluation: {eval_id}")

    # Apply evaluation
    success, msg, apply_result = store.apply_evaluation(eval_id)
    if not success:
        log(f"Apply failed: {msg}", "FAIL")
        all_passed = False
    else:
        log("Applied evaluation")

    # Verify confidence bump
    after_artifact = store.get_artifact("ss_planning")
    after_conf = after_artifact["data"]["confidence"]
    expected_conf = round(baseline_conf + 0.01, 2)

    if not assert_close(after_conf, expected_conf, 0.001, "Confidence after new format"):
        all_passed = False

    # =========================================================
    # TEST 2: Legacy Payload Format Compatibility
    # =========================================================
    print("\n--- TEST 2: Legacy Payload Format Compatibility ---\n")

    # Get new baseline (after test 1)
    baseline2 = store.get_artifact("ss_planning")["data"]["confidence"]
    log(f"Baseline confidence: {baseline2}")

    # Evaluate with LEGACY payload format
    # Disable auto_skill_updates since we're testing legacy format explicitly
    success, eval_id2, _ = store.store_evaluation(
        episode_id=ep_id,  # Reuse same episode
        rubric={
            "outcome_quality": {"score": 5, "notes": "Legacy compat test"},
            "cost": {"tools_used": 1, "duration_mins": 0.1, "tokens_bucket": "XS"},
            "correctness_risk": {"score": 0, "notes": "Format test"},
            "reusability": {"score": 4, "notes": "Compat check"},
            "reproducibility": {"score": 5, "notes": "Deterministic"}
        },
        grade="A",
        memory_updates={
            "reinforce": [{"artifact_id": "ss_planning", "reason": "legacy format test"}],
            "decay": []
        },
        auto_skill_updates=False  # Testing legacy format - disable auto
    )
    if not success:
        log(f"Failed to create evaluation: {eval_id2}", "FAIL")
        return False
    log(f"Created evaluation (legacy format): {eval_id2}")

    # Apply evaluation
    success, msg, apply_result2 = store.apply_evaluation(eval_id2)
    if not success:
        log(f"Apply failed: {msg}", "FAIL")
        all_passed = False
    else:
        log("Applied evaluation (legacy format)")

    # Check if normalization breadcrumb was added (check AFTER apply)
    eval_artifact = store.get_artifact(eval_id2)
    reinforce_items = eval_artifact["data"]["memory_updates"]["reinforce"]
    if reinforce_items:
        item = reinforce_items[0]
        if item.get("normalized_from") == "legacy":
            log(f"Normalization breadcrumb present: normalized_from='legacy'", "PASS")
            if "original_item" in item:
                log(f"Original item preserved: {item['original_item']}")
        else:
            log("Normalization breadcrumb missing", "FAIL")
            all_passed = False

    # Verify confidence bump with default delta
    after2 = store.get_artifact("ss_planning")["data"]["confidence"]
    expected2 = round(baseline2 + 0.01, 2)  # Default delta for legacy

    if not assert_close(after2, expected2, 0.001, "Confidence after legacy format"):
        all_passed = False

    # =========================================================
    # TEST 3: Skill Auto-Update from Episode Result
    # =========================================================
    print("\n--- TEST 3: Skill Auto-Update from Episode Result ---\n")

    # Ensure a fresh skill exists for this test
    store.ensure_skill_stats("auto_test", "Auto Test Skill", confidence=0.5)
    store.reindex()  # Make sure it's indexed

    baseline3 = store.get_artifact("ss_auto_test")["data"]["confidence"]
    log(f"Baseline confidence (ss_auto_test): {baseline3}")

    # Create episode with skills_used
    success, ep_id3, _ = store.store_episode(
        goal="Smoke test - skill auto-update",
        plan=["Test auto skill updates"],
        tags=["smoke-test", "auto-update"],
        context={"test_type": "auto_skill_update"}
    )
    if not success:
        log(f"Failed to create episode: {ep_id3}", "FAIL")
        return False
    log(f"Created episode: {ep_id3}")

    # Close episode with success and skills_used
    success, msg = store.update_episode(ep_id3, {
        "status": "closed",
        "result": "success",
        "result_summary": "Auto skill update test",
        "links": {"skills_used": ["auto_test"]}
    })
    if not success:
        log(f"Failed to close episode: {msg}", "FAIL")
        return False
    log("Closed episode with result=success, skills_used=[auto_test]")

    # Create evaluation WITHOUT explicit memory_updates (let auto-generation happen)
    success, eval_id3, _ = store.store_evaluation(
        episode_id=ep_id3,
        rubric={
            "outcome_quality": {"score": 5, "notes": "Auto skill update test"},
            "cost": {"tools_used": 1, "duration_mins": 0.1, "tokens_bucket": "XS"},
            "correctness_risk": {"score": 0, "notes": "Test only"},
            "reusability": {"score": 5, "notes": "Automated"},
            "reproducibility": {"score": 5, "notes": "Deterministic"}
        },
        grade="A"
        # No memory_updates provided - should auto-generate from episode
    )
    if not success:
        log(f"Failed to create evaluation: {eval_id3}", "FAIL")
        return False
    log(f"Created evaluation (no explicit memory_updates): {eval_id3}")

    # Check that auto-generated updates exist
    eval_artifact = store.get_artifact(eval_id3)
    auto_reinforce = eval_artifact["data"]["memory_updates"]["reinforce"]
    if auto_reinforce:
        auto_item = auto_reinforce[0]
        if auto_item.get("auto_generated"):
            log(f"Auto-generated update found: {auto_item['id']} delta={auto_item['delta']}", "PASS")
        else:
            log("Auto-generated flag missing on update", "WARN")
    else:
        log("No auto-generated updates found", "FAIL")
        all_passed = False

    # Apply evaluation
    success, msg, _ = store.apply_evaluation(eval_id3)
    if not success:
        log(f"Apply failed: {msg}", "FAIL")
        all_passed = False
    else:
        log("Applied evaluation")

    # Verify confidence bump (+0.01 for success)
    after3 = store.get_artifact("ss_auto_test")["data"]["confidence"]
    expected3 = round(baseline3 + 0.01, 2)

    if not assert_close(after3, expected3, 0.001, "Confidence after auto-update"):
        all_passed = False

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
        print(f"Final ss_planning confidence: {after2}")
    else:
        print("SOME TESTS FAILED")
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
