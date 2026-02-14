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
MIGRATIONS_DIR = Path(__file__).parent / "migrations"

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

def ensure_migrations_applied():
    """
    Check for pending migrations and auto-apply them.
    Returns (success, message).

    Control with env var:
      DURO_SMOKE_APPLY_MIGRATIONS=0  -> fail on pending (strict, good for CI)
      DURO_SMOKE_APPLY_MIGRATIONS=1  -> auto-apply (default, good for dev)
    """
    auto_apply = os.environ.get("DURO_SMOKE_APPLY_MIGRATIONS", "1") == "1"

    try:
        # Import from migrations package (stable API)
        from migrations import get_pending_migrations, run_all_pending

        # Log migrations dir for debugging wrong-cwd issues
        migration_files = list(MIGRATIONS_DIR.glob("m[0-9][0-9][0-9]_*.py"))
        log(f"Migrations dir: {MIGRATIONS_DIR} ({len(migration_files)} files)")

        pending = get_pending_migrations(MIGRATIONS_DIR, str(DEFAULT_DB_PATH))

        if not pending:
            return True, "All migrations applied"

        pending_ids = [m["migration_id"] for m in pending]

        if not auto_apply:
            # Strict mode: fail with clear instructions (exact command)
            return False, f"Pending migrations: {pending_ids}. Run: python -m migrations.runner \"{DEFAULT_DB_PATH}\" up"

        log(f"Pending migrations detected: {pending_ids}")
        log("Auto-applying migrations...")

        result = run_all_pending(MIGRATIONS_DIR, str(DEFAULT_DB_PATH), dry_run=False)

        if result["success"]:
            log(f"Applied migrations: {result['applied']}")
            return True, f"Applied {len(result['applied'])} migrations"
        else:
            return False, f"Migration failed: {result['failed']}"

    except ImportError as e:
        return False, f"Cannot import migrations module: {e}"
    except Exception as e:
        return False, f"Migration check error: {e}"


def run_smoke_tests():
    """Run all smoke tests. Returns True if all pass."""

    # Log the DB path being used (prevents wrong-DB debugging pain)
    log(f"Using DB: {DEFAULT_DB_PATH}")
    log(f"Memory dir: {DEFAULT_MEMORY_DIR}")

    # Ensure migrations are applied before testing
    migration_ok, migration_msg = ensure_migrations_applied()
    if not migration_ok:
        log(f"MIGRATION FAILED: {migration_msg}", "FAIL")
        log(f"Fix: python migrations/runner.py {DEFAULT_DB_PATH} up")
        return False
    log(migration_msg)

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

    # Verify confidence bump (capped at 0.99 ceiling)
    after_artifact = store.get_artifact("ss_planning")
    after_conf = after_artifact["data"]["confidence"]
    expected_conf = min(0.99, round(baseline_conf + 0.01, 2))

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

    # Verify confidence bump with default delta (capped at 0.99 ceiling)
    after2 = store.get_artifact("ss_planning")["data"]["confidence"]
    expected2 = min(0.99, round(baseline2 + 0.01, 2))  # Default delta for legacy

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

    # Verify confidence bump (+0.01 for success, capped at 0.99)
    after3 = store.get_artifact("ss_auto_test")["data"]["confidence"]
    expected3 = min(0.99, round(baseline3 + 0.01, 2))

    if not assert_close(after3, expected3, 0.001, "Confidence after auto-update"):
        all_passed = False

    # =========================================================
    # TEST 4: Guardrail - Block Updates for Skills Not in skills_used
    # =========================================================
    print("\n--- TEST 4: Guardrail - Block Unauthorized Skill Updates ---\n")

    # Get baseline for both skills
    baseline_planning = store.get_artifact("ss_planning")["data"]["confidence"]
    store.ensure_skill_stats("source_verification", "Source Verification", confidence=0.5)
    store.reindex()
    baseline_sv = store.get_artifact("ss_source_verification")["data"]["confidence"]

    log(f"Baseline ss_planning: {baseline_planning}")
    log(f"Baseline ss_source_verification: {baseline_sv}")

    # Create episode with skills_used=["planning"] only
    success, ep_id4, _ = store.store_episode(
        goal="Smoke test - guardrail verification",
        plan=["Test guardrail blocks unauthorized updates"],
        tags=["smoke-test", "guardrail"],
        context={"test_type": "guardrail_test"}
    )
    if not success:
        log(f"Failed to create episode: {ep_id4}", "FAIL")
        return False
    log(f"Created episode: {ep_id4}")

    # Close with skills_used=["planning"] - NOT source_verification
    success, msg = store.update_episode(ep_id4, {
        "status": "closed",
        "result": "success",
        "result_summary": "Guardrail test",
        "links": {"skills_used": ["planning"]}  # Only planning!
    })
    if not success:
        log(f"Failed to close episode: {msg}", "FAIL")
        return False
    log("Closed episode with skills_used=[planning] only")

    # Try to sneak in an update for ss_source_verification (should be blocked)
    success, eval_id4, _ = store.store_evaluation(
        episode_id=ep_id4,
        rubric={
            "outcome_quality": {"score": 5, "notes": "Guardrail test"},
            "cost": {"tools_used": 1, "duration_mins": 0.1, "tokens_bucket": "XS"},
            "correctness_risk": {"score": 0, "notes": "Test only"},
            "reusability": {"score": 5, "notes": "Automated"},
            "reproducibility": {"score": 5, "notes": "Deterministic"}
        },
        grade="A",
        memory_updates={
            # Trying to update source_verification - should be BLOCKED by guardrail
            "reinforce": [{"type": "skill_stats", "id": "ss_source_verification", "delta": 0.02}],
            "decay": []
        }
        # auto_skill_updates=True (default) - should auto-add planning
    )
    if not success:
        log(f"Failed to create evaluation: {eval_id4}", "FAIL")
        return False
    log(f"Created evaluation with sneaky ss_source_verification update: {eval_id4}")

    # Check what actually made it into the evaluation
    eval_artifact = store.get_artifact(eval_id4)
    reinforce_items = eval_artifact["data"]["memory_updates"]["reinforce"]
    ids_in_updates = [item.get("id") for item in reinforce_items]

    # ss_source_verification should NOT be in the updates (guardrail filtered it)
    if "ss_source_verification" in ids_in_updates:
        log("GUARDRAIL FAILED: ss_source_verification was NOT filtered out", "FAIL")
        all_passed = False
    else:
        log("Guardrail filtered out ss_source_verification (not in skills_used)", "PASS")

    # ss_planning SHOULD be in the updates (auto-generated)
    if "ss_planning" in ids_in_updates:
        log("Auto-generated ss_planning update present", "PASS")
    else:
        log("Auto-generated ss_planning update missing", "FAIL")
        all_passed = False

    # Apply and verify
    success, msg, _ = store.apply_evaluation(eval_id4)
    if not success:
        log(f"Apply failed: {msg}", "FAIL")
        all_passed = False
    else:
        log("Applied evaluation")

    # Verify ss_planning got +0.01 (capped at 0.99)
    after_planning = store.get_artifact("ss_planning")["data"]["confidence"]
    expected_planning = min(0.99, round(baseline_planning + 0.01, 2))
    if not assert_close(after_planning, expected_planning, 0.001, "ss_planning confidence (should increase)"):
        all_passed = False

    # Verify ss_source_verification is UNCHANGED (guardrail worked)
    after_sv = store.get_artifact("ss_source_verification")["data"]["confidence"]
    if not assert_eq(after_sv, baseline_sv, "ss_source_verification confidence (should be unchanged)"):
        all_passed = False

    # =========================================================
    # TEST 5: Decision Outcomes - Validate and Link to Episode
    # =========================================================
    print("\n--- TEST 5: Decision Outcomes Validation ---\n")

    # Create a decision with initial confidence
    decision_id = f"dec_smoke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    success, dec_id, _ = store.store_decision(
        decision="Use async processing for smoke test batch jobs",
        rationale="Async improves throughput without blocking the main thread",
        context="smoke test decision outcomes",
        alternatives=["sync processing", "thread pool"],
        tags=["smoke-test", "decision-outcome"],
        reversible=True,
        sensitivity="internal"
    )
    if not success:
        log(f"Failed to create decision: {dec_id}", "FAIL")
        return False
    log(f"Created decision: {dec_id}")

    # Get initial confidence
    dec_artifact = store.get_artifact(dec_id)
    initial_conf = dec_artifact["data"].get("outcome", {}).get("confidence", 0.5)
    log(f"Initial decision confidence: {initial_conf}")

    # Create episode that uses this decision
    success, ep_id5, _ = store.store_episode(
        goal="Smoke test - decision validation",
        plan=["Make decision", "Execute", "Validate outcome"],
        tags=["smoke-test", "decision-outcome"],
        context={"decision_under_test": dec_id}
    )
    if not success:
        log(f"Failed to create episode: {ep_id5}", "FAIL")
        return False
    log(f"Created episode: {ep_id5}")

    # Close episode with success
    success, msg = store.update_episode(ep_id5, {
        "status": "closed",
        "result": "success",
        "result_summary": "Decision validated - async processing worked well",
        "links": {"skills_used": ["planning"], "decisions_created": [dec_id]}
    })
    if not success:
        log(f"Failed to close episode: {msg}", "FAIL")
        return False
    log("Closed episode with success")

    # Validate the decision with episode evidence
    success, msg = store.validate_decision(
        decision_id=dec_id,
        status="validated",
        episode_id=ep_id5,
        result="success",
        notes="Async processing confirmed to improve throughput in smoke test"
    )
    if not success:
        log(f"Failed to validate decision: {msg}", "FAIL")
        all_passed = False
    else:
        log(f"Validated decision: {msg}")

    # Verify confidence increased (validated = +0.1)
    dec_after = store.get_artifact(dec_id)
    final_conf = dec_after["data"]["outcome"]["confidence"]
    expected_conf5 = min(0.99, initial_conf + 0.1)

    if not assert_close(final_conf, expected_conf5, 0.01, "Decision confidence after validation"):
        all_passed = False

    # Verify outcome status is now "validated"
    if not assert_eq(dec_after["data"]["outcome"]["status"], "validated", "Decision outcome status"):
        all_passed = False

    # Verify episode is linked
    episodes_used = dec_after["data"].get("episodes_used", [])
    if ep_id5 in episodes_used:
        log(f"Episode linked to decision: {ep_id5}", "PASS")
    else:
        log(f"Episode NOT linked to decision", "FAIL")
        all_passed = False

    # =========================================================
    # TEST 6: Decision Outcomes via Evaluation (Auto-Update)
    # =========================================================
    print("\n--- TEST 6: Decision Outcomes via Evaluation (Auto-Update) ---\n")

    # Create a decision to be used in an episode
    success, dec_id6, _ = store.store_decision(
        decision="Use semantic search for improved recall",
        rationale="Hybrid FTS + embeddings provides best of both worlds",
        context="smoke test auto decision outcomes",
        alternatives=["FTS only", "embeddings only"],
        tags=["smoke-test", "decision-auto-outcome"],
        reversible=True,
        sensitivity="internal"
    )
    if not success:
        log(f"Failed to create decision: {dec_id6}", "FAIL")
        return False
    log(f"Created decision: {dec_id6}")

    # Get initial confidence
    dec_before = store.get_artifact(dec_id6)
    initial_conf6 = dec_before["data"].get("outcome", {}).get("confidence", 0.5)
    log(f"Initial decision confidence: {initial_conf6}")

    # Create episode that USES this decision (decisions_used, not decisions_created)
    success, ep_id6, _ = store.store_episode(
        goal="Smoke test - decision auto-outcome via evaluation",
        plan=["Apply decision", "Measure results"],
        tags=["smoke-test", "decision-auto-outcome"],
        context={"testing_auto_decision_outcome": True}
    )
    if not success:
        log(f"Failed to create episode: {ep_id6}", "FAIL")
        return False
    log(f"Created episode: {ep_id6}")

    # Close episode with decisions_used (past decision relied upon)
    success, msg = store.update_episode(ep_id6, {
        "status": "closed",
        "result": "success",
        "result_summary": "Semantic search improved recall as expected",
        "links": {"skills_used": ["planning"], "decisions_used": [dec_id6]}
    })
    if not success:
        log(f"Failed to close episode: {msg}", "FAIL")
        return False
    log("Closed episode with decisions_used")

    # Create evaluation - should auto-generate decision update
    success, eval_id6, _ = store.store_evaluation(
        episode_id=ep_id6,
        rubric={
            "outcome_quality": {"score": 5, "notes": "Decision proved correct"},
            "cost": {"tools_used": 1, "duration_mins": 0.1, "tokens_bucket": "XS"},
            "correctness_risk": {"score": 0, "notes": "Verified outcome"},
            "reusability": {"score": 5, "notes": "Pattern reusable"},
            "reproducibility": {"score": 5, "notes": "Deterministic"}
        },
        grade="A"
        # Let auto_skill_updates and auto decision updates happen
    )
    if not success:
        log(f"Failed to create evaluation: {eval_id6}", "FAIL")
        return False
    log(f"Created evaluation: {eval_id6}")

    # Check that decision update was auto-generated
    eval_artifact6 = store.get_artifact(eval_id6)
    reinforce_items6 = eval_artifact6["data"]["memory_updates"]["reinforce"]
    decision_ids_in_updates = [item.get("id") for item in reinforce_items6 if item.get("type") == "decision"]

    if dec_id6 in decision_ids_in_updates:
        log(f"Auto-generated decision update for {dec_id6}", "PASS")
    else:
        log(f"Decision update NOT auto-generated (expected {dec_id6})", "FAIL")
        all_passed = False

    # Apply evaluation
    success, msg, applied = store.apply_evaluation(eval_id6)
    if not success:
        log(f"Apply failed: {msg}", "FAIL")
        all_passed = False
    else:
        log(f"Applied evaluation: {msg}")

    # Verify decision confidence increased (+0.005 for success)
    dec_after6 = store.get_artifact(dec_id6)
    final_conf6 = dec_after6["data"]["outcome"]["confidence"]
    expected_conf6 = initial_conf6 + 0.005

    if not assert_close(final_conf6, expected_conf6, 0.0001, "Decision confidence after success"):
        all_passed = False

    # Verify status is still "unverified" (below 0.7 threshold)
    status6 = dec_after6["data"]["outcome"]["status"]
    if status6 == "unverified":
        log(f"Decision status is 'unverified' (confidence {final_conf6} < 0.7)", "PASS")
    else:
        log(f"Decision status unexpected: {status6}", "WARN")

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
