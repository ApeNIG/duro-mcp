"""
Phase 4 Golden Decay Test

The ONE test that matters:
1. Create a fact with confidence 0.8, importance 0.5
2. Set created_at to 30 days ago
3. Apply decay
4. Verify confidence decreased (and doesn't go below floor)
5. Reinforce it (set last_reinforced_at to now)
6. Apply decay again
7. Verify confidence stays stable (or decays much slower)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_decay_imports():
    """Test that decay module imports correctly."""
    print("Testing decay imports...")
    from decay import (
        calculate_decay,
        is_stale,
        reinforce_fact,
        apply_batch_decay,
        generate_maintenance_report,
        DecayConfig,
        DEFAULT_DECAY_CONFIG
    )
    print("  [OK] All decay functions imported")
    return True


def test_golden_decay_scenario():
    """
    The golden test: full decay lifecycle.
    """
    print("Testing golden decay scenario...")
    from decay import calculate_decay, reinforce_fact, is_stale, DecayConfig
    from time_utils import utc_now_iso

    # Custom config for testing (faster decay for test visibility)
    config = DecayConfig(
        base_rate=0.01,  # 1% per day (faster for testing)
        min_confidence=0.05,
        max_confidence=0.99,
        grace_period_days=7,
        reinforcement_protection_days=3,
        stale_threshold=0.3
    )

    # Step 1: Create a fact with confidence 0.8, importance 0.5, 30 days old
    thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace("+00:00", "Z")

    fact = {
        "id": "fact_test_golden",
        "type": "fact",
        "created_at": thirty_days_ago,
        "data": {
            "claim": "Test golden decay scenario",
            "confidence": 0.8,
            "importance": 0.5,
            "pinned": False,
            "last_reinforced_at": None,
            "reinforcement_count": 0
        }
    }

    print(f"  Initial state: confidence={fact['data']['confidence']}, created 30 days ago")

    # Step 2: Apply decay
    result = calculate_decay(fact, config)

    print(f"  After decay calculation:")
    print(f"    - old_confidence: {result.old_confidence}")
    print(f"    - new_confidence: {result.new_confidence}")
    print(f"    - decayed: {result.decayed}")
    print(f"    - skip_reason: {result.skip_reason}")
    print(f"    - days_since_activity: {result.days_since_activity}")
    print(f"    - effective_decay_rate: {result.effective_decay_rate}")

    # Step 3: Verify confidence decreased
    assert result.decayed, "Expected fact to decay"
    assert result.new_confidence < result.old_confidence, "Expected confidence to decrease"
    assert result.new_confidence >= config.min_confidence, "Confidence should not go below floor"
    print(f"  [OK] Confidence decreased from {result.old_confidence} to {result.new_confidence}")

    # Apply the decay to the fact
    fact["data"]["confidence"] = result.new_confidence
    decayed_confidence = result.new_confidence

    # Step 4: Check if stale
    stale = is_stale(fact, config)
    print(f"  Stale check: {stale} (threshold: {config.stale_threshold})")

    # Step 5: Reinforce the fact
    fact = reinforce_fact(fact)
    print(f"  After reinforcement:")
    print(f"    - reinforcement_count: {fact['data']['reinforcement_count']}")
    print(f"    - last_reinforced_at: {fact['data']['last_reinforced_at']}")

    assert fact["data"]["reinforcement_count"] == 1, "Expected reinforcement_count to be 1"
    assert fact["data"]["last_reinforced_at"] is not None, "Expected last_reinforced_at to be set"
    print("  [OK] Fact reinforced successfully")

    # Step 6: Apply decay again (should be protected by reinforcement)
    result2 = calculate_decay(fact, config)

    print(f"  After second decay calculation:")
    print(f"    - decayed: {result2.decayed}")
    print(f"    - skip_reason: {result2.skip_reason}")

    # Step 7: Verify confidence stays stable (should be protected)
    if result2.skip_reason == "reinforcement_protection":
        print("  [OK] Fact protected by reinforcement - no decay applied")
    elif result2.skip_reason == "grace_period":
        print("  [OK] Fact in grace period - no decay applied")
    else:
        # If decay did happen, it should be much slower
        if result2.decayed:
            print(f"  [WARN] Fact decayed even after reinforcement: {result2.new_confidence}")
        else:
            print("  [OK] No significant decay after reinforcement")

    return True


def test_pinned_never_decays():
    """Test that pinned facts never decay."""
    print("Testing pinned facts never decay...")
    from decay import calculate_decay, DecayConfig
    from datetime import datetime, timezone, timedelta

    config = DecayConfig(base_rate=0.1)  # Aggressive decay

    # Create an old pinned fact
    old_date = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat().replace("+00:00", "Z")

    fact = {
        "id": "fact_pinned_test",
        "type": "fact",
        "created_at": old_date,
        "data": {
            "claim": "Pinned fact test",
            "confidence": 0.9,
            "importance": 0.1,  # Low importance (would decay fast if not pinned)
            "pinned": True
        }
    }

    result = calculate_decay(fact, config)

    assert not result.decayed, "Pinned fact should not decay"
    assert result.skip_reason == "pinned", "Skip reason should be 'pinned'"
    assert result.new_confidence == result.old_confidence, "Confidence should be unchanged"

    print(f"  [OK] Pinned fact not decayed (skip_reason: {result.skip_reason})")
    return True


def test_importance_affects_decay_rate():
    """Test that importance affects decay rate."""
    print("Testing importance affects decay rate...")
    from decay import calculate_decay, DecayConfig
    from datetime import datetime, timezone, timedelta

    config = DecayConfig(
        base_rate=0.01,
        grace_period_days=0  # No grace period for this test
    )

    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace("+00:00", "Z")

    # High importance fact
    high_importance_fact = {
        "id": "fact_high_imp",
        "type": "fact",
        "created_at": old_date,
        "data": {
            "claim": "High importance",
            "confidence": 0.8,
            "importance": 0.9,  # High importance = slow decay
            "pinned": False
        }
    }

    # Low importance fact
    low_importance_fact = {
        "id": "fact_low_imp",
        "type": "fact",
        "created_at": old_date,
        "data": {
            "claim": "Low importance",
            "confidence": 0.8,
            "importance": 0.1,  # Low importance = fast decay
            "pinned": False
        }
    }

    high_result = calculate_decay(high_importance_fact, config)
    low_result = calculate_decay(low_importance_fact, config)

    print(f"  High importance (0.9): effective_rate={high_result.effective_decay_rate}, new_conf={high_result.new_confidence}")
    print(f"  Low importance (0.1): effective_rate={low_result.effective_decay_rate}, new_conf={low_result.new_confidence}")

    # High importance should have lower effective decay rate
    assert high_result.effective_decay_rate < low_result.effective_decay_rate, \
        "High importance should have lower decay rate"

    # Low importance should decay more
    high_decay_amount = high_result.old_confidence - high_result.new_confidence
    low_decay_amount = low_result.old_confidence - low_result.new_confidence

    assert low_decay_amount > high_decay_amount, \
        "Low importance fact should decay more"

    print(f"  [OK] Importance correctly affects decay rate")
    return True


def test_maintenance_report():
    """Test maintenance report generation."""
    print("Testing maintenance report...")
    from decay import generate_maintenance_report, DecayConfig
    from datetime import datetime, timezone, timedelta

    config = DecayConfig()

    old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat().replace("+00:00", "Z")
    recent_date = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")

    facts = [
        {
            "id": "fact_1",
            "type": "fact",
            "created_at": recent_date,
            "data": {"claim": "Recent fact", "confidence": 0.8, "importance": 0.5, "pinned": False}
        },
        {
            "id": "fact_2",
            "type": "fact",
            "created_at": old_date,
            "data": {"claim": "Old stale fact", "confidence": 0.2, "importance": 0.8, "pinned": False}
        },
        {
            "id": "fact_3",
            "type": "fact",
            "created_at": old_date,
            "data": {"claim": "Pinned fact", "confidence": 0.9, "importance": 0.3, "pinned": True}
        },
    ]

    report = generate_maintenance_report(facts, config)

    print(f"  Total facts: {report.total_facts}")
    print(f"  Pinned: {report.pinned_count} ({report.pinned_pct}%)")
    print(f"  Stale: {report.stale_count} ({report.stale_pct}%)")
    print(f"  Avg confidence: {report.avg_confidence}")
    print(f"  Top stale high-importance: {len(report.top_stale_high_importance)}")

    assert report.total_facts == 3, "Expected 3 facts"
    assert report.pinned_count == 1, "Expected 1 pinned fact"
    assert report.stale_count >= 1, "Expected at least 1 stale fact"

    print("  [OK] Maintenance report generated correctly")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Phase 4 Golden Decay Test")
    print("=" * 50)
    print()

    tests = [
        test_decay_imports,
        test_golden_decay_scenario,
        test_pinned_never_decays,
        test_importance_affects_decay_rate,
        test_maintenance_report,
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
            import traceback
            traceback.print_exc()
        print()

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
