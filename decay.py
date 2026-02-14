"""
Confidence Decay module for Duro.

Implements time-based confidence decay for facts with explicit,
documented math. No magic numbers - everything is configurable.

Key principles:
- Pinned facts never decay
- Importance affects decay rate (high importance = slower decay)
- Recently reinforced facts decay slower
- Grace period protects new facts
- Confidence has a floor (never goes to 0)
"""

from dataclasses import dataclass
from typing import Optional

from time_utils import utc_now_iso, days_since, parse_iso_datetime


# =============================================================================
# Configuration (all tunable, no magic numbers)
# =============================================================================

@dataclass
class DecayConfig:
    """
    Decay configuration - all parameters explicit and documented.
    """
    # Base decay rate per day (0.001 = 0.1% per day)
    base_rate: float = 0.001

    # Minimum confidence (floor - facts never go below this)
    min_confidence: float = 0.05

    # Maximum confidence (ceiling)
    max_confidence: float = 0.99

    # Grace period in days (no decay for new facts)
    grace_period_days: int = 7

    # Reinforcement protection: days of protection per reinforcement
    reinforcement_protection_days: int = 3

    # Stale threshold: below this, fact is considered stale
    stale_threshold: float = 0.3


# Default config
DEFAULT_DECAY_CONFIG = DecayConfig()


# =============================================================================
# Decay Math (explicit, documented, predictable)
# =============================================================================

@dataclass
class DecayResult:
    """Result of decay calculation."""
    old_confidence: float
    new_confidence: float
    decayed: bool
    skip_reason: Optional[str]
    days_since_activity: int
    effective_decay_rate: float


def calculate_decay(
    fact: dict,
    config: DecayConfig = DEFAULT_DECAY_CONFIG
) -> DecayResult:
    """
    Calculate confidence decay for a single fact.

    Math:
    1. Skip if pinned
    2. Skip if within grace period
    3. Skip if recently reinforced
    4. Calculate days since last activity
    5. Apply importance-weighted decay:
       effective_rate = base_rate * (1 - importance)
       new_confidence = current * (1 - effective_rate * days)
    6. Clamp to [min_confidence, max_confidence]

    Args:
        fact: Full fact artifact dict
        config: Decay configuration

    Returns:
        DecayResult with old/new confidence and metadata
    """
    data = fact.get("data", {})

    # Current state
    current_confidence = data.get("confidence", 0.5)
    importance = data.get("importance", 0.5)
    pinned = data.get("pinned", False)
    created_at = fact.get("created_at")
    last_reinforced_at = data.get("last_reinforced_at")

    # Rule 1: Pinned facts never decay
    if pinned:
        return DecayResult(
            old_confidence=current_confidence,
            new_confidence=current_confidence,
            decayed=False,
            skip_reason="pinned",
            days_since_activity=0,
            effective_decay_rate=0
        )

    # Determine last activity date (reinforcement or creation)
    if last_reinforced_at:
        last_activity = last_reinforced_at
    elif created_at:
        last_activity = created_at
    else:
        # No timestamp - can't decay
        return DecayResult(
            old_confidence=current_confidence,
            new_confidence=current_confidence,
            decayed=False,
            skip_reason="no_timestamp",
            days_since_activity=0,
            effective_decay_rate=0
        )

    # Calculate days since last activity
    try:
        days_inactive = days_since(last_activity)
    except Exception:
        return DecayResult(
            old_confidence=current_confidence,
            new_confidence=current_confidence,
            decayed=False,
            skip_reason="invalid_timestamp",
            days_since_activity=0,
            effective_decay_rate=0
        )

    # Rule 2: Grace period for new facts
    if days_inactive <= config.grace_period_days:
        return DecayResult(
            old_confidence=current_confidence,
            new_confidence=current_confidence,
            decayed=False,
            skip_reason="grace_period",
            days_since_activity=days_inactive,
            effective_decay_rate=0
        )

    # Rule 3: Reinforcement protection
    reinforcement_count = data.get("reinforcement_count", 0)
    protection_days = reinforcement_count * config.reinforcement_protection_days
    if days_inactive <= (config.grace_period_days + protection_days):
        return DecayResult(
            old_confidence=current_confidence,
            new_confidence=current_confidence,
            decayed=False,
            skip_reason="reinforcement_protection",
            days_since_activity=days_inactive,
            effective_decay_rate=0
        )

    # Calculate effective days to decay (subtract protected days)
    effective_decay_days = days_inactive - config.grace_period_days - protection_days

    if effective_decay_days <= 0:
        return DecayResult(
            old_confidence=current_confidence,
            new_confidence=current_confidence,
            decayed=False,
            skip_reason="no_decay_days",
            days_since_activity=days_inactive,
            effective_decay_rate=0
        )

    # Rule 4: Importance-weighted decay
    # High importance (1.0) = 0% decay rate
    # Low importance (0.0) = 100% base decay rate
    importance_factor = 1 - importance
    effective_rate = config.base_rate * importance_factor

    # Apply decay: confidence * (1 - rate)^days
    # For small rates, this is approximately: confidence * (1 - rate * days)
    # Using the simpler linear approximation for predictability
    decay_factor = 1 - (effective_rate * effective_decay_days)
    decay_factor = max(0, decay_factor)  # Can't go negative

    new_confidence = current_confidence * decay_factor

    # Clamp to bounds
    new_confidence = max(config.min_confidence, min(config.max_confidence, new_confidence))

    # Determine if actually decayed
    decayed = new_confidence < current_confidence

    return DecayResult(
        old_confidence=current_confidence,
        new_confidence=new_confidence,
        decayed=decayed,
        skip_reason=None,
        days_since_activity=days_inactive,
        effective_decay_rate=effective_rate
    )


def is_stale(fact: dict, config: DecayConfig = DEFAULT_DECAY_CONFIG) -> bool:
    """
    Check if a fact is stale (confidence below threshold).

    Args:
        fact: Full fact artifact dict
        config: Decay configuration

    Returns:
        True if confidence is below stale threshold
    """
    data = fact.get("data", {})
    confidence = data.get("confidence", 0.5)
    return confidence < config.stale_threshold


def reinforce_fact(fact: dict) -> dict:
    """
    Reinforce a fact - update reinforcement tracking.

    This should be called when:
    - A fact is used in an answer
    - A fact is explicitly confirmed
    - A fact is cited in a decision

    Args:
        fact: Full fact artifact dict

    Returns:
        Updated fact with reinforcement data
    """
    data = fact.get("data", {})

    # Increment reinforcement count
    data["reinforcement_count"] = data.get("reinforcement_count", 0) + 1

    # Update last reinforced timestamp
    data["last_reinforced_at"] = utc_now_iso()

    fact["data"] = data
    return fact


# =============================================================================
# Batch Operations
# =============================================================================

@dataclass
class BatchDecayResult:
    """Result of batch decay operation."""
    total_facts: int
    decayed_count: int
    skipped_pinned: int
    skipped_grace_period: int
    skipped_reinforcement: int
    skipped_other: int
    stale_count: int
    results: list[dict]  # Individual results for reporting


def apply_batch_decay(
    facts: list[dict],
    config: DecayConfig = DEFAULT_DECAY_CONFIG,
    dry_run: bool = True
) -> BatchDecayResult:
    """
    Apply decay to a batch of facts.

    Args:
        facts: List of full fact artifact dicts
        config: Decay configuration
        dry_run: If True, calculate but don't modify

    Returns:
        BatchDecayResult with summary and details
    """
    results = []
    decayed_count = 0
    skipped_pinned = 0
    skipped_grace_period = 0
    skipped_reinforcement = 0
    skipped_other = 0
    stale_count = 0

    for fact in facts:
        decay_result = calculate_decay(fact, config)

        result_dict = {
            "id": fact.get("id"),
            "old_confidence": decay_result.old_confidence,
            "new_confidence": decay_result.new_confidence,
            "decayed": decay_result.decayed,
            "skip_reason": decay_result.skip_reason,
            "days_since_activity": decay_result.days_since_activity
        }

        if decay_result.decayed:
            decayed_count += 1
            if not dry_run:
                # Update fact confidence
                fact["data"]["confidence"] = decay_result.new_confidence

        if decay_result.skip_reason == "pinned":
            skipped_pinned += 1
        elif decay_result.skip_reason == "grace_period":
            skipped_grace_period += 1
        elif decay_result.skip_reason in ["reinforcement_protection", "no_decay_days"]:
            skipped_reinforcement += 1
        elif decay_result.skip_reason:
            skipped_other += 1

        # Check if stale after decay
        if is_stale(fact, config):
            stale_count += 1
            result_dict["stale"] = True

        results.append(result_dict)

    return BatchDecayResult(
        total_facts=len(facts),
        decayed_count=decayed_count,
        skipped_pinned=skipped_pinned,
        skipped_grace_period=skipped_grace_period,
        skipped_reinforcement=skipped_reinforcement,
        skipped_other=skipped_other,
        stale_count=stale_count,
        results=results
    )


# =============================================================================
# Maintenance Report
# =============================================================================

@dataclass
class MaintenanceReport:
    """Maintenance report for memory health."""
    total_facts: int
    pinned_count: int
    pinned_pct: float
    stale_count: int
    stale_pct: float
    avg_confidence: float
    avg_importance: float
    avg_reinforcement_count: float
    oldest_unreinforced_days: int
    top_stale_high_importance: list[dict]


def generate_maintenance_report(
    facts: list[dict],
    config: DecayConfig = DEFAULT_DECAY_CONFIG,
    top_n_stale: int = 10
) -> MaintenanceReport:
    """
    Generate a maintenance report for facts.

    Args:
        facts: List of all fact artifacts
        config: Decay configuration
        top_n_stale: Number of stale high-importance facts to report

    Returns:
        MaintenanceReport with health metrics
    """
    if not facts:
        return MaintenanceReport(
            total_facts=0,
            pinned_count=0,
            pinned_pct=0,
            stale_count=0,
            stale_pct=0,
            avg_confidence=0,
            avg_importance=0,
            avg_reinforcement_count=0,
            oldest_unreinforced_days=0,
            top_stale_high_importance=[]
        )

    pinned_count = 0
    stale_count = 0
    total_confidence = 0
    total_importance = 0
    total_reinforcement = 0
    oldest_unreinforced = 0
    stale_high_importance = []

    for fact in facts:
        data = fact.get("data", {})
        confidence = data.get("confidence", 0.5)
        importance = data.get("importance", 0.5)
        pinned = data.get("pinned", False)
        reinforcement_count = data.get("reinforcement_count", 0)
        last_reinforced_at = data.get("last_reinforced_at")
        created_at = fact.get("created_at")

        # Counts
        if pinned:
            pinned_count += 1

        is_fact_stale = is_stale(fact, config)
        if is_fact_stale:
            stale_count += 1
            if importance >= 0.5:  # High importance threshold
                stale_high_importance.append({
                    "id": fact.get("id"),
                    "claim": data.get("claim", "")[:100],
                    "confidence": confidence,
                    "importance": importance,
                    "days_inactive": 0  # Will be calculated below
                })

        # Totals
        total_confidence += confidence
        total_importance += importance
        total_reinforcement += reinforcement_count

        # Days since activity
        last_activity = last_reinforced_at or created_at
        if last_activity:
            try:
                days_inactive = days_since(last_activity)
                if days_inactive > oldest_unreinforced:
                    oldest_unreinforced = days_inactive
                # Update stale entry
                if is_fact_stale and importance >= 0.5:
                    for entry in stale_high_importance:
                        if entry["id"] == fact.get("id"):
                            entry["days_inactive"] = days_inactive
            except Exception:
                pass

    n = len(facts)

    # Sort stale high-importance by importance (descending), then confidence (ascending)
    stale_high_importance.sort(key=lambda x: (-x["importance"], x["confidence"]))

    return MaintenanceReport(
        total_facts=n,
        pinned_count=pinned_count,
        pinned_pct=round(100 * pinned_count / n, 1),
        stale_count=stale_count,
        stale_pct=round(100 * stale_count / n, 1),
        avg_confidence=round(total_confidence / n, 3),
        avg_importance=round(total_importance / n, 3),
        avg_reinforcement_count=round(total_reinforcement / n, 2),
        oldest_unreinforced_days=oldest_unreinforced,
        top_stale_high_importance=stale_high_importance[:top_n_stale]
    )
