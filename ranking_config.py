"""
Ranking Configuration for Duro semantic search.

All ranking weights and rules are explicit here - no magic numbers buried in code.
This enables tuning and debugging of search quality.

Key Design Decisions:
- Vector weight > BM25 weight (semantic understanding prioritized)
- Recency boost decays over 14 days
- Type weights reflect search importance (facts > decisions > episodes)
- All thresholds are tunable
"""

from datetime import datetime, timezone
from typing import Optional


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


# Main Ranking Config
# ====================
# All weights are tunable. Change these to adjust search behavior.

RANKING = {
    # Hybrid search weights (must sum to 1.0)
    "vector_weight": 0.6,           # Semantic similarity weight
    "bm25_weight": 0.4,             # Keyword relevance weight

    # RRF (Reciprocal Rank Fusion) constant
    # Higher k = more weight to lower-ranked results
    "rrf_k": 60,

    # Recency boost configuration
    "recency_boost": {
        "enabled": True,
        "max_boost": 0.15,          # Max boost for today's artifacts
        "decay_days": 14,           # Full decay after 14 days
        "decay_curve": "linear"     # linear | exponential
    },

    # Type-based weights
    # Higher = more important in search results
    "type_weights": {
        "fact": 1.0,                # Facts are primary knowledge
        "decision": 0.9,            # Decisions are contextual
        "episode": 0.5,             # Episodes are operational history
        "evaluation": 0.3,          # Evaluations are meta
        "skill_stats": 0.2,         # Rarely searched directly
        "log": 0.1                  # Logs are noise, low priority
    },

    # Confidence boost for high-confidence facts
    "confidence_boost": {
        "enabled": True,
        "threshold": 0.7,           # Only boost if confidence >= this
        "max_boost": 0.1,           # Maximum boost amount
        "scale": "linear"           # How confidence maps to boost
    },

    # Score thresholds
    "min_score_threshold": 0.12,    # Below this, don't return
    "max_results": 50,              # Hard cap on results

    # Debug/explain options
    "include_score_breakdown": True  # Include score_components in results
}


# Decay Config (for Phase 4)
# ==========================
# Importance-aware decay: new = current * (1 - decay * (1 - importance))

DECAY_CONFIG = {
    "base_rate": 0.001,             # 0.1% per day base decay
    "min_confidence": 0.05,         # Floor - never go below this
    "max_confidence": 0.99,         # Ceiling
    "grace_period_days": 7,         # Don't decay recent facts
    "skip_if_pinned": True,         # Pinned facts never decay

    # Importance scaling
    # importance=0.9 (high): decay = 0.001 * 0.1 = 0.0001 (10x slower)
    # importance=0.5 (medium): decay = 0.001 * 0.5 = 0.0005
    # importance=0.1 (low): decay = 0.001 * 0.9 = 0.0009 (near full rate)
    "importance_scale": True
}


def calculate_recency_boost(created_at: str, now: Optional[datetime] = None) -> float:
    """
    Calculate recency boost for an artifact.

    Args:
        created_at: ISO timestamp of artifact creation
        now: Current time (defaults to utcnow)

    Returns:
        Boost value between 0 and RANKING["recency_boost"]["max_boost"]
    """
    if not RANKING["recency_boost"]["enabled"]:
        return 0.0

    try:
        if now is None:
            now = _utc_now()

        # Parse created_at
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo:
            created = created.replace(tzinfo=None)

        # Calculate days since creation
        days_old = (now - created).days

        if days_old < 0:
            return RANKING["recency_boost"]["max_boost"]

        decay_days = RANKING["recency_boost"]["decay_days"]
        max_boost = RANKING["recency_boost"]["max_boost"]

        if days_old >= decay_days:
            return 0.0

        # Linear decay
        if RANKING["recency_boost"]["decay_curve"] == "linear":
            return max_boost * (1 - days_old / decay_days)

        # Exponential decay
        elif RANKING["recency_boost"]["decay_curve"] == "exponential":
            import math
            half_life = decay_days / 3  # Decay to ~12% at full decay_days
            return max_boost * math.exp(-days_old / half_life)

        return 0.0

    except Exception:
        return 0.0


def calculate_type_weight(artifact_type: str) -> float:
    """
    Get type weight for an artifact type.

    Returns weight between 0 and 1.
    """
    return RANKING["type_weights"].get(artifact_type, 0.5)


def calculate_confidence_boost(confidence: float, artifact_type: str = "fact") -> float:
    """
    Calculate confidence boost for facts.

    Only applies to facts with confidence >= threshold.

    Args:
        confidence: Artifact confidence (0-1)
        artifact_type: Type of artifact

    Returns:
        Boost value between 0 and RANKING["confidence_boost"]["max_boost"]
    """
    if not RANKING["confidence_boost"]["enabled"]:
        return 0.0

    # Only boost facts
    if artifact_type != "fact":
        return 0.0

    threshold = RANKING["confidence_boost"]["threshold"]
    max_boost = RANKING["confidence_boost"]["max_boost"]

    if confidence < threshold:
        return 0.0

    # Scale confidence above threshold to boost
    # e.g., 0.7 threshold, 0.9 confidence = (0.9-0.7)/(1-0.7) = 0.67 of max_boost
    scale = (confidence - threshold) / (1 - threshold)
    return max_boost * scale


def calculate_decay(fact: dict) -> float:
    """
    Calculate decayed confidence for a fact (Phase 4).

    Uses importance-aware decay:
    new = current * (1 - decay * (1 - importance))

    Args:
        fact: Fact artifact dict

    Returns:
        New confidence value
    """
    data = fact.get("data", {})
    current = data.get("confidence", 0.5)
    importance = data.get("importance", 0.5)
    pinned = data.get("pinned", False)

    # Skip pinned facts
    if DECAY_CONFIG["skip_if_pinned"] and pinned:
        return current

    # Check grace period
    created_at = fact.get("created_at", "")
    if created_at:
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created.tzinfo:
                created = created.replace(tzinfo=None)
            days_old = (_utc_now() - created).days
            if days_old < DECAY_CONFIG["grace_period_days"]:
                return current
        except Exception:
            pass

    # Calculate effective decay
    base_rate = DECAY_CONFIG["base_rate"]

    if DECAY_CONFIG["importance_scale"]:
        # High importance = low decay
        effective_decay = base_rate * (1 - importance)
    else:
        effective_decay = base_rate

    # Apply decay
    new_confidence = current * (1 - effective_decay)

    # Enforce bounds
    new_confidence = max(DECAY_CONFIG["min_confidence"], new_confidence)
    new_confidence = min(DECAY_CONFIG["max_confidence"], new_confidence)

    return round(new_confidence, 4)


def calculate_combined_score(
    vector_score: float,
    bm25_score: float,
    created_at: str,
    artifact_type: str,
    confidence: float = 0.5
) -> tuple[float, dict]:
    """
    Calculate combined search score with full breakdown.

    Args:
        vector_score: Semantic similarity score (0-1)
        bm25_score: BM25 keyword score (0-1)
        created_at: ISO timestamp
        artifact_type: Type of artifact
        confidence: Artifact confidence (0-1)

    Returns:
        (final_score, score_components)
    """
    # Base hybrid score
    hybrid_score = (
        RANKING["vector_weight"] * vector_score +
        RANKING["bm25_weight"] * bm25_score
    )

    # Calculate boosts
    recency_boost = calculate_recency_boost(created_at)
    type_weight = calculate_type_weight(artifact_type)
    confidence_boost = calculate_confidence_boost(confidence, artifact_type)

    # Final score with boosts
    final_score = hybrid_score * type_weight + recency_boost + confidence_boost

    # Build score components
    components = {
        "vector": round(vector_score, 4),
        "bm25": round(bm25_score, 4),
        "hybrid_base": round(hybrid_score, 4),
        "type_weight": round(type_weight, 4),
        "recency_boost": round(recency_boost, 4),
        "confidence_boost": round(confidence_boost, 4),
        "final": round(final_score, 4)
    }

    return final_score, components


def explain_score(components: dict) -> str:
    """
    Generate human-readable explanation of score components.

    Args:
        components: Score components dict from calculate_combined_score

    Returns:
        Explanation string
    """
    explanations = []

    # Vector score interpretation
    vec = components.get("vector", 0)
    if vec >= 0.8:
        explanations.append(f"High vector match ({vec:.2f})")
    elif vec >= 0.5:
        explanations.append(f"Moderate vector match ({vec:.2f})")
    else:
        explanations.append(f"Low vector match ({vec:.2f})")

    # BM25 interpretation
    bm25 = components.get("bm25", 0)
    if bm25 >= 0.5:
        explanations.append(f"strong keyword match")
    elif bm25 >= 0.2:
        explanations.append(f"moderate keyword match")

    # Recency
    recency = components.get("recency_boost", 0)
    if recency > 0.1:
        explanations.append(f"recent ({recency:.0%} boost)")
    elif recency > 0.05:
        explanations.append(f"fairly recent")

    # Confidence
    conf = components.get("confidence_boost", 0)
    if conf > 0:
        explanations.append(f"high confidence fact")

    return ", ".join(explanations) if explanations else "baseline match"
