"""
Auto-capture module for Duro.

Implements three-path auto-extraction architecture:
- Hot Path (<100ms): Keyword classifier on message → trigger retrieval
- Warm Path (<5s): Extract facts from tool outputs (async)
- Cold Path (>60s): Session-end consolidation

The goal is to reduce manual discipline by auto-extracting
learnings and facts from conversations and tool outputs.
"""

import re
from dataclasses import dataclass
from typing import Optional

from time_utils import utc_now_iso


# =============================================================================
# Hot Path: Keyword Classifier (<100ms)
# =============================================================================

# Keywords that suggest memory retrieval would be helpful
RETRIEVAL_TRIGGERS = {
    # Project/task context
    "project": ["working on", "project", "codebase", "repository", "repo"],
    "deployment": ["deploy", "render", "vercel", "production", "staging"],
    "database": ["database", "postgres", "sqlite", "migration", "schema"],
    "api": ["api", "endpoint", "route", "request", "response"],

    # Technical decisions
    "decision": ["decided", "chose", "picked", "went with", "using"],
    "architecture": ["architecture", "pattern", "approach", "design"],

    # Past experience
    "recall": ["last time", "before", "previously", "remember when"],
    "error": ["error", "bug", "issue", "problem", "failed", "broken"],
    "fix": ["fixed", "solved", "resolved", "figured out"],

    # User preferences
    "preference": ["prefer", "like", "want", "always", "never", "style"],
}

# Keywords that suggest learning extraction
LEARNING_SIGNALS = [
    r"\blearned\b",
    r"\brealize[ds]?\b",
    r"\bfigured out\b",
    r"\bturns out\b",
    r"\bthe trick is\b",
    r"\bkey insight\b",
    r"\bimportant(ly)?\b",
    r"\bremember to\b",
    r"\bnote to self\b",
    r"\bpro tip\b",
    r"\bgotcha\b",
    r"\bwatch out for\b",
    r"\bavoid\b",
    r"\balways\b.*\bshould\b",
    r"\bnever\b.*\bshould\b",
]


@dataclass
class HotPathResult:
    """Result from hot path classification."""
    should_retrieve: bool
    categories: list[str]
    confidence: float
    reason: str


def hot_path_classify(message: str) -> HotPathResult:
    """
    Fast keyword-based classification of user message.

    Determines:
    1. Whether memory retrieval would be helpful
    2. What categories of memory to search

    Must complete in <100ms (typically <10ms).

    Args:
        message: User message text

    Returns:
        HotPathResult with retrieval recommendation
    """
    if not message or len(message.strip()) < 10:
        return HotPathResult(
            should_retrieve=False,
            categories=[],
            confidence=0.0,
            reason="Message too short"
        )

    message_lower = message.lower()
    matched_categories = []

    # Check each category for keyword matches
    for category, keywords in RETRIEVAL_TRIGGERS.items():
        for keyword in keywords:
            if keyword in message_lower:
                if category not in matched_categories:
                    matched_categories.append(category)
                break

    # Calculate confidence based on matches
    if not matched_categories:
        return HotPathResult(
            should_retrieve=False,
            categories=[],
            confidence=0.0,
            reason="No retrieval triggers detected"
        )

    # More matches = higher confidence
    confidence = min(0.9, 0.3 + (len(matched_categories) * 0.2))

    return HotPathResult(
        should_retrieve=True,
        categories=matched_categories,
        confidence=confidence,
        reason=f"Matched categories: {', '.join(matched_categories)}"
    )


def detect_learning_signal(text: str) -> bool:
    """
    Check if text contains signals that suggest a learning/insight.

    Used to identify text worth capturing as facts/learnings.

    Args:
        text: Text to analyze

    Returns:
        True if learning signals detected
    """
    if not text:
        return False

    text_lower = text.lower()

    for pattern in LEARNING_SIGNALS:
        if re.search(pattern, text_lower):
            return True

    return False


# =============================================================================
# Warm Path: Fact Extraction (<5s)
# =============================================================================

@dataclass
class ExtractedFact:
    """A fact extracted from tool output or conversation."""
    claim: str
    confidence: float
    source_type: str  # "tool_output", "conversation", "user_stated"
    tags: list[str]
    snippet: Optional[str] = None


@dataclass
class WarmPathResult:
    """Result from warm path extraction."""
    facts: list[ExtractedFact]
    learnings: list[str]
    decisions: list[dict]


# Patterns for extracting facts from tool outputs
FACT_PATTERNS = [
    # "X is Y" patterns
    (r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+is\s+(.+?)(?:\.|$)", "definition"),
    # "X uses Y" patterns
    (r"(\w+(?:\s+\w+)?)\s+uses?\s+(.+?)(?:\.|$)", "usage"),
    # "X requires Y" patterns
    (r"(\w+(?:\s+\w+)?)\s+requires?\s+(.+?)(?:\.|$)", "requirement"),
    # Version patterns
    (r"(\w+)\s+version\s*[:\s]?\s*(\d+[\d\.]+)", "version"),
    # Configuration patterns
    (r"(?:set|configured?|using)\s+(\w+)\s+(?:to|as)\s+(.+?)(?:\.|$)", "config"),
]

# Patterns for extracting learnings
LEARNING_PATTERNS = [
    r"(?:i\s+)?learned\s+(?:that\s+)?(.+?)(?:\.|$)",
    r"(?:the\s+)?(?:key\s+)?(?:insight|lesson)\s+(?:is|was)\s+(?:that\s+)?(.+?)(?:\.|$)",
    r"turns?\s+out\s+(?:that\s+)?(.+?)(?:\.|$)",
    r"(?:important|remember):\s*(.+?)(?:\.|$)",
    r"pro\s+tip:\s*(.+?)(?:\.|$)",
    r"gotcha:\s*(.+?)(?:\.|$)",
]

# Patterns for extracting decisions
DECISION_PATTERNS = [
    r"(?:i\s+)?(?:decided|chose|picked)\s+(?:to\s+)?(.+?)\s+(?:because|since|as)\s+(.+?)(?:\.|$)",
    r"going\s+(?:to\s+)?(?:use|with)\s+(.+?)\s+(?:because|since|for)\s+(.+?)(?:\.|$)",
]


def warm_path_extract(text: str, source_type: str = "conversation") -> WarmPathResult:
    """
    Extract facts, learnings, and decisions from text.

    This is the warm path - takes <5s for typical text.
    Used for processing tool outputs and conversation chunks.

    Args:
        text: Text to extract from
        source_type: Where the text came from ("tool_output", "conversation", "user_stated")

    Returns:
        WarmPathResult with extracted items
    """
    if not text or len(text.strip()) < 20:
        return WarmPathResult(facts=[], learnings=[], decisions=[])

    text_clean = text.strip()
    facts = []
    learnings = []
    decisions = []

    # Extract facts
    for pattern, fact_type in FACT_PATTERNS:
        for match in re.finditer(pattern, text_clean, re.IGNORECASE):
            subject = match.group(1).strip()
            predicate = match.group(2).strip()

            # Skip if too short or too generic
            if len(subject) < 3 or len(predicate) < 5:
                continue

            claim = f"{subject} {fact_type}: {predicate}"

            facts.append(ExtractedFact(
                claim=claim,
                confidence=0.4,  # Low confidence - needs verification
                source_type=source_type,
                tags=["auto-extracted", fact_type],
                snippet=match.group(0)[:200]
            ))

    # Extract learnings
    for pattern in LEARNING_PATTERNS:
        for match in re.finditer(pattern, text_clean, re.IGNORECASE):
            learning = match.group(1).strip()
            if len(learning) > 20:  # Meaningful length
                learnings.append(learning)

    # Extract decisions
    for pattern in DECISION_PATTERNS:
        for match in re.finditer(pattern, text_clean, re.IGNORECASE):
            decision = match.group(1).strip()
            rationale = match.group(2).strip()
            if len(decision) > 10 and len(rationale) > 10:
                decisions.append({
                    "decision": decision,
                    "rationale": rationale,
                    "extracted_at": utc_now_iso()
                })

    # Deduplicate
    seen_facts = set()
    unique_facts = []
    for fact in facts:
        key = fact.claim.lower()
        if key not in seen_facts:
            seen_facts.add(key)
            unique_facts.append(fact)

    learnings = list(set(learnings))

    return WarmPathResult(
        facts=unique_facts[:10],  # Cap at 10 per extraction
        learnings=learnings[:5],
        decisions=decisions[:3]
    )


# =============================================================================
# Cold Path: Session Consolidation (>60s)
# =============================================================================

@dataclass
class SessionSummary:
    """Summary of a session for memory consolidation."""
    tasks_completed: list[str]
    key_learnings: list[str]
    decisions_made: list[dict]
    facts_discovered: list[ExtractedFact]
    topics_discussed: list[str]
    tools_used: list[str]


def cold_path_consolidate(conversation: str, tool_calls: list[dict] = None) -> SessionSummary:
    """
    Consolidate a full session into structured memory artifacts.

    This is the cold path - runs at session end, can take >60s.
    Processes the full conversation to extract:
    - Tasks completed
    - Key learnings
    - Decisions made
    - Facts discovered
    - Topics discussed

    Args:
        conversation: Full conversation text
        tool_calls: List of tool calls made during session

    Returns:
        SessionSummary with consolidated insights
    """
    tool_calls = tool_calls or []

    # Split conversation into chunks for analysis
    chunks = _split_conversation(conversation)

    all_facts = []
    all_learnings = []
    all_decisions = []
    topics = set()

    # Process each chunk
    for chunk in chunks:
        result = warm_path_extract(chunk, source_type="conversation")
        all_facts.extend(result.facts)
        all_learnings.extend(result.learnings)
        all_decisions.extend(result.decisions)

        # Extract topics from chunk
        chunk_topics = _extract_topics(chunk)
        topics.update(chunk_topics)

    # Extract tasks from conversation
    tasks = _extract_tasks(conversation)

    # Extract tools used
    tools_used = list(set(tc.get("tool", tc.get("name", "unknown")) for tc in tool_calls))

    # Deduplicate and rank facts by importance
    unique_facts = _dedupe_and_rank_facts(all_facts)

    return SessionSummary(
        tasks_completed=tasks[:10],
        key_learnings=list(set(all_learnings))[:10],
        decisions_made=all_decisions[:5],
        facts_discovered=unique_facts[:15],
        topics_discussed=list(topics)[:10],
        tools_used=tools_used
    )


def _split_conversation(text: str, max_chunk_size: int = 2000) -> list[str]:
    """Split conversation into processable chunks."""
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    # Split on double newlines (message boundaries)
    parts = text.split("\n\n")

    current_chunk = ""
    for part in parts:
        if len(current_chunk) + len(part) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = part
        else:
            current_chunk += "\n\n" + part if current_chunk else part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _extract_topics(text: str) -> set[str]:
    """Extract likely topics from text."""
    topics = set()

    # Common technical topics to detect
    topic_keywords = {
        "python": ["python", "pip", "pytest", "django", "flask"],
        "javascript": ["javascript", "node", "npm", "react", "vue", "typescript"],
        "database": ["database", "sql", "postgres", "sqlite", "mongodb"],
        "api": ["api", "rest", "graphql", "endpoint"],
        "deployment": ["deploy", "docker", "kubernetes", "render", "vercel"],
        "git": ["git", "commit", "branch", "merge", "pull request"],
        "testing": ["test", "unittest", "pytest", "jest", "coverage"],
        "security": ["security", "auth", "token", "password", "encryption"],
    }

    text_lower = text.lower()
    for topic, keywords in topic_keywords.items():
        if any(kw in text_lower for kw in keywords):
            topics.add(topic)

    return topics


def _extract_tasks(text: str) -> list[str]:
    """Extract completed tasks from conversation."""
    tasks = []

    # Patterns that indicate task completion
    task_patterns = [
        r"(?:completed|finished|done with|implemented)\s+(.+?)(?:\.|$)",
        r"(?:successfully|just)\s+(?:created|built|fixed|updated|added)\s+(.+?)(?:\.|$)",
        r"task[:\s]+(.+?)\s+(?:is\s+)?(?:complete|done|finished)",
    ]

    for pattern in task_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            task = match.group(1).strip()
            if 10 < len(task) < 200:
                tasks.append(task)

    return list(set(tasks))


def _dedupe_and_rank_facts(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Deduplicate facts and rank by confidence/importance."""
    # Group by similar claims
    seen = {}
    for fact in facts:
        key = fact.claim.lower()[:50]  # First 50 chars as key
        if key not in seen:
            seen[key] = fact
        else:
            # Keep higher confidence version
            if fact.confidence > seen[key].confidence:
                seen[key] = fact

    # Sort by confidence
    ranked = sorted(seen.values(), key=lambda f: f.confidence, reverse=True)
    return ranked


# =============================================================================
# Utility: Category to Search Query Mapping
# =============================================================================

def category_to_search_params(categories: list[str]) -> dict:
    """
    Convert hot path categories to search parameters.

    Args:
        categories: List of matched categories from hot path

    Returns:
        Dict with search parameters (tags, types, etc.)
    """
    search_params = {
        "tags": [],
        "artifact_types": [],
        "keywords": []
    }

    category_mappings = {
        "project": {
            "tags": ["project", "codebase"],
            "types": ["fact", "decision"]
        },
        "deployment": {
            "tags": ["deployment", "render", "vercel"],
            "types": ["fact", "decision"]
        },
        "database": {
            "tags": ["database", "postgres", "sqlite", "migration"],
            "types": ["fact", "decision"]
        },
        "api": {
            "tags": ["api", "endpoint"],
            "types": ["fact"]
        },
        "decision": {
            "tags": [],
            "types": ["decision"]
        },
        "architecture": {
            "tags": ["architecture", "design", "pattern"],
            "types": ["decision", "fact"]
        },
        "recall": {
            "tags": [],
            "types": ["episode", "fact", "decision"]
        },
        "error": {
            "tags": ["error", "bug", "fix"],
            "types": ["fact", "episode"]
        },
        "fix": {
            "tags": ["fix", "solution"],
            "types": ["fact", "episode"]
        },
        "preference": {
            "tags": ["preference", "style"],
            "types": ["fact", "decision"]
        },
    }

    for category in categories:
        if category in category_mappings:
            mapping = category_mappings[category]
            search_params["tags"].extend(mapping.get("tags", []))
            search_params["artifact_types"].extend(mapping.get("types", []))

    # Deduplicate
    search_params["tags"] = list(set(search_params["tags"]))
    search_params["artifact_types"] = list(set(search_params["artifact_types"]))

    return search_params
