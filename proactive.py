"""
Proactive Recall module for Duro.

Provides context-aware memory injection - surfacing relevant memories
before they're explicitly requested.

The ProactiveRecall class uses:
1. Hot path classification to determine if recall would be helpful
2. Hybrid search (FTS + vector) to find relevant memories
3. Ranking and filtering to return only high-value results

Integration points:
- Called at the start of task processing
- Results injected into agent context
- Can be triggered explicitly via duro_proactive_recall tool
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from time_utils import utc_now_iso
from autocapture import hot_path_classify, category_to_search_params, HotPathResult


@dataclass
class RecallResult:
    """Result from proactive recall."""
    triggered: bool
    reason: str
    memories: list[dict]
    search_mode: str  # "hybrid", "fts_only", "keyword_only"
    categories_matched: list[str]
    recall_time_ms: int


class ProactiveRecall:
    """
    Context-aware memory retrieval.

    Uses hot path classification + hybrid search to surface
    relevant memories based on current task context.
    """

    def __init__(self, artifact_store, index):
        """
        Initialize proactive recall.

        Args:
            artifact_store: ArtifactStore instance for loading artifacts
            index: ArtifactIndex instance for searching
        """
        self.artifact_store = artifact_store
        self.index = index

    def recall(
        self,
        context: str,
        limit: int = 10,
        include_types: Optional[list[str]] = None,
        min_confidence: float = 0.3,
        force: bool = False
    ) -> RecallResult:
        """
        Proactively recall relevant memories for the given context.

        Args:
            context: Current task/conversation context
            limit: Maximum memories to return
            include_types: Filter to specific artifact types (e.g., ["fact", "decision"])
            min_confidence: Minimum search score threshold
            force: If True, always search even if hot path says no

        Returns:
            RecallResult with matched memories
        """
        import time
        start_time = time.time()

        # Step 1: Hot path classification
        hot_result = hot_path_classify(context)

        if not force and not hot_result.should_retrieve:
            return RecallResult(
                triggered=False,
                reason=hot_result.reason,
                memories=[],
                search_mode="none",
                categories_matched=[],
                recall_time_ms=int((time.time() - start_time) * 1000)
            )

        # Step 2: Convert categories to search params
        search_params = category_to_search_params(hot_result.categories)

        # Override types if specified
        if include_types:
            search_params["artifact_types"] = include_types

        # Step 3: Search using hybrid search (or FTS fallback)
        memories = self._search_memories(
            context=context,
            tags=search_params.get("tags"),
            artifact_types=search_params.get("artifact_types"),
            limit=limit * 2,  # Get extra for filtering
            min_score=min_confidence
        )

        # Step 4: Load full artifacts and format results
        results = []
        for memory in memories[:limit]:
            artifact = self.artifact_store.get_artifact(memory["id"])
            if artifact:
                results.append(self._format_memory(artifact, memory))

        recall_time_ms = int((time.time() - start_time) * 1000)

        return RecallResult(
            triggered=True,
            reason=f"Recalled {len(results)} memories for: {hot_result.reason}",
            memories=results,
            search_mode=memories[0].get("source", "unknown") if memories else "none",
            categories_matched=hot_result.categories,
            recall_time_ms=recall_time_ms
        )

    def _search_memories(
        self,
        context: str,
        tags: Optional[list[str]] = None,
        artifact_types: Optional[list[str]] = None,
        limit: int = 20,
        min_score: float = 0.3
    ) -> list[dict]:
        """
        Search for relevant memories using hybrid search.

        Falls back gracefully: hybrid -> FTS -> keyword
        """
        # Try hybrid search first
        try:
            from embeddings import embed_text, is_embedding_available

            query_embedding = None
            if is_embedding_available():
                query_embedding = embed_text(context)

            # Use hybrid search
            search_result = self.index.hybrid_search(
                query=context,
                query_embedding=query_embedding,
                artifact_type=artifact_types[0] if artifact_types and len(artifact_types) == 1 else None,
                tags=tags,
                limit=limit,
                explain=False
            )

            results = search_result.get("results", [])

            # Filter by min score
            results = [r for r in results if r.get("search_score", 0) >= min_score]

            # Filter by types if multiple specified
            if artifact_types and len(artifact_types) > 1:
                results = [r for r in results if r.get("type") in artifact_types]

            return results

        except Exception as e:
            # Fall back to FTS search
            print(f"[WARN] Hybrid search failed, falling back to FTS: {e}")
            return self._fts_fallback(context, tags, artifact_types, limit)

    def _fts_fallback(
        self,
        context: str,
        tags: Optional[list[str]],
        artifact_types: Optional[list[str]],
        limit: int
    ) -> list[dict]:
        """Fallback to FTS-only search."""
        try:
            # Extract key terms from context
            terms = self._extract_search_terms(context)
            query = " ".join(terms[:5])  # Use top 5 terms

            results = self.index.fts_search(
                query=query,
                artifact_type=artifact_types[0] if artifact_types and len(artifact_types) == 1 else None,
                limit=limit
            )

            # Add source field
            for r in results:
                r["source"] = "fts"

            return results

        except Exception as e:
            print(f"[WARN] FTS search also failed: {e}")
            return []

    def _extract_search_terms(self, text: str) -> list[str]:
        """Extract important search terms from text."""
        import re

        # Remove common words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "or", "because",
            "if", "while", "this", "that", "these", "those", "what",
            "which", "who", "whom", "whose", "it", "its", "i", "me",
            "my", "we", "our", "you", "your", "he", "him", "his",
            "she", "her", "they", "them", "their"
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*[a-zA-Z0-9]\b', text.lower())

        # Filter and score
        terms = []
        for word in words:
            if word not in stopwords and len(word) > 2:
                terms.append(word)

        # Return unique terms preserving order
        seen = set()
        unique = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique.append(term)

        return unique

    def _format_memory(self, artifact: dict, search_result: dict) -> dict:
        """Format an artifact for memory injection."""
        data = artifact.get("data", {})
        artifact_type = artifact.get("type", "unknown")

        # Build summary based on type
        if artifact_type == "fact":
            summary = data.get("claim", "")
        elif artifact_type == "decision":
            summary = f"{data.get('decision', '')} (Rationale: {data.get('rationale', '')[:100]})"
        elif artifact_type == "episode":
            summary = f"{data.get('goal', '')} - {data.get('result_summary', '')}"
        else:
            summary = search_result.get("title", "")

        return {
            "id": artifact.get("id"),
            "type": artifact_type,
            "summary": summary[:300],
            "tags": artifact.get("tags", []),
            "created_at": artifact.get("created_at"),
            "relevance_score": search_result.get("search_score", 0),
            "source": search_result.get("source", "unknown")
        }

    def recall_for_task(self, task_description: str, limit: int = 5) -> list[dict]:
        """
        Convenience method for task-focused recall.

        Returns formatted memories most relevant to a specific task.

        Args:
            task_description: Description of the task to work on
            limit: Maximum memories to return

        Returns:
            List of formatted memory dicts
        """
        result = self.recall(
            context=task_description,
            limit=limit,
            include_types=["fact", "decision", "episode"],
            min_confidence=0.25,
            force=True  # Always search for task context
        )

        return result.memories

    def recall_related(self, artifact_id: str, limit: int = 5) -> list[dict]:
        """
        Recall memories related to a specific artifact.

        Uses both explicit relations and semantic similarity.

        Args:
            artifact_id: ID of the artifact to find relations for
            limit: Maximum memories to return

        Returns:
            List of related memory dicts
        """
        related = []

        # Get explicit relations
        relations = self.index.get_relations(artifact_id, direction="both")
        for rel in relations[:limit]:
            related_id = rel["target_id"] if rel["source_id"] == artifact_id else rel["source_id"]
            artifact = self.artifact_store.get_artifact(related_id)
            if artifact:
                related.append({
                    "id": related_id,
                    "type": artifact.get("type"),
                    "summary": self._get_artifact_summary(artifact),
                    "relation": rel["relation"],
                    "source": "explicit_relation"
                })

        # If we have room, add semantically similar
        if len(related) < limit:
            artifact = self.artifact_store.get_artifact(artifact_id)
            if artifact:
                from embeddings import artifact_to_text
                text = artifact_to_text(artifact)
                if text:
                    recall_result = self.recall(
                        context=text,
                        limit=limit - len(related),
                        force=True
                    )
                    for mem in recall_result.memories:
                        if mem["id"] != artifact_id and mem["id"] not in [r["id"] for r in related]:
                            mem["source"] = "semantic_similarity"
                            related.append(mem)

        return related[:limit]

    def _get_artifact_summary(self, artifact: dict) -> str:
        """Get a brief summary of an artifact."""
        data = artifact.get("data", {})
        artifact_type = artifact.get("type", "")

        if artifact_type == "fact":
            return data.get("claim", "")[:200]
        elif artifact_type == "decision":
            return data.get("decision", "")[:200]
        elif artifact_type == "episode":
            return data.get("goal", "")[:200]
        else:
            return str(data)[:200]


# =============================================================================
# Standalone extraction function for MCP tool
# =============================================================================

def extract_learnings_from_text(
    text: str,
    artifact_store=None,
    auto_save: bool = False
) -> dict:
    """
    Extract learnings from conversation or text.

    This is the entry point for the duro_extract_learnings MCP tool.

    Args:
        text: Conversation or text to extract from
        artifact_store: Optional ArtifactStore for saving (if auto_save=True)
        auto_save: If True, automatically save extracted items

    Returns:
        Dict with extracted learnings, facts, and decisions
    """
    from autocapture import warm_path_extract, cold_path_consolidate

    # Use warm path for shorter text, cold path for longer
    if len(text) < 3000:
        result = warm_path_extract(text, source_type="conversation")
        learnings = result.learnings
        facts = [
            {
                "claim": f.claim,
                "confidence": f.confidence,
                "tags": f.tags
            }
            for f in result.facts
        ]
        decisions = result.decisions
    else:
        # Use cold path consolidation for longer text
        summary = cold_path_consolidate(text)
        learnings = summary.key_learnings
        facts = [
            {
                "claim": f.claim,
                "confidence": f.confidence,
                "tags": f.tags
            }
            for f in summary.facts_discovered
        ]
        decisions = summary.decisions_made

    saved_ids = []

    # Auto-save if requested and artifact_store provided
    if auto_save and artifact_store:
        # Save learnings as facts
        for learning in learnings:
            success, artifact_id, _ = artifact_store.store_fact(
                claim=learning,
                confidence=0.5,
                tags=["auto-extracted", "learning"],
                workflow="auto_capture",
                sensitivity="internal"
            )
            if success:
                saved_ids.append(artifact_id)

        # Save extracted facts
        for fact in facts:
            success, artifact_id, _ = artifact_store.store_fact(
                claim=fact["claim"],
                confidence=fact["confidence"],
                tags=fact.get("tags", []) + ["auto-extracted"],
                workflow="auto_capture",
                sensitivity="internal"
            )
            if success:
                saved_ids.append(artifact_id)

        # Save decisions
        for dec in decisions:
            success, artifact_id, _ = artifact_store.store_decision(
                decision=dec.get("decision", ""),
                rationale=dec.get("rationale", ""),
                tags=["auto-extracted"],
                workflow="auto_capture",
                sensitivity="internal"
            )
            if success:
                saved_ids.append(artifact_id)

    return {
        "learnings": learnings,
        "facts": facts,
        "decisions": decisions,
        "count": len(learnings) + len(facts) + len(decisions),
        "saved_ids": saved_ids if auto_save else [],
        "auto_saved": auto_save
    }
