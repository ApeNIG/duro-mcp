"""
Embeddings module for Duro semantic search.

This module defines the text representation spec for embedding artifacts.
The spec is LOCKED DOWN - changes affect search quality across all existing embeddings.

Key Design Decisions:
- Embed summaries only for episodes/logs (they grow fast)
- Skip metadata like timestamps, source_urls, confidence (not semantically meaningful)
- Include tags for all types (they're intentionally semantic)
- Keep text representations compact and focused
"""

from typing import Optional


# Text Representation Spec
# ========================
# This defines what gets embedded for each artifact type.
# Changes here affect search quality - modify with care.

TEXT_REPRESENTATION_SPEC = {
    "fact": {
        "include": ["claim", "snippet", "tags"],
        "skip": ["source_urls", "confidence", "provenance", "timestamps", "evidence_type", "verified"]
    },
    "decision": {
        "include": ["decision", "rationale", "context", "tags"],
        "skip": ["alternatives", "outcome", "episodes_used", "timestamps"]
    },
    "episode": {
        "include": ["goal", "result_summary", "tags"],
        "skip": ["actions", "timestamps", "links", "duration", "plan", "context"]
    },
    "evaluation": {
        "include": ["next_change", "grade"],
        "skip": ["rubric", "memory_updates", "applied", "timestamps"]
    },
    "skill_stats": {
        "include": ["name"],
        "skip": ["counters", "timestamps", "confidence", "skill_id"]
    },
    "log": {
        "include": ["summary"],  # SUMMARY ONLY - never embed full logs
        "skip": ["message", "raw_content", "timestamps"]
    }
}


def artifact_to_text(artifact: dict) -> str:
    """
    Convert an artifact to text for embedding.

    Returns ONLY the semantically meaningful text.
    This is the canonical function for text representation.

    Args:
        artifact: Full artifact dict with id, type, data, tags, etc.

    Returns:
        Clean text suitable for embedding (may be empty string)
    """
    artifact_type = artifact.get("type", "")
    data = artifact.get("data", {})
    tags = artifact.get("tags", [])

    # Join tags into space-separated string
    tags_str = " ".join(tags) if tags else ""

    if artifact_type == "fact":
        # Facts: claim + snippet + tags
        claim = data.get("claim", "") or ""
        snippet = data.get("snippet", "") or ""
        return _clean_text(f"{claim} {snippet} {tags_str}")

    elif artifact_type == "decision":
        # Decisions: decision + rationale + context + tags
        decision = data.get("decision", "") or ""
        rationale = data.get("rationale", "") or ""
        context = data.get("context", "") or ""
        return _clean_text(f"{decision} {rationale} {context} {tags_str}")

    elif artifact_type == "episode":
        # Episodes: goal + result_summary + tags (NO actions array)
        goal = data.get("goal", "") or ""
        result_summary = data.get("result_summary", "") or ""
        return _clean_text(f"{goal} {result_summary} {tags_str}")

    elif artifact_type == "evaluation":
        # Evaluations: next_change + grade
        next_change = data.get("next_change", "") or ""
        grade = data.get("grade", "") or ""
        return _clean_text(f"{next_change} Grade: {grade}")

    elif artifact_type == "skill_stats":
        # Skill stats: just name (rarely searched directly)
        name = data.get("name", "") or ""
        return _clean_text(name)

    elif artifact_type == "log":
        # Logs: SUMMARY ONLY - never embed full logs
        # Truncate to 500 chars max for safety
        summary = data.get("summary", "") or ""
        return _clean_text(summary[:500])

    # Unknown type - return empty
    return ""


def _clean_text(text: str) -> str:
    """
    Clean text for embedding.
    - Remove excessive whitespace
    - Strip leading/trailing whitespace
    - Return empty string if None
    """
    if not text:
        return ""
    # Normalize whitespace: collapse multiple spaces/newlines into single space
    import re
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()


def get_embeddable_types() -> list[str]:
    """
    Return list of artifact types that should be embedded.
    """
    return list(TEXT_REPRESENTATION_SPEC.keys())


def should_embed(artifact: dict) -> bool:
    """
    Check if an artifact should be embedded.

    Returns False if:
    - Type not in spec
    - Text representation would be empty
    """
    artifact_type = artifact.get("type", "")
    if artifact_type not in TEXT_REPRESENTATION_SPEC:
        return False

    text = artifact_to_text(artifact)
    return len(text.strip()) > 0


def compute_content_hash(artifact: dict) -> str:
    """
    Compute hash of embeddable content.

    Used to detect when content changes and re-embedding is needed.
    Only hashes the parts that go into the embedding, not metadata.
    """
    import hashlib
    text = artifact_to_text(artifact)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


# Embedding Model Config
# ======================
# FastEmbed model configuration

EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-small-en-v1.5",
    "embedding_dim": 384,
    "max_length": 512,
    "batch_size": 32
}

# Singleton embedding model (lazy loaded)
_embedding_model = None
_embedding_available = None


def is_embedding_available() -> bool:
    """
    Check if embedding functionality is available.

    Returns True if fastembed can be imported and model can load.
    Caches result for performance.
    """
    global _embedding_available

    if _embedding_available is not None:
        return _embedding_available

    try:
        from fastembed import TextEmbedding
        _embedding_available = True
    except ImportError:
        _embedding_available = False

    return _embedding_available


def get_embedding_model():
    """
    Get or create the embedding model singleton.

    Returns None if fastembed is not available.
    Lazy loads the model on first call.
    """
    global _embedding_model

    if not is_embedding_available():
        return None

    if _embedding_model is not None:
        return _embedding_model

    try:
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(
            model_name=EMBEDDING_CONFIG["model_name"]
        )
        return _embedding_model
    except Exception as e:
        print(f"[WARN] Failed to load embedding model: {e}")
        return None


def embed_text(text: str) -> Optional[list[float]]:
    """
    Generate embedding for a single text string.

    Args:
        text: Text to embed

    Returns:
        List of floats (embedding vector) or None if embedding unavailable
    """
    model = get_embedding_model()
    if model is None:
        return None

    if not text or not text.strip():
        return None

    try:
        # FastEmbed returns a generator, get first result
        embeddings = list(model.embed([text]))
        if embeddings:
            return embeddings[0].tolist()
        return None
    except Exception as e:
        print(f"[WARN] Embedding failed: {e}")
        return None


def embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """
    Generate embeddings for a batch of texts.

    More efficient than calling embed_text repeatedly.

    Args:
        texts: List of texts to embed

    Returns:
        List of embeddings (or None for failed/empty texts)
    """
    model = get_embedding_model()
    if model is None:
        return [None] * len(texts)

    # Filter empty texts but track their positions
    non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    if not non_empty:
        return [None] * len(texts)

    try:
        # Embed non-empty texts
        embeddings = list(model.embed([t for _, t in non_empty]))

        # Build result list with None for empty texts
        result = [None] * len(texts)
        for (orig_idx, _), emb in zip(non_empty, embeddings):
            result[orig_idx] = emb.tolist()

        return result
    except Exception as e:
        print(f"[WARN] Batch embedding failed: {e}")
        return [None] * len(texts)


def embed_artifact(artifact: dict) -> Optional[list[float]]:
    """
    Generate embedding for an artifact.

    Convenience function that combines artifact_to_text and embed_text.

    Args:
        artifact: Full artifact dict

    Returns:
        Embedding vector or None
    """
    if not should_embed(artifact):
        return None

    text = artifact_to_text(artifact)
    return embed_text(text)


def get_embedding_status() -> dict:
    """
    Get current embedding system status.

    Returns:
        {
            "available": bool,
            "model_name": str or None,
            "embedding_dim": int,
            "model_loaded": bool
        }
    """
    return {
        "available": is_embedding_available(),
        "model_name": EMBEDDING_CONFIG["model_name"] if is_embedding_available() else None,
        "embedding_dim": EMBEDDING_CONFIG["embedding_dim"],
        "model_loaded": _embedding_model is not None
    }
