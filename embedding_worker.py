"""
Embedding Worker for Duro semantic search.

Implements async queue-based embedding pipeline.
Key design: Non-blocking on artifact save. Queue-based processing.

Flow:
1. Store artifact -> Write JSON -> Index SQLite -> Queue ID to pending/
2. Background worker (periodic) -> Read pending/ -> Embed -> Update vectors -> Delete from pending/

The pending/ directory is file-based for crash resilience.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    """Return current UTC time as ISO string with Z suffix."""
    return _utc_now().isoformat().replace("+00:00", "Z")
from typing import Optional, Callable

from embeddings import artifact_to_text, should_embed, compute_content_hash


class EmbeddingQueue:
    """
    File-based embedding queue for crash-resilient processing.

    Pending files survive restart and are reprocessed.
    Each pending file is just a marker with the artifact ID.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.pending_dir = self.memory_dir / "pending_embeddings"
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    def queue_for_embedding(self, artifact_id: str, priority: int = 0) -> bool:
        """
        Queue an artifact for embedding. Non-blocking.

        Args:
            artifact_id: The artifact ID to embed
            priority: Lower = higher priority (default 0)

        Returns:
            True if queued successfully
        """
        try:
            # Filename format: {priority}_{timestamp}_{artifact_id}.pending
            timestamp = _utc_now().strftime("%Y%m%d%H%M%S")
            filename = f"{priority:03d}_{timestamp}_{artifact_id}.pending"
            pending_file = self.pending_dir / filename

            # Write minimal metadata
            metadata = {
                "artifact_id": artifact_id,
                "queued_at": _utc_now_iso(),
                "priority": priority
            }
            pending_file.write_text(json.dumps(metadata), encoding='utf-8')
            return True
        except Exception as e:
            print(f"[WARN] Failed to queue {artifact_id} for embedding: {e}", file=sys.stderr)
            return False

    def get_pending_count(self) -> int:
        """Get count of pending embeddings."""
        return len(list(self.pending_dir.glob("*.pending")))

    def get_pending_items(self, limit: int = 50) -> list[dict]:
        """
        Get pending items sorted by priority then time.

        Returns list of {artifact_id, queued_at, priority, path}
        """
        items = []
        for pending_file in sorted(self.pending_dir.glob("*.pending")):
            if len(items) >= limit:
                break
            try:
                content = pending_file.read_text(encoding='utf-8')
                metadata = json.loads(content)
                metadata["path"] = str(pending_file)
                items.append(metadata)
            except Exception:
                # Corrupted pending file - add with minimal info
                items.append({
                    "artifact_id": pending_file.stem.split("_", 2)[-1] if "_" in pending_file.stem else pending_file.stem,
                    "queued_at": None,
                    "priority": 999,
                    "path": str(pending_file)
                })
        return items

    def mark_complete(self, pending_path: str) -> bool:
        """Mark a pending item as complete (delete the pending file)."""
        try:
            Path(pending_path).unlink()
            return True
        except Exception:
            return False

    def mark_failed(self, pending_path: str, error: str) -> bool:
        """
        Mark a pending item as failed. Moves to failed/ directory.
        """
        try:
            pending_file = Path(pending_path)
            failed_dir = self.pending_dir.parent / "failed_embeddings"
            failed_dir.mkdir(parents=True, exist_ok=True)

            # Read current metadata and add error info
            metadata = json.loads(pending_file.read_text(encoding='utf-8'))
            metadata["failed_at"] = _utc_now_iso()
            metadata["error"] = error

            # Write to failed directory
            failed_file = failed_dir / pending_file.name.replace(".pending", ".failed")
            failed_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

            # Remove pending file
            pending_file.unlink()
            return True
        except Exception:
            return False

    def clear_queue(self) -> int:
        """Clear all pending items. Returns count cleared."""
        count = 0
        for pending_file in self.pending_dir.glob("*.pending"):
            try:
                pending_file.unlink()
                count += 1
            except Exception:
                pass
        return count


class EmbeddingWorker:
    """
    Background worker for processing embedding queue.

    In Phase 1A, this just processes the queue structure without actual embeddings.
    Phase 1B will add FastEmbed integration.
    """

    def __init__(
        self,
        memory_dir: Path,
        artifact_loader: Callable[[str], Optional[dict]],
        embedding_callback: Optional[Callable[[str, str, list[float]], bool]] = None
    ):
        """
        Args:
            memory_dir: Path to memory directory
            artifact_loader: Function to load artifact by ID (returns dict or None)
            embedding_callback: Function to store embedding (id, text, vector) -> success
                               If None, uses placeholder (Phase 1A)
        """
        self.memory_dir = Path(memory_dir)
        self.queue = EmbeddingQueue(memory_dir)
        self.artifact_loader = artifact_loader
        self.embedding_callback = embedding_callback or self._placeholder_embed

        # Processing stats
        self.stats = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "last_run": None
        }

    def _placeholder_embed(self, artifact_id: str, text: str, vector: list[float]) -> bool:
        """
        Placeholder for Phase 1A - just log that we would embed.
        Phase 1B will replace this with actual FastEmbed.
        """
        # For Phase 1A, just record that we processed it
        return True

    def _generate_placeholder_vector(self, text: str) -> list[float]:
        """
        Generate placeholder vector for Phase 1A testing.
        Returns list of 384 zeros (matching BGE-small dimensions).
        """
        return [0.0] * 384

    def process_queue(self, batch_size: int = 10) -> dict:
        """
        Process pending embeddings in batches.

        Args:
            batch_size: Number of items to process per call

        Returns:
            {processed: int, failed: int, skipped: int, remaining: int}
        """
        pending_items = self.queue.get_pending_items(limit=batch_size)
        results = {"processed": 0, "failed": 0, "skipped": 0}

        for item in pending_items:
            artifact_id = item["artifact_id"]
            pending_path = item["path"]

            try:
                # Load artifact
                artifact = self.artifact_loader(artifact_id)
                if not artifact:
                    # Artifact was deleted - skip
                    self.queue.mark_complete(pending_path)
                    results["skipped"] += 1
                    continue

                # Check if should embed
                if not should_embed(artifact):
                    self.queue.mark_complete(pending_path)
                    results["skipped"] += 1
                    continue

                # Get text representation
                text = artifact_to_text(artifact)
                if not text.strip():
                    self.queue.mark_complete(pending_path)
                    results["skipped"] += 1
                    continue

                # Generate embedding (placeholder in Phase 1A)
                vector = self._generate_placeholder_vector(text)

                # Store embedding via callback
                success = self.embedding_callback(artifact_id, text, vector)
                if success:
                    self.queue.mark_complete(pending_path)
                    results["processed"] += 1
                    self.stats["processed"] += 1
                else:
                    self.queue.mark_failed(pending_path, "Embedding callback failed")
                    results["failed"] += 1
                    self.stats["failed"] += 1

            except Exception as e:
                self.queue.mark_failed(pending_path, str(e))
                results["failed"] += 1
                self.stats["failed"] += 1

        results["remaining"] = self.queue.get_pending_count()
        self.stats["last_run"] = _utc_now_iso()
        return results

    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            **self.stats,
            "pending": self.queue.get_pending_count()
        }


def create_embedding_queue(memory_dir: Path) -> EmbeddingQueue:
    """Factory function to create embedding queue."""
    return EmbeddingQueue(memory_dir)


def create_embedding_worker(
    memory_dir: Path,
    artifact_loader: Callable[[str], Optional[dict]],
    embedding_callback: Optional[Callable[[str, str, list[float]], bool]] = None
) -> EmbeddingWorker:
    """Factory function to create embedding worker."""
    return EmbeddingWorker(memory_dir, artifact_loader, embedding_callback)
