"""
Artifact storage layer for Duro memory system.
Handles creation, validation, storage, and retrieval of artifacts.
"""

import hashlib
import json
import os
import random
import string
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from time_utils import utc_now, utc_now_iso
from typing import Any, Optional

from schemas import validate_artifact, TYPE_DIRECTORIES, apply_backward_compat_defaults
from index import ArtifactIndex
from embedding_worker import EmbeddingQueue

# Module-level lock for audit chain atomicity
# Prevents concurrent prev_hash read + append races within a single process.
#
# LIMITATION: This is an IN-PROCESS lock only. It does NOT protect against
# race conditions when multiple processes write to the audit log concurrently.
# For single-process usage (typical MCP server), this is sufficient.
# For multi-process deployments, consider:
#   1. SQLite-backed audit log with BEGIN IMMEDIATE transactions (best)
#   2. Single-writer architecture (one process owns audit writes)
#   3. Cross-platform file locking via portalocker library
#
# See KNOWN_LIMITATIONS.md for details.
_AUDIT_CHAIN_LOCK = threading.Lock()

# Platform-specific file locking
if sys.platform == "win32":
    import msvcrt
    LOCK_EX = msvcrt.LK_NBLCK  # Non-blocking exclusive lock
    LOCK_UN = None  # Not needed on Windows
else:
    import fcntl
    LOCK_EX = fcntl.LOCK_EX
    LOCK_UN = fcntl.LOCK_UN


@contextmanager
def file_lock(file_handle, timeout_ms: int = 5000):
    """
    Cross-platform file locking context manager.
    Acquires exclusive lock on file handle for atomic operations.
    """
    if sys.platform == "win32":
        # Windows locking via msvcrt
        import time
        start = time.time()
        while True:
            try:
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except IOError:
                if (time.time() - start) * 1000 > timeout_ms:
                    raise TimeoutError("Could not acquire file lock")
                time.sleep(0.01)
        try:
            yield
        finally:
            file_handle.seek(0)
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        # Unix locking via fcntl
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


def generate_id(artifact_type: str) -> str:
    """Generate a unique artifact ID."""
    now = utc_now()
    date_part = now.strftime("%Y%m%d")
    time_part = now.strftime("%H%M%S")
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

    # Use short prefixes for some types
    prefix_map = {
        "episode": "ep",
        "evaluation": "eval",
        "skill_stats": "ss"
    }
    prefix = prefix_map.get(artifact_type, artifact_type)
    return f"{prefix}_{date_part}_{time_part}_{random_part}"


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content (truncated for general use)."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def _canonical_json(obj: dict) -> str:
    """Deterministic JSON serialization for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_full(s: str) -> str:
    """Full SHA-256 hex digest for audit chain integrity."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _append_with_retry(
    log_path: Path,
    content: str,
    max_retries: int = 10,
    base_delay_ms: int = 50
) -> tuple[bool, str]:
    """
    Append content to a file with retry on permission/lock errors.

    Uses exponential backoff. Opens file only during write, no persistent handles.

    Args:
        log_path: Path to the log file
        content: String to append (should include newline if needed)
        max_retries: Maximum number of retry attempts
        base_delay_ms: Base delay between retries in milliseconds

    Returns:
        (success, error_message)
    """
    import time

    for attempt in range(max_retries):
        try:
            # Open only for append, write, close immediately
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(content)
                f.flush()
            return True, ""
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 50ms, 100ms, 200ms, 400ms...
                delay = (base_delay_ms * (2 ** attempt)) / 1000.0
                time.sleep(delay)
            else:
                return False, f"Failed after {max_retries} retries: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    return False, "Max retries exceeded"


def _read_last_entry_hash(log_path: Path) -> Optional[str]:
    """
    Read the entry_hash from the last valid JSONL entry.
    Returns None if file doesn't exist, is empty, or last entry has no entry_hash (legacy).
    """
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None

    # Read backwards in chunks to find the last non-empty line
    with log_path.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        buf = b""
        while pos > 0:
            step = min(4096, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            if b"\n" in buf:
                lines = buf.split(b"\n")
                # Try from the end to find last valid JSON with entry_hash
                for raw in reversed(lines):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw.decode("utf-8"))
                        # Return entry_hash if present, None if legacy entry
                        return entry.get("entry_hash")
                    except Exception:
                        continue
        # Fall back: entire file is one line
        try:
            entry = json.loads(buf.decode("utf-8"))
            return entry.get("entry_hash")
        except Exception:
            return None


class ArtifactStore:
    """
    Dual-store artifact manager.
    Files are canonical, SQLite is the query index.
    """

    def __init__(self, memory_dir: str | Path, db_path: str | Path):
        self.memory_dir = Path(memory_dir)
        self.index = ArtifactIndex(db_path)
        # Ensure backup directory exists
        self.backup_dir = self.memory_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        # Initialize embedding queue (Phase 1A)
        self.embedding_queue = EmbeddingQueue(self.memory_dir)

    def _backup_artifact(
        self,
        artifact_id: str,
        operation: str = "delete"
    ) -> tuple[bool, str, str]:
        """
        Create a backup of an artifact before destructive operations.

        Args:
            artifact_id: The artifact ID to backup
            operation: Type of destructive operation (delete, update, etc.)

        Returns:
            (success, backup_path, message)

        Backups are stored in: memory/backups/{artifact_type}/{timestamp}_{artifact_id}.json
        """
        # Get artifact from index
        entry = self.index.get_by_id(artifact_id)
        if not entry:
            return False, "", f"Artifact '{artifact_id}' not found in index"

        file_path = Path(entry["file_path"])
        if not file_path.exists():
            return False, "", f"Artifact file not found: {file_path}"

        # Read the artifact content
        try:
            content = file_path.read_text(encoding='utf-8')
            artifact = json.loads(content)
        except Exception as e:
            return False, "", f"Failed to read artifact: {e}"

        artifact_type = artifact.get("type", "unknown")

        # Create backup directory structure
        type_backup_dir = self.backup_dir / artifact_type
        type_backup_dir.mkdir(parents=True, exist_ok=True)

        # Generate backup filename with timestamp
        timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{timestamp}_{artifact_id}.json"
        backup_path = type_backup_dir / backup_filename

        # Create backup envelope with metadata
        backup_envelope = {
            "backup_meta": {
                "backed_up_at": utc_now_iso(),
                "operation": operation,
                "original_path": str(file_path),
                "original_hash": compute_hash(content)
            },
            "artifact": artifact
        }

        # Write backup
        try:
            backup_content = json.dumps(backup_envelope, indent=2, ensure_ascii=False)
            backup_path.write_text(backup_content, encoding='utf-8')
        except Exception as e:
            return False, "", f"Failed to write backup: {e}"

        return True, str(backup_path), f"Backup created: {backup_path}"

    def restore_from_backup(
        self,
        backup_path: str
    ) -> tuple[bool, str, str]:
        """
        Restore an artifact from a backup file.

        Args:
            backup_path: Path to the backup file

        Returns:
            (success, artifact_id, message)
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            return False, "", f"Backup file not found: {backup_path}"

        try:
            content = backup_file.read_text(encoding='utf-8')
            backup_envelope = json.loads(content)
        except Exception as e:
            return False, "", f"Failed to read backup: {e}"

        artifact = backup_envelope.get("artifact")
        if not artifact:
            return False, "", "Invalid backup format: missing 'artifact' key"

        # Store the artifact (will validate and index)
        success, artifact_id, path = self._store_artifact(artifact)
        if success:
            return True, artifact_id, f"Restored artifact '{artifact_id}' from backup"
        return False, "", f"Failed to restore: {path}"

    def list_backups(
        self,
        artifact_type: Optional[str] = None,
        limit: int = 50
    ) -> list[dict]:
        """
        List available backups.

        Args:
            artifact_type: Filter by artifact type
            limit: Maximum number of backups to return

        Returns:
            List of backup info dicts
        """
        backups = []

        if artifact_type:
            type_dirs = [self.backup_dir / artifact_type]
        else:
            type_dirs = [d for d in self.backup_dir.iterdir() if d.is_dir()]

        for type_dir in type_dirs:
            if not type_dir.exists():
                continue
            for backup_file in sorted(type_dir.glob("*.json"), reverse=True):
                if len(backups) >= limit:
                    break
                try:
                    stat = backup_file.stat()
                    # Extract artifact_id from filename
                    parts = backup_file.stem.split("_", 2)
                    if len(parts) >= 3:
                        artifact_id = parts[2]
                    else:
                        artifact_id = backup_file.stem

                    backups.append({
                        "path": str(backup_file),
                        "artifact_id": artifact_id,
                        "type": type_dir.name,
                        "size_bytes": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception:
                    continue

        return backups[:limit]

    def store_fact(
        self,
        claim: str,
        source_urls: Optional[list[str]] = None,
        snippet: Optional[str] = None,
        confidence: float = 0.5,
        tags: Optional[list[str]] = None,
        workflow: str = "manual",
        sensitivity: str = "public",
        evidence_type: str = "none",
        provenance: str = "unknown"
    ) -> tuple[bool, str, str]:
        """
        Store a fact artifact.
        Returns (success, artifact_id, file_path).

        Args:
            claim: The factual claim being recorded
            source_urls: URLs supporting this fact
            snippet: Relevant excerpt or context
            confidence: Confidence score 0-1
            tags: Searchable tags
            workflow: Source workflow name
            sensitivity: Data sensitivity level
            evidence_type: How evidence supports claim (quote, paraphrase, inference, none)
            provenance: Where fact came from (web, local_file, user, tool_output, unknown)

        Note: High confidence (>=0.8) facts should have source_urls and evidence_type != 'none'.
              See rule_005 (Fact Verification Requirements) for enforcement details.
        """
        # Validate evidence_type
        valid_evidence_types = ["quote", "paraphrase", "inference", "none"]
        if evidence_type not in valid_evidence_types:
            evidence_type = "none"

        # Validate provenance
        valid_provenances = ["web", "local_file", "user", "tool_output", "unknown"]
        if provenance not in valid_provenances:
            provenance = "unknown"

        # Auto-downgrade: High confidence without sources gets capped
        if confidence >= 0.8 and not source_urls:
            # Log this downgrade for awareness
            print(f"[WARN] Fact confidence downgraded from {confidence} to 0.5 (no source_urls)")
            confidence = 0.5

        artifact_id = generate_id("fact")

        artifact = {
            "id": artifact_id,
            "type": "fact",
            "version": "1.1",  # Updated version for new fields
            "created_at": utc_now_iso(),
            "updated_at": None,
            "sensitivity": sensitivity,
            "tags": tags or [],
            "source": {
                "workflow": workflow,
                "run_id": None,
                "tool_trace_path": None
            },
            "data": {
                "claim": claim,
                "source_urls": source_urls or [],
                "snippet": snippet,
                "confidence": confidence,
                "verified": bool(source_urls and evidence_type != "none"),
                "evidence_type": evidence_type,
                "provenance": provenance
            }
        }

        return self._store_artifact(artifact)

    def store_decision(
        self,
        decision: str,
        rationale: str,
        alternatives: Optional[list[str]] = None,
        context: Optional[str] = None,
        reversible: bool = True,
        tags: Optional[list[str]] = None,
        workflow: str = "manual",
        sensitivity: str = "internal"
    ) -> tuple[bool, str, str]:
        """
        Store a decision artifact.
        Returns (success, artifact_id, file_path).
        """
        artifact_id = generate_id("decision")

        artifact = {
            "id": artifact_id,
            "type": "decision",
            "version": "1.0",
            "created_at": utc_now_iso(),
            "updated_at": None,
            "sensitivity": sensitivity,
            "tags": tags or [],
            "source": {
                "workflow": workflow,
                "run_id": None,
                "tool_trace_path": None
            },
            "data": {
                "decision": decision,
                "rationale": rationale,
                "alternatives": alternatives or [],
                "context": context,
                "reversible": reversible,
                "outcome": {
                    "status": "unverified",
                    "verified_at": None,
                    "evidence": [],
                    "confidence": 0.5
                },
                "episodes_used": []
            }
        }

        return self._store_artifact(artifact)

    def validate_decision(
        self,
        decision_id: str,
        status: str,
        episode_id: Optional[str] = None,
        result: Optional[str] = None,
        notes: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Validate or reverse a decision based on evidence.

        Args:
            decision_id: The decision to validate
            status: New status - "validated", "reversed", or "superseded"
            episode_id: Optional episode that provides evidence
            result: Episode result - "success", "partial", or "failed"
            notes: Additional context about the evidence

        Returns (success, message).
        """
        # Get decision
        artifact = self.get_artifact(decision_id)
        if not artifact:
            return False, f"Decision '{decision_id}' not found"

        if artifact.get("type") != "decision":
            return False, f"Artifact '{decision_id}' is not a decision"

        # Validate status
        valid_statuses = ["validated", "reversed", "superseded"]
        if status not in valid_statuses:
            return False, f"Invalid status '{status}'. Must be one of: {valid_statuses}"

        data = artifact["data"]

        # Ensure outcome structure exists
        if "outcome" not in data or data["outcome"] is None:
            data["outcome"] = {
                "status": "unverified",
                "verified_at": None,
                "evidence": [],
                "confidence": 0.5
            }
        elif isinstance(data["outcome"], str):
            # Convert old string format
            old_status = data["outcome"]
            data["outcome"] = {
                "status": old_status if old_status in valid_statuses + ["unverified"] else "unverified",
                "verified_at": None,
                "evidence": [],
                "confidence": 0.5
            }

        # Update status
        now = utc_now_iso()
        data["outcome"]["status"] = status
        data["outcome"]["verified_at"] = now

        # Add evidence if provided
        if episode_id:
            evidence_entry = {"episode_id": episode_id}
            if result:
                evidence_entry["result"] = result
            if notes:
                evidence_entry["notes"] = notes
            data["outcome"]["evidence"].append(evidence_entry)

            # Also track in episodes_used
            if "episodes_used" not in data:
                data["episodes_used"] = []
            if episode_id not in data["episodes_used"]:
                data["episodes_used"].append(episode_id)

        # Adjust confidence based on status
        current_conf = data["outcome"].get("confidence", 0.5)
        if status == "validated":
            # Boost confidence (capped at 0.99)
            new_conf = min(0.99, current_conf + 0.1)
        elif status == "reversed":
            # Drop confidence significantly
            new_conf = max(0.05, current_conf - 0.2)
        elif status == "superseded":
            # Slight drop - decision was replaced, not wrong
            new_conf = max(0.05, current_conf - 0.05)
        else:
            new_conf = current_conf
        data["outcome"]["confidence"] = round(new_conf, 2)

        artifact["data"] = data
        artifact["updated_at"] = now

        return self._update_artifact_file(artifact)

    def link_decision_to_episode(
        self,
        decision_id: str,
        episode_id: str
    ) -> tuple[bool, str]:
        """
        Link a decision to an episode where it was used.
        Returns (success, message).
        """
        # Get decision
        artifact = self.get_artifact(decision_id)
        if not artifact:
            return False, f"Decision '{decision_id}' not found"

        if artifact.get("type") != "decision":
            return False, f"Artifact '{decision_id}' is not a decision"

        data = artifact["data"]

        # Add to episodes_used
        if "episodes_used" not in data:
            data["episodes_used"] = []
        if episode_id not in data["episodes_used"]:
            data["episodes_used"].append(episode_id)

        artifact["data"] = data
        artifact["updated_at"] = utc_now_iso()

        return self._update_artifact_file(artifact)

    def get_active_decisions(
        self,
        limit: int = 5,
        max_age_days: int = 7,
        min_confidence: float = 0.5
    ) -> list[dict]:
        """
        Get recently active decisions for lean context loading.

        Active = recently updated + has some validation signal.

        Args:
            limit: Max decisions to return (default 5)
            max_age_days: Only include decisions updated within N days (default 7)
            min_confidence: Minimum outcome.confidence threshold (default 0.5)

        Returns:
            List of decision summaries sorted by updated_at DESC:
            [
                {
                    "id": "decision_...",
                    "decision": "Use Redis for caching",
                    "confidence": 0.7,
                    "status": "validated",
                    "updated_at": "2026-02-14T10:00:00Z",
                    "age_hours": 2.5
                },
                ...
            ]
        """
        from datetime import timedelta

        cutoff = utc_now() - timedelta(days=max_age_days)
        cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")

        # Query all decisions
        entries = self.index.query(artifact_type="decision", limit=200)

        active = []
        now = utc_now()

        for entry in entries:
            # Load full artifact to check outcome.confidence
            artifact = self.get_artifact(entry["id"])
            if not artifact:
                continue

            data = artifact.get("data", {})
            outcome = data.get("outcome")

            # Handle missing/None outcome or old string format
            if outcome is None or isinstance(outcome, str):
                continue  # Skip - no confidence field

            confidence = outcome.get("confidence", 0.5)
            if confidence < min_confidence:
                continue

            # Check updated_at (fall back to created_at)
            updated_at = artifact.get("updated_at") or artifact.get("created_at", "")
            if not updated_at or updated_at < cutoff_iso:
                continue

            # Calculate age
            try:
                updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                age_hours = round((now - updated_dt).total_seconds() / 3600, 1)
            except Exception:
                age_hours = None

            active.append({
                "id": artifact["id"],
                "decision": data.get("decision", "")[:100],  # Truncate long decisions
                "confidence": confidence,
                "status": outcome.get("status", "unverified"),
                "updated_at": updated_at,
                "age_hours": age_hours
            })

        # Sort by updated_at DESC
        active.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return active[:limit]

    def supersede_fact(
        self,
        old_fact_id: str,
        new_fact_id: str,
        reason: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Mark an old fact as superseded by a new fact.

        Updates the old fact's data:
        - Sets valid_until to current time
        - Sets superseded_by to new_fact_id
        - Creates a 'supersedes' relation in artifact_relations

        Args:
            old_fact_id: The fact being superseded
            new_fact_id: The fact that replaces it
            reason: Optional explanation for the supersession

        Returns (success, message).
        """
        # Get old fact
        old_fact = self.get_artifact(old_fact_id)
        if not old_fact:
            return False, f"Old fact '{old_fact_id}' not found"

        if old_fact.get("type") != "fact":
            return False, f"Artifact '{old_fact_id}' is not a fact"

        # Get new fact (verify it exists)
        new_fact = self.get_artifact(new_fact_id)
        if not new_fact:
            return False, f"New fact '{new_fact_id}' not found"

        if new_fact.get("type") != "fact":
            return False, f"Artifact '{new_fact_id}' is not a fact"

        # Check not already superseded
        if old_fact["data"].get("superseded_by"):
            return False, f"Fact '{old_fact_id}' is already superseded by '{old_fact['data']['superseded_by']}'"

        # Update old fact
        now = utc_now_iso()
        old_fact["data"]["valid_until"] = now
        old_fact["data"]["superseded_by"] = new_fact_id
        old_fact["updated_at"] = now

        # Save old fact
        success, msg = self._update_artifact_file(old_fact)
        if not success:
            return False, f"Failed to update old fact: {msg}"

        # Add relation to index
        self.index.add_relation(
            source_id=new_fact_id,
            target_id=old_fact_id,
            relation="supersedes",
            metadata={"reason": reason} if reason else None
        )

        return True, f"Fact '{old_fact_id}' superseded by '{new_fact_id}'"

    def store_log(
        self,
        event_type: str,
        message: str,
        task: Optional[str] = None,
        outcome: Optional[str] = None,
        error: Optional[str] = None,
        lesson: Optional[str] = None,
        duration_ms: Optional[int] = None,
        tags: Optional[list[str]] = None,
        workflow: str = "session",
        sensitivity: str = "internal"
    ) -> tuple[bool, str, str]:
        """
        Store a log artifact.
        Returns (success, artifact_id, file_path).
        """
        artifact_id = generate_id("log")

        artifact = {
            "id": artifact_id,
            "type": "log",
            "version": "1.0",
            "created_at": utc_now_iso(),
            "updated_at": None,
            "sensitivity": sensitivity,
            "tags": tags or [],
            "source": {
                "workflow": workflow,
                "run_id": None,
                "tool_trace_path": None
            },
            "data": {
                "event_type": event_type,
                "message": message,
                "task": task,
                "outcome": outcome,
                "error": error,
                "lesson": lesson,
                "duration_ms": duration_ms
            }
        }

        return self._store_artifact(artifact)

    def store_episode(
        self,
        goal: str,
        plan: Optional[list[str]] = None,
        context: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        workflow: str = "manual",
        sensitivity: str = "internal"
    ) -> tuple[bool, str, str]:
        """
        Create a new episode artifact.
        Episodes track goal-level work with actions and outcomes.
        Returns (success, artifact_id, file_path).
        """
        artifact_id = generate_id("episode")
        now = utc_now_iso()

        artifact = {
            "id": artifact_id,
            "type": "episode",
            "version": "1.0",
            "created_at": now,
            "updated_at": None,
            "sensitivity": sensitivity,
            "tags": tags or [],
            "source": {
                "workflow": workflow,
                "run_id": None,
                "tool_trace_path": None
            },
            "data": {
                "goal": goal,
                "status": "open",
                "plan": plan or [],
                "context": context or {},
                "actions": [],
                "result": None,
                "result_summary": None,
                "links": {
                    "facts_created": [],
                    "decisions_created": [],
                    "decisions_used": [],
                    "skills_used": [],
                    "evaluation_id": None
                },
                "started_at": now,
                "closed_at": None,
                "duration_mins": None
            }
        }

        return self._store_artifact(artifact)

    def update_episode(
        self,
        episode_id: str,
        updates: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Update an existing episode.
        Allowed updates: actions (append), result, result_summary, links, status.
        Returns (success, message).
        """
        # Get the episode
        artifact = self.get_artifact(episode_id)
        if not artifact:
            return False, f"Episode '{episode_id}' not found"

        if artifact.get("type") != "episode":
            return False, f"Artifact '{episode_id}' is not an episode"

        data = artifact["data"]

        # Append actions if provided
        if "action" in updates:
            action = updates["action"]
            action["timestamp"] = utc_now_iso()
            data["actions"].append(action)

        # Update other fields
        for key in ["result", "result_summary", "status"]:
            if key in updates:
                data[key] = updates[key]

        # Update links (merge)
        if "links" in updates:
            for link_type, link_ids in updates["links"].items():
                if link_type in data["links"]:
                    if isinstance(data["links"][link_type], list):
                        data["links"][link_type].extend(link_ids)
                    else:
                        data["links"][link_type] = link_ids

        # If closing the episode
        if updates.get("status") == "closed":
            data["closed_at"] = utc_now_iso()
            # Calculate duration
            if data.get("started_at"):
                try:
                    started = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
                    closed = datetime.fromisoformat(data["closed_at"].replace("Z", "+00:00"))
                    data["duration_mins"] = round((closed - started).total_seconds() / 60, 2)
                except Exception:
                    pass

            # Hygiene rule: warn if closing with result but no skills_used
            if "warnings" not in data or data["warnings"] is None:
                data["warnings"] = []

            links = data.get("links", {}) or {}
            skills_used = links.get("skills_used", []) or []

            if data.get("result") in ("success", "partial", "failed") and not skills_used:
                warn = "skills_used missing: episode closed with result but no skills recorded"
                if warn not in data["warnings"]:
                    data["warnings"].append(warn)

        artifact["data"] = data
        artifact["updated_at"] = utc_now_iso()

        # Rewrite the file
        return self._update_artifact_file(artifact)

    def _update_artifact_file(self, artifact: dict[str, Any]) -> tuple[bool, str]:
        """
        Rewrite an artifact file with updates.
        Returns (success, message).
        """
        artifact_type = artifact["type"]
        type_dir = TYPE_DIRECTORIES[artifact_type]
        file_path = self.memory_dir / type_dir / f"{artifact['id']}.json"

        try:
            content = json.dumps(artifact, indent=2, ensure_ascii=False)
            file_path.write_text(content, encoding='utf-8')
            file_hash = compute_hash(content)
            self.index.upsert(artifact, str(file_path), file_hash)
            return True, "Updated successfully"
        except Exception as e:
            return False, f"Update failed: {e}"

    def store_evaluation(
        self,
        episode_id: str,
        rubric: dict,
        grade: str,
        memory_updates: Optional[dict] = None,
        next_change: Optional[str] = None,
        tags: Optional[list[str]] = None,
        workflow: str = "manual",
        sensitivity: str = "internal",
        auto_skill_updates: bool = True
    ) -> tuple[bool, str, str]:
        """
        Store an evaluation artifact for an episode.

        Auto-generates skill_stats updates based on episode result and skills_used:
        - success: +0.01 to each skill used
        - partial: +0.005 to each skill used
        - failed: -0.01 to each skill used

        User-provided memory_updates are merged with auto-generated ones.
        Guardrail: only skills listed in episode.links.skills_used can be auto-updated.

        Set auto_skill_updates=False to disable auto-generation.

        Returns (success, artifact_id, file_path).
        """
        # Verify episode exists
        episode = self.get_artifact(episode_id)
        if not episode:
            return False, "", f"Episode '{episode_id}' not found"

        episode_data = episode.get("data", {})
        episode_result = episode_data.get("result")
        skills_used = episode_data.get("links", {}).get("skills_used", [])

        # Auto-generate skill updates based on episode result
        auto_updates = {"reinforce": [], "decay": []}
        if auto_skill_updates and skills_used:
            # Determine delta based on result
            if episode_result == "success":
                delta = 0.01
                update_type = "reinforce"
            elif episode_result == "partial":
                delta = 0.005
                update_type = "reinforce"
            elif episode_result == "failed":
                delta = 0.01  # Will be negated for decay
                update_type = "decay"
            else:
                # No result yet or unknown - skip auto updates
                delta = None
                update_type = None

            if delta and update_type:
                for skill_id in skills_used:
                    # Ensure deterministic ID format
                    artifact_id = f"ss_{skill_id}" if not skill_id.startswith("ss_") else skill_id
                    # Ensure skill_stats exists (idempotent)
                    skill_name = skill_id.replace("_", " ").title()
                    self.ensure_skill_stats(skill_id, skill_name)

                    update_item = {
                        "type": "skill_stats",
                        "id": artifact_id,
                        "delta": delta if update_type == "reinforce" else -delta,
                        "reason": f"auto: episode {episode_result}",
                        "auto_generated": True
                    }
                    auto_updates[update_type].append(update_item)

        # Auto-generate decision updates based on episode result
        # Decisions are hypotheses that survive contact with reality
        decisions_created = episode_data.get("links", {}).get("decisions_created", [])
        decisions_used = episode_data.get("links", {}).get("decisions_used", [])
        # De-duplicate: a decision in both lists gets one update only
        all_decisions = set(decisions_created) | set(decisions_used)

        if all_decisions:
            # Deltas for decisions (smaller than skills - decisions are higher-level)
            if episode_result == "success":
                decision_delta = 0.005
                decision_update_type = "reinforce"
            elif episode_result == "partial":
                decision_delta = 0.002
                decision_update_type = "reinforce"
            elif episode_result == "failed":
                decision_delta = -0.01  # negative: stronger penalty for bad decisions
                decision_update_type = "decay"
            else:
                decision_delta = None
                decision_update_type = None

            if decision_delta is not None and decision_update_type:
                for decision_id in all_decisions:
                    update_item = {
                        "type": "decision",
                        "id": decision_id,
                        "delta": decision_delta,
                        "reason": f"auto: episode {episode_result}",
                        "auto_generated": True
                    }
                    auto_updates[decision_update_type].append(update_item)

        # Merge user-provided updates with auto-generated ones
        final_updates = {"reinforce": [], "decay": []}

        # Add auto-generated updates first
        final_updates["reinforce"].extend(auto_updates["reinforce"])
        final_updates["decay"].extend(auto_updates["decay"])

        # Track guardrail blocks for visibility
        guardrail = {"skipped": 0, "skipped_items": []}

        def _guardrail_skip(item: dict, reason: str):
            guardrail["skipped"] += 1
            guardrail["skipped_items"].append({
                "item_id": item.get("id") or item.get("artifact_id") or "",
                "reason": reason,
                "raw": item
            })

        # Merge user-provided updates (guardrail: filter to allowed items only)
        if memory_updates:
            for item in memory_updates.get("reinforce", []):
                item_id = item.get("id") or item.get("artifact_id", "")
                item_type = item.get("type", "")
                # Guardrail: if it's a skill_stats update, verify it's in skills_used
                if item_type == "skill_stats" or item_id.startswith("ss_"):
                    skill_id = item_id.replace("ss_", "")
                    if skill_id not in skills_used and item_id not in skills_used:
                        _guardrail_skip(item, "skill_stats update blocked: not in episode.links.skills_used")
                        continue
                # Guardrail: if it's a decision update, verify it's in all_decisions
                if item_type == "decision" or item_id.startswith("decision_"):
                    if item_id not in all_decisions:
                        _guardrail_skip(item, "decision update blocked: not in episode.links.decisions_created or decisions_used")
                        continue
                final_updates["reinforce"].append(item)

            for item in memory_updates.get("decay", []):
                item_id = item.get("id") or item.get("artifact_id", "")
                item_type = item.get("type", "")
                # Guardrail: if it's a skill_stats update, verify it's in skills_used
                if item_type == "skill_stats" or item_id.startswith("ss_"):
                    skill_id = item_id.replace("ss_", "")
                    if skill_id not in skills_used and item_id not in skills_used:
                        _guardrail_skip(item, "skill_stats decay blocked: not in episode.links.skills_used")
                        continue
                # Guardrail: if it's a decision update, verify it's in all_decisions
                if item_type == "decision" or item_id.startswith("decision_"):
                    if item_id not in all_decisions:
                        _guardrail_skip(item, "decision decay blocked: not in episode.links.decisions_created or decisions_used")
                        continue
                final_updates["decay"].append(item)

        artifact_id = generate_id("evaluation")
        now = utc_now_iso()

        artifact = {
            "id": artifact_id,
            "type": "evaluation",
            "version": "1.0",
            "created_at": now,
            "updated_at": None,
            "sensitivity": sensitivity,
            "tags": tags or [],
            "source": {
                "workflow": workflow,
                "run_id": None,
                "tool_trace_path": None
            },
            "data": {
                "episode_id": episode_id,
                "rubric": rubric,
                "grade": grade,
                "memory_updates": final_updates,
                "applied": False,
                "applied_at": None,
                "next_change": next_change,
                "guardrail": {
                    "skipped_count": guardrail["skipped"],
                    "skipped_items": guardrail["skipped_items"]
                },
                "breadcrumbs": [f"guardrail_skipped={guardrail['skipped']}"] if guardrail["skipped"] else []
            }
        }

        success, artifact_id, path = self._store_artifact(artifact)

        # Link evaluation to episode
        if success:
            self.update_episode(episode_id, {
                "links": {"evaluation_id": artifact_id}
            })

        return success, artifact_id, path

    def store_skill_stats(
        self,
        skill_id: str,
        name: str,
        confidence: float = 0.5,
        tags: Optional[list[str]] = None,
        workflow: str = "manual"
    ) -> tuple[bool, str, str]:
        """
        Create or update skill stats artifact.
        Returns (success, artifact_id, file_path).
        """
        # Cap confidence to valid range
        confidence = max(0.05, min(0.99, confidence))

        artifact_id = generate_id("skill_stats")

        artifact = {
            "id": artifact_id,
            "type": "skill_stats",
            "version": "1.0",
            "created_at": utc_now_iso(),
            "updated_at": None,
            "sensitivity": "internal",
            "tags": tags or [],
            "source": {
                "workflow": workflow,
                "run_id": None,
                "tool_trace_path": None
            },
            "data": {
                "skill_id": skill_id,
                "name": name,
                "total_uses": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "avg_duration_ms": None,
                "last_used": None,
                "last_outcome": None,
                "confidence": confidence
            }
        }

        return self._store_artifact(artifact)

    def ensure_skill_stats(
        self,
        skill_id: str,
        name: str,
        confidence: float = 0.5,
        tags: Optional[list[str]] = None
    ) -> tuple[bool, str, str]:
        """
        Ensure a skill stats artifact exists with a deterministic ID (ss_{skill_id}).
        If it already exists, returns the existing artifact info.
        If not, creates it with proper schema validation and indexing.

        This is the preferred way to seed core skills - it's idempotent.
        Returns (created, artifact_id, message).
        """
        deterministic_id = f"ss_{skill_id}"

        # Check if already exists
        existing = self.get_artifact(deterministic_id)
        if existing:
            return False, deterministic_id, "Already exists"

        # Cap confidence to valid range
        confidence = max(0.05, min(0.99, confidence))

        artifact = {
            "id": deterministic_id,
            "type": "skill_stats",
            "version": "1.0",
            "created_at": utc_now_iso(),
            "updated_at": None,
            "sensitivity": "internal",
            "tags": tags or ["core", skill_id, "seeded"],
            "source": {
                "workflow": "startup_seed",
                "run_id": None,
                "tool_trace_path": None
            },
            "data": {
                "skill_id": skill_id,
                "name": name,
                "total_uses": 0,
                "successes": 0,
                "failures": 0,
                "success_rate": 0.0,
                "avg_duration_ms": None,
                "last_used": None,
                "last_outcome": None,
                "confidence": confidence
            }
        }

        success, artifact_id, path = self._store_artifact(artifact)
        if success:
            return True, artifact_id, f"Created at {path}"
        return False, "", f"Failed to create: {path}"

    def _normalize_memory_update_item(self, item: dict, is_reinforce: bool) -> dict:
        """
        Normalize a memory update item to support both payload formats.

        New format: {type, id, delta}
        Old format: {artifact_id, reason, delta?}

        Infers type from ID prefix if missing:
          - fact_ → fact
          - decision_ → decision
          - ss_ → skill_stats
          - ep_ → episode
          - eval_ → evaluation

        Defaults delta to +0.01 (reinforce) or -0.01 (decay) if missing.

        When normalizing legacy format, adds audit breadcrumb:
          - normalized_from: "legacy"
          - original_item: copy of the original item
        """
        normalized = {}

        # Detect legacy format: has artifact_id instead of id, or missing type
        is_legacy = "artifact_id" in item or ("id" not in item and "type" not in item)

        # Handle id/artifact_id
        artifact_id = item.get("id") or item.get("artifact_id")
        normalized["id"] = artifact_id

        # Handle type - infer from prefix if missing
        artifact_type = item.get("type")
        type_was_inferred = False
        if not artifact_type and artifact_id:
            type_was_inferred = True
            if artifact_id.startswith("fact_"):
                artifact_type = "fact"
            elif artifact_id.startswith("decision_"):
                artifact_type = "decision"
            elif artifact_id.startswith("ss_"):
                artifact_type = "skill_stats"
            elif artifact_id.startswith("ep_"):
                artifact_type = "episode"
            elif artifact_id.startswith("eval_"):
                artifact_type = "evaluation"
            else:
                # Assume skill_stats for unprefixed IDs (e.g., "planning")
                artifact_type = "skill_stats"
        normalized["type"] = artifact_type

        # Handle delta - default based on reinforce/decay
        delta = item.get("delta") or item.get("delta_confidence")
        delta_was_defaulted = delta is None
        if delta is None:
            delta = 0.01 if is_reinforce else -0.01
        normalized["delta"] = delta

        # Preserve reason for auditability
        if "reason" in item:
            normalized["reason"] = item["reason"]

        # Add audit breadcrumb for legacy format normalization
        if is_legacy or type_was_inferred or delta_was_defaulted:
            normalized["normalized_from"] = "legacy"
            normalized["original_item"] = {k: v for k, v in item.items()}

        return normalized

    def apply_evaluation(
        self,
        evaluation_id: str
    ) -> tuple[bool, str, dict]:
        """
        Apply memory updates from an evaluation.
        - Reinforce: increase confidence/stats for artifacts that performed well
        - Decay: decrease confidence for artifacts that underperformed

        Constraints:
        - Confidence deltas capped at ±0.02 per episode
        - Final confidence capped to [0.05, 0.99]
        - Only apply once per evaluation

        Supports both payload formats:
        - New: {type, id, delta}
        - Old: {artifact_id, reason, delta?}

        Returns (success, message, applied_updates).
        """
        # Get evaluation
        evaluation = self.get_artifact(evaluation_id)
        if not evaluation:
            return False, f"Evaluation '{evaluation_id}' not found", {}

        if evaluation.get("type") != "evaluation":
            return False, f"Artifact '{evaluation_id}' is not an evaluation", {}

        data = evaluation["data"]

        # Check if already applied
        if data.get("applied"):
            return False, "Evaluation already applied", {}

        memory_updates = data.get("memory_updates", {})
        applied = {"reinforced": [], "decayed": [], "errors": []}

        # Track normalized items for audit trail
        normalized_reinforce = []
        normalized_decay = []

        # Apply reinforcements
        for item in memory_updates.get("reinforce", []):
            normalized = self._normalize_memory_update_item(item, is_reinforce=True)
            normalized_reinforce.append(normalized)
            artifact_type = normalized["type"]
            artifact_id = normalized["id"]
            delta = normalized["delta"]
            reason = normalized.get("reason")

            # Cap delta
            delta = min(0.02, max(-0.02, delta))

            result = self._apply_confidence_delta(artifact_type, artifact_id, delta)
            if result["success"]:
                record = {
                    "id": artifact_id,
                    "type": artifact_type,
                    "delta": delta,
                    "new_confidence": result["new_confidence"]
                }
                if reason:
                    record["reason"] = reason
                applied["reinforced"].append(record)
            else:
                applied["errors"].append({
                    "id": artifact_id,
                    "error": result["error"]
                })

        # Apply decays
        for item in memory_updates.get("decay", []):
            normalized = self._normalize_memory_update_item(item, is_reinforce=False)
            normalized_decay.append(normalized)
            artifact_type = normalized["type"]
            artifact_id = normalized["id"]
            delta = normalized["delta"]
            reason = normalized.get("reason")

            # Ensure negative and cap
            if delta > 0:
                delta = -delta
            delta = max(-0.02, delta)

            result = self._apply_confidence_delta(artifact_type, artifact_id, delta)
            if result["success"]:
                record = {
                    "id": artifact_id,
                    "type": artifact_type,
                    "delta": delta,
                    "new_confidence": result["new_confidence"]
                }
                if reason:
                    record["reason"] = reason
                applied["decayed"].append(record)
            else:
                applied["errors"].append({
                    "id": artifact_id,
                    "error": result["error"]
                })

        # Mark evaluation as applied and store normalized items for audit trail
        data["applied"] = True
        data["applied_at"] = utc_now_iso()
        # Replace memory_updates with normalized versions (includes breadcrumbs)
        data["memory_updates"] = {
            "reinforce": normalized_reinforce,
            "decay": normalized_decay
        }
        evaluation["data"] = data
        evaluation["updated_at"] = data["applied_at"]
        self._update_artifact_file(evaluation)

        return True, "Evaluation applied successfully", applied

    def _apply_confidence_delta(
        self,
        artifact_type: str,
        artifact_id: str,
        delta: float
    ) -> dict:
        """
        Apply a confidence delta to an artifact.
        Handles facts, decisions, and skill_stats.
        Returns {"success": bool, "new_confidence": float, "error": str}.
        """
        # Normalize type aliases
        if artifact_type == "skill":
            artifact_type = "skill_stats"

        # For skill_stats, try lookup by skill_id if artifact_id doesn't start with ss_
        if artifact_type == "skill_stats" and not artifact_id.startswith("ss_"):
            resolved = self.get_skill_stats_by_skill_id(artifact_id)
            if resolved:
                artifact_id = resolved["id"]
            else:
                return {"success": False, "error": f"Skill stats for '{artifact_id}' not found"}

        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return {"success": False, "error": f"Artifact '{artifact_id}' not found"}

        data = artifact.get("data", {})

        # Get current confidence field
        if artifact_type == "fact":
            current = data.get("stored_confidence", data.get("confidence", 0.5))
        elif artifact_type == "skill_stats":
            current = data.get("confidence", 0.5)
        elif artifact_type == "decision":
            # Decisions have confidence in outcome object
            outcome = data.get("outcome", {})
            if isinstance(outcome, str):
                # Old format - convert
                outcome = {"status": "unverified", "verified_at": None, "evidence": [], "confidence": 0.5}
                data["outcome"] = outcome
            current = outcome.get("confidence", 0.5)
        else:
            return {"success": False, "error": f"Unknown artifact type for confidence: {artifact_type}"}

        # Apply delta with caps
        new_confidence = current + delta
        new_confidence = max(0.05, min(0.99, new_confidence))

        # Update the artifact
        if artifact_type == "fact":
            data["stored_confidence"] = new_confidence
            data["confidence"] = new_confidence  # Keep in sync
            data["reinforcement_count"] = data.get("reinforcement_count", 0) + (1 if delta > 0 else 0)
        elif artifact_type == "skill_stats":
            data["confidence"] = new_confidence
        elif artifact_type == "decision":
            data["outcome"]["confidence"] = new_confidence
            now = utc_now_iso()
            # Deterministic status rules based on confidence thresholds
            old_status = data["outcome"].get("status", "unverified")
            if new_confidence >= 0.7:
                data["outcome"]["status"] = "validated"
                # Only set verified_at when actually becoming validated
                if old_status != "validated":
                    data["outcome"]["verified_at"] = now
            elif new_confidence <= 0.3:
                data["outcome"]["status"] = "reversed"
            else:
                data["outcome"]["status"] = "unverified"
            # Always track when last evaluated
            data["outcome"]["last_evaluated_at"] = now

        artifact["data"] = data
        artifact["updated_at"] = utc_now_iso()

        success, msg = self._update_artifact_file(artifact)
        if success:
            return {"success": True, "new_confidence": new_confidence, "error": None}
        else:
            return {"success": False, "new_confidence": None, "error": msg}

    def get_artifact_with_defaults(self, artifact_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve artifact with backward compat defaults applied.
        Use this when reading artifacts that may be missing new fields.
        """
        artifact = self.get_artifact(artifact_id)
        if artifact:
            artifact = apply_backward_compat_defaults(artifact)
        return artifact

    def get_skill_stats_by_skill_id(self, skill_id: str) -> Optional[dict[str, Any]]:
        """
        Look up skill stats by the skill_id field (e.g., "web_research").
        This allows using natural skill names in evaluations instead of artifact IDs.
        Returns the full artifact or None if not found.
        """
        # First, try deterministic ID format: ss_{skill_id}
        deterministic_id = f"ss_{skill_id}"
        artifact = self.get_artifact(deterministic_id)
        if artifact and artifact.get("type") == "skill_stats":
            return artifact

        # Fall back to searching all skill_stats
        skill_stats_dir = self.memory_dir / "skill_stats"
        if not skill_stats_dir.exists():
            return None

        for file_path in skill_stats_dir.glob("*.json"):
            try:
                content = file_path.read_text(encoding='utf-8')
                artifact = json.loads(content)
                if artifact.get("data", {}).get("skill_id") == skill_id:
                    return artifact
            except Exception:
                continue

        return None

    def _store_artifact(self, artifact: dict[str, Any]) -> tuple[bool, str, str]:
        """
        Internal method to store any artifact.
        1. Validate against schema
        2. Compute hash
        3. Write file
        4. Update index
        Returns (success, artifact_id, file_path).
        """
        # Validate
        is_valid, error = validate_artifact(artifact)
        if not is_valid:
            return False, "", f"Validation failed: {error}"

        # Determine file path
        artifact_type = artifact["type"]
        type_dir = TYPE_DIRECTORIES[artifact_type]
        artifact_dir = self.memory_dir / type_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        file_path = artifact_dir / f"{artifact['id']}.json"

        # Serialize and hash
        content = json.dumps(artifact, indent=2, ensure_ascii=False)
        file_hash = compute_hash(content)

        # Write file
        try:
            file_path.write_text(content, encoding='utf-8')
        except Exception as e:
            return False, "", f"File write failed: {e}"

        # Update index
        success = self.index.upsert(artifact, str(file_path), file_hash)
        if not success:
            return False, artifact["id"], str(file_path)  # File written but index failed

        # Populate FTS text column (best effort, never block save)
        try:
            self.index.populate_fts_text(artifact["id"], artifact)
        except Exception as e:
            # FTS failure shouldn't block artifact storage
            print(f"[WARN] Failed to populate FTS text: {e}", file=sys.stderr)

        # Queue for embedding (non-blocking, Phase 1A)
        try:
            from embeddings import should_embed
            if should_embed(artifact):
                self.embedding_queue.queue_for_embedding(artifact["id"])
        except Exception as e:
            # Embedding queue failure shouldn't block artifact storage
            print(f"[WARN] Failed to queue for embedding: {e}", file=sys.stderr)

        return True, artifact["id"], str(file_path)

    def get_artifact(self, artifact_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve full artifact by ID.
        Reads from file (canonical source).
        """
        # First check index for file path
        entry = self.index.get_by_id(artifact_id)
        if not entry:
            return None

        file_path = Path(entry["file_path"])
        if not file_path.exists():
            return None

        try:
            content = file_path.read_text(encoding='utf-8')
            return json.loads(content)
        except Exception as e:
            print(f"Error reading artifact: {e}")
            return None

    def query(self, **kwargs) -> list[dict]:
        """
        Query artifacts via index.
        See ArtifactIndex.query for parameters.
        """
        return self.index.query(**kwargs)

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        limit: int = 50
    ) -> list[dict]:
        """List artifacts, optionally filtered by type."""
        return self.index.query(artifact_type=artifact_type, limit=limit)

    def reindex(self) -> tuple[int, int]:
        """
        Rebuild index from files.
        Returns (success_count, error_count).
        """
        self.index.clear()
        success = 0
        errors = 0

        for type_name, dir_name in TYPE_DIRECTORIES.items():
            type_dir = self.memory_dir / dir_name
            if not type_dir.exists():
                continue

            for file_path in type_dir.glob("*.json"):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    artifact = json.loads(content)
                    file_hash = compute_hash(content)

                    if self.index.upsert(artifact, str(file_path), file_hash):
                        success += 1
                    else:
                        errors += 1
                except Exception as e:
                    print(f"Reindex error for {file_path}: {e}")
                    errors += 1

        # Conditionally rebuild FTS semantic text (only if needed)
        fts_status = self.index.get_fts_completeness()
        if fts_status.get("fts_exists") and fts_status.get("missing_text_count", 0) > 0:
            fts_result = self.index.rebuild_fts()
            if not fts_result.get("success"):
                print(f"FTS rebuild warning: {fts_result}")

        return success, errors

    def get_stats(self) -> dict:
        """Get statistics about the artifact store."""
        return self.index.get_stats()

    def delete_artifact(
        self,
        artifact_id: str,
        reason: str,
        force: bool = False,
        skip_backup: bool = False
    ) -> tuple[bool, str]:
        """
        Delete an artifact with guardrails.

        Args:
            artifact_id: The artifact ID to delete
            reason: Required explanation for why this is being deleted
            force: Override sensitivity protection (use with caution)
            skip_backup: Skip creating backup (use with caution)

        Returns:
            (success, message)

        Guardrails:
            - MUST provide reason
            - MUST NOT delete sensitive artifacts unless force=True
            - MUST backup artifact before deletion (unless skip_backup=True)
            - MUST log all deletions to memory/logs/deletions.jsonl
        """
        # Guardrail 1: Reason is required
        if not reason or not reason.strip():
            return False, "Deletion requires a reason"

        # Get artifact from index
        entry = self.index.get_by_id(artifact_id)
        if not entry:
            return False, f"Artifact '{artifact_id}' not found"

        # Load full artifact for logging
        file_path = Path(entry["file_path"])
        if not file_path.exists():
            # File missing but index entry exists - clean up index
            self.index.delete(artifact_id)
            return True, f"Artifact file already missing, cleaned up index entry"

        try:
            content = file_path.read_text(encoding='utf-8')
            artifact = json.loads(content)
        except Exception as e:
            return False, f"Failed to read artifact: {e}"

        # Guardrail 2: Protect sensitive artifacts
        sensitivity = artifact.get("sensitivity", "public")
        if sensitivity == "sensitive" and not force:
            return False, "Cannot delete sensitive artifacts without force=True"

        # Guardrail 3: Backup before deletion (BLOCKS on failure)
        backup_path = None
        if not skip_backup:
            backup_success, backup_path, backup_msg = self._backup_artifact(artifact_id, "delete")
            if not backup_success:
                # Backup failure BLOCKS deletion - we don't delete without a safety net
                return False, f"Backup failed, deletion blocked: {backup_msg}"

        # Compute hash for audit log
        file_hash = compute_hash(content)

        logs_dir = self.memory_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        deletions_log = logs_dir / "deletions.jsonl"

        # Guardrail 4: Write audit entry BEFORE deleting (blocks on failure)
        # Use lock to ensure atomic prev_hash read + append (prevents chain races)
        with _AUDIT_CHAIN_LOCK:
            prev_hash = _read_last_entry_hash(deletions_log) if deletions_log.exists() else None

            log_entry = {
                "timestamp": utc_now_iso(),
                "action": "delete",
                "artifact_id": artifact_id,
                "artifact_type": artifact.get("type"),
                "sensitivity": sensitivity,
                "file_hash": file_hash,
                "reason": reason,
                "force_used": force,
                "backup_path": backup_path,
                "prev_hash": prev_hash,
                "chain_version": 1
            }
            # Compute entry_hash from canonical JSON (full SHA-256)
            entry_hash = _sha256_full(_canonical_json(log_entry))
            log_entry["entry_hash"] = entry_hash

            # Append with retry - blocks deletion if audit can't be written
            audit_success, audit_error = _append_with_retry(
                deletions_log,
                _canonical_json(log_entry) + "\n",
                max_retries=10,
                base_delay_ms=50
            )
        # Lock released here - chain is now consistent

        if not audit_success:
            # Audit failure BLOCKS deletion - unlogged deletions are unacceptable
            return False, f"Audit log write failed, deletion blocked: {audit_error}"

        # Delete the file
        try:
            file_path.unlink()
        except Exception as e:
            return False, f"Failed to delete file: {e}"

        # Remove from index
        self.index.delete(artifact_id)

        msg = f"Artifact '{artifact_id}' deleted. Reason logged."
        if backup_path:
            msg += f" Backup: {backup_path}"
        return True, msg

    def log_audit_repair(
        self,
        backup_path: str,
        backup_hash: str,
        repaired_hash: str,
        entries_before: int,
        entries_after: int,
        reason: str
    ) -> bool:
        """
        Log a repair event to the meta-audit log.
        Called after manual chain repairs to maintain provenance.

        Args:
            backup_path: Path to backup file
            backup_hash: SHA256 of backup file content
            repaired_hash: SHA256 of repaired file content
            entries_before: Number of entries in backup
            entries_after: Number of entries after repair
            reason: Why the repair was needed

        Returns:
            True if logged successfully
        """
        logs_dir = self.memory_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        repairs_log = logs_dir / "audit_repairs.jsonl"

        repair_entry = {
            "timestamp": utc_now_iso(),
            "action": "repair",
            "backup_path": backup_path,
            "backup_hash": backup_hash,
            "repaired_hash": repaired_hash,
            "entries_before": entries_before,
            "entries_after": entries_after,
            "reason": reason
        }

        try:
            with open(repairs_log, "a", encoding="utf-8") as f:
                f.write(_canonical_json(repair_entry) + "\n")
            return True
        except Exception as e:
            print(f"Failed to log repair: {e}")
            return False

    def query_audit_log(
        self,
        event_type: Optional[str] = None,
        artifact_id: Optional[str] = None,
        search_text: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
        verify_chain: bool = False
    ) -> dict:
        """
        Query the audit log (deletions.jsonl).

        Args:
            event_type: Filter by action type (currently only 'delete')
            artifact_id: Filter by specific artifact ID
            search_text: Search in reason field
            since: ISO date string to filter from
            limit: Max entries to return
            verify_chain: If True, verify integrity chain hashes

        Returns:
            {
                "entries": [...],
                "total": int,
                "chain_valid": bool or None (if not verified),
                "chain_details": [...] (per-link status if verify_chain)
            }
        """
        logs_dir = self.memory_dir / "logs"
        deletions_log = logs_dir / "deletions.jsonl"

        if not deletions_log.exists():
            return {"entries": [], "total": 0, "chain_valid": None}

        entries = []
        chain_valid = True if verify_chain else None
        chain_details = [] if verify_chain else None
        prev_entry_hash = None  # None = first entry or legacy boundary

        try:
            with open(deletions_log, "r", encoding="utf-8") as f:
                entry_num = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    entry_num += 1
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        if verify_chain:
                            chain_details.append({
                                "entry": entry_num,
                                "status": "parse_error",
                                "message": "Failed to parse JSON"
                            })
                            chain_valid = False
                        continue

                    # Verify chain if requested
                    if verify_chain:
                        actual_prev = entry.get("prev_hash")
                        current_entry_hash = entry.get("entry_hash")
                        has_chain_fields = current_entry_hash is not None

                        # Hash length info for diagnostics
                        entry_hash_len = len(current_entry_hash) if current_entry_hash else 0
                        prev_hash_len = len(actual_prev) if actual_prev else 0
                        timestamp = entry.get("timestamp", "")[:19]  # Truncate to seconds

                        if not has_chain_fields:
                            # Legacy entry (no chain fields)
                            chain_details.append({
                                "entry": entry_num,
                                "timestamp": timestamp,
                                "entry_hash_len": entry_hash_len,
                                "prev_hash_len": prev_hash_len,
                                "link_ok": None,
                                "status": "legacy",
                                "message": "No chain fields (pre-chain entry)"
                            })
                        elif prev_entry_hash is None and actual_prev is None:
                            # First chained entry after legacy or genesis
                            chain_details.append({
                                "entry": entry_num,
                                "timestamp": timestamp,
                                "entry_hash_len": entry_hash_len,
                                "prev_hash_len": prev_hash_len,
                                "link_ok": True,
                                "status": "valid",
                                "message": "Chain start (prev_hash=null)"
                            })
                        elif actual_prev == prev_entry_hash:
                            # Valid link
                            chain_details.append({
                                "entry": entry_num,
                                "timestamp": timestamp,
                                "entry_hash_len": entry_hash_len,
                                "prev_hash_len": prev_hash_len,
                                "link_ok": True,
                                "status": "valid",
                                "message": f"Links to entry {entry_num - 1}"
                            })
                        else:
                            # Broken link
                            chain_details.append({
                                "entry": entry_num,
                                "timestamp": timestamp,
                                "entry_hash_len": entry_hash_len,
                                "prev_hash_len": prev_hash_len,
                                "link_ok": False,
                                "status": "broken",
                                "message": f"prev_hash mismatch (expected {prev_entry_hash[:16] if prev_entry_hash else 'null'}..., got {actual_prev[:16] if actual_prev else 'null'}...)"
                            })
                            chain_valid = False

                        # Update expected for next iteration (use entry_hash if present)
                        prev_entry_hash = current_entry_hash

                    # Apply filters
                    if event_type and entry.get("action") != event_type:
                        continue
                    if artifact_id and entry.get("artifact_id") != artifact_id:
                        continue
                    if search_text and search_text.lower() not in entry.get("reason", "").lower():
                        continue
                    if since:
                        entry_ts = entry.get("timestamp", "")
                        if entry_ts < since:
                            continue

                    entries.append(entry)

        except Exception as e:
            return {"entries": [], "total": 0, "error": str(e), "chain_valid": None}

        # Apply limit (return most recent first)
        entries = entries[-limit:][::-1]

        result = {
            "entries": entries,
            "total": len(entries),
            "chain_valid": chain_valid
        }
        if chain_details is not None:
            result["chain_details"] = chain_details

        return result
