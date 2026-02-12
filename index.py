"""
SQLite index layer for Duro artifacts.
Provides fast querying while files remain canonical.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ArtifactIndex:
    """SQLite-backed index for artifact discovery and querying."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    sensitivity TEXT NOT NULL,
                    title TEXT,
                    tags TEXT,
                    source_workflow TEXT,
                    source_urls TEXT,
                    file_path TEXT NOT NULL,
                    hash TEXT NOT NULL
                )
            """)
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON artifacts(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON artifacts(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sensitivity ON artifacts(sensitivity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_workflow ON artifacts(source_workflow)")
            conn.commit()

    def upsert(self, artifact: dict[str, Any], file_path: str, file_hash: str) -> bool:
        """
        Insert or update an artifact in the index.
        Returns True on success.
        """
        try:
            # Extract title based on type
            data = artifact.get("data", {})
            title = self._extract_title(artifact["type"], data)

            # Convert lists to JSON strings for storage
            tags_json = json.dumps(artifact.get("tags", []))
            source = artifact.get("source", {})
            source_urls = data.get("source_urls", [])
            source_urls_json = json.dumps(source_urls) if source_urls else None

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO artifacts (id, type, created_at, updated_at, sensitivity,
                                          title, tags, source_workflow, source_urls, file_path, hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        sensitivity = excluded.sensitivity,
                        title = excluded.title,
                        tags = excluded.tags,
                        source_urls = excluded.source_urls,
                        file_path = excluded.file_path,
                        hash = excluded.hash
                """, (
                    artifact["id"],
                    artifact["type"],
                    artifact["created_at"],
                    artifact.get("updated_at"),
                    artifact["sensitivity"],
                    title,
                    tags_json,
                    source.get("workflow"),
                    source_urls_json,
                    file_path,
                    file_hash
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"Index upsert error: {e}")
            return False

    def _extract_title(self, artifact_type: str, data: dict) -> str:
        """Extract a title/summary from artifact data."""
        if artifact_type == "fact":
            claim = data.get("claim", "")
            return claim[:100] if claim else "Untitled fact"
        elif artifact_type == "decision":
            decision = data.get("decision", "")
            return decision[:100] if decision else "Untitled decision"
        elif artifact_type == "skill":
            return data.get("name", "Untitled skill")
        elif artifact_type == "rule":
            return data.get("name", "Untitled rule")
        elif artifact_type == "log":
            return data.get("message", "")[:100]
        elif artifact_type == "episode":
            goal = data.get("goal", "")
            status = data.get("status", "open")
            return f"[{status}] {goal[:80]}" if goal else "Untitled episode"
        elif artifact_type == "evaluation":
            episode_id = data.get("episode_id", "")
            grade = data.get("grade", "?")
            return f"Eval of {episode_id} - Grade: {grade}"
        elif artifact_type == "skill_stats":
            name = data.get("name", "")
            return f"Stats: {name}" if name else "Untitled skill stats"
        return "Unknown"

    def delete(self, artifact_id: str) -> bool:
        """Remove an artifact from the index."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
                conn.commit()
            return True
        except Exception as e:
            print(f"Index delete error: {e}")
            return False

    def query(
        self,
        artifact_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        sensitivity: Optional[str] = None,
        workflow: Optional[str] = None,
        search_text: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """
        Query artifacts with filters.
        Returns list of index entries (not full artifacts).
        """
        conditions = []
        params = []

        if artifact_type:
            conditions.append("type = ?")
            params.append(artifact_type)

        if sensitivity:
            conditions.append("sensitivity = ?")
            params.append(sensitivity)

        if workflow:
            conditions.append("source_workflow = ?")
            params.append(workflow)

        if since:
            conditions.append("created_at >= ?")
            params.append(since)

        if tags:
            # Search for any matching tag
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            conditions.append(f"({' OR '.join(tag_conditions)})")

        if search_text:
            conditions.append("title LIKE ?")
            params.append(f"%{search_text}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT id, type, created_at, updated_at, sensitivity, title,
                   tags, source_workflow, file_path
            FROM artifacts
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                results = []
                for row in cursor:
                    results.append({
                        "id": row["id"],
                        "type": row["type"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "sensitivity": row["sensitivity"],
                        "title": row["title"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "source_workflow": row["source_workflow"],
                        "file_path": row["file_path"]
                    })
                return results
        except Exception as e:
            print(f"Index query error: {e}")
            return []

    def get_by_id(self, artifact_id: str) -> Optional[dict]:
        """Get a single artifact entry by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM artifacts WHERE id = ?",
                    (artifact_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "type": row["type"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "sensitivity": row["sensitivity"],
                        "title": row["title"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "source_workflow": row["source_workflow"],
                        "source_urls": json.loads(row["source_urls"]) if row["source_urls"] else [],
                        "file_path": row["file_path"],
                        "hash": row["hash"]
                    }
                return None
        except Exception as e:
            print(f"Index get error: {e}")
            return None

    def count(self, artifact_type: Optional[str] = None) -> int:
        """Get count of artifacts, optionally filtered by type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if artifact_type:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM artifacts WHERE type = ?",
                        (artifact_type,)
                    )
                else:
                    cursor = conn.execute("SELECT COUNT(*) FROM artifacts")
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Index count error: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get statistics about the index."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]

                type_counts = {}
                for row in conn.execute("SELECT type, COUNT(*) FROM artifacts GROUP BY type"):
                    type_counts[row[0]] = row[1]

                sensitivity_counts = {}
                for row in conn.execute("SELECT sensitivity, COUNT(*) FROM artifacts GROUP BY sensitivity"):
                    sensitivity_counts[row[0]] = row[1]

                return {
                    "total_artifacts": total,
                    "by_type": type_counts,
                    "by_sensitivity": sensitivity_counts
                }
        except Exception as e:
            print(f"Index stats error: {e}")
            return {"total_artifacts": 0, "by_type": {}, "by_sensitivity": {}}

    def clear(self):
        """Clear all entries from the index. Use with caution."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM artifacts")
            conn.commit()
