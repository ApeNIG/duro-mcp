"""
SQLite index layer for Duro artifacts.
Provides fast querying while files remain canonical.

Phase 1B: Adds hybrid search (vector + FTS5) with graceful degradation.
If sqlite-vec is not available, falls back to FTS-only search.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from time_utils import utc_now, utc_now_iso, normalize_iso_z
from typing import Any, Optional

# Check for sqlite-vec availability at module load
_VEC_AVAILABLE = False
try:
    import sqlite_vec
    _VEC_AVAILABLE = True
except ImportError:
    pass


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

            # Extract temporal fields (for facts, but columns exist for all)
            valid_from = data.get("valid_from")
            valid_until = data.get("valid_until")
            superseded_by = data.get("superseded_by")
            importance = data.get("importance", 0.5)
            pinned = 1 if data.get("pinned", False) else 0

            # Extract reinforcement fields (Phase 4)
            last_reinforced_at = data.get("last_reinforced_at")
            reinforcement_count = data.get("reinforcement_count", 0)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO artifacts (id, type, created_at, updated_at, sensitivity,
                                          title, tags, source_workflow, source_urls, file_path, hash,
                                          valid_from, valid_until, superseded_by, importance, pinned,
                                          last_reinforced_at, reinforcement_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        sensitivity = excluded.sensitivity,
                        title = excluded.title,
                        tags = excluded.tags,
                        source_urls = excluded.source_urls,
                        file_path = excluded.file_path,
                        hash = excluded.hash,
                        valid_from = excluded.valid_from,
                        valid_until = excluded.valid_until,
                        superseded_by = excluded.superseded_by,
                        importance = excluded.importance,
                        pinned = excluded.pinned,
                        last_reinforced_at = excluded.last_reinforced_at,
                        reinforcement_count = excluded.reinforcement_count
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
                    file_hash,
                    valid_from,
                    valid_until,
                    superseded_by,
                    importance,
                    pinned,
                    last_reinforced_at,
                    reinforcement_count
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

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        confidence: float = 1.0,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Add a relationship between two artifacts.

        Args:
            source_id: The source artifact (e.g., new fact)
            target_id: The target artifact (e.g., old fact)
            relation: Type of relation (e.g., "supersedes", "references", "supports")
            confidence: Confidence in this relation (0-1)
            metadata: Optional JSON-serializable metadata

        Returns True on success.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if relations table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_relations'"
                )
                if not cursor.fetchone():
                    # Table doesn't exist yet - migration not applied
                    return True  # Silent success, relation just not tracked

                metadata_json = json.dumps(metadata) if metadata else None
                conn.execute("""
                    INSERT OR REPLACE INTO artifact_relations
                    (source_id, target_id, relation, confidence, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    source_id,
                    target_id,
                    relation,
                    confidence,
                    utc_now_iso(),
                    metadata_json
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"Add relation error: {e}")
            return False

    def get_relations(
        self,
        artifact_id: str,
        direction: str = "both",
        relation_type: Optional[str] = None
    ) -> list[dict]:
        """
        Get relations for an artifact.

        Args:
            artifact_id: The artifact to find relations for
            direction: "outgoing" (as source), "incoming" (as target), or "both"
            relation_type: Filter by relation type (e.g., "supersedes")

        Returns list of relation dicts.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if relations table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_relations'"
                )
                if not cursor.fetchone():
                    return []

                conn.row_factory = sqlite3.Row
                relations = []

                if direction in ("outgoing", "both"):
                    query = "SELECT * FROM artifact_relations WHERE source_id = ?"
                    params = [artifact_id]
                    if relation_type:
                        query += " AND relation = ?"
                        params.append(relation_type)
                    cursor = conn.execute(query, params)
                    for row in cursor:
                        relations.append({
                            "source_id": row["source_id"],
                            "target_id": row["target_id"],
                            "relation": row["relation"],
                            "confidence": row["confidence"],
                            "created_at": row["created_at"],
                            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                            "direction": "outgoing"
                        })

                if direction in ("incoming", "both"):
                    query = "SELECT * FROM artifact_relations WHERE target_id = ?"
                    params = [artifact_id]
                    if relation_type:
                        query += " AND relation = ?"
                        params.append(relation_type)
                    cursor = conn.execute(query, params)
                    for row in cursor:
                        relations.append({
                            "source_id": row["source_id"],
                            "target_id": row["target_id"],
                            "relation": row["relation"],
                            "confidence": row["confidence"],
                            "created_at": row["created_at"],
                            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                            "direction": "incoming"
                        })

                return relations
        except Exception as e:
            print(f"Get relations error: {e}")
            return []

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

    def query_current_facts(
        self,
        tags: Optional[list[str]] = None,
        search_text: Optional[str] = None,
        min_importance: Optional[float] = None,
        include_pinned: bool = True,
        limit: int = 100,
        as_of: Optional[str] = None
    ) -> list[dict]:
        """
        Query facts that are currently valid (not superseded AND within time bounds).

        Args:
            tags: Filter by tags
            search_text: Search in title/claim
            min_importance: Filter by importance >= this value
            include_pinned: Include pinned facts regardless of filters
            limit: Maximum results
            as_of: Query as of this time (ISO string). Defaults to now.

        Returns list of current facts (not superseded, valid_from <= now, valid_until > now or NULL).

        Note on pinned: Pinned facts still respect temporal bounds. Pinned = "don't decay",
        not "ignore time". A pinned fact with valid_until in the past is still excluded.
        """
        # Normalize as_of to consistent Z format for reliable text comparison
        now = normalize_iso_z(as_of) if as_of else utc_now_iso()

        # Core temporal conditions:
        # 1. Not superseded
        # 2. valid_from is NULL (always valid) OR valid_from <= now (became valid)
        # 3. valid_until is NULL (still valid) OR valid_until > now (not yet expired)
        conditions = [
            "type = 'fact'",
            "superseded_by IS NULL",
            "(valid_from IS NULL OR valid_from <= ?)",
            "(valid_until IS NULL OR valid_until > ?)"
        ]
        params = [now, now]

        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            conditions.append(f"({' OR '.join(tag_conditions)})")

        if search_text:
            conditions.append("title LIKE ?")
            params.append(f"%{search_text}%")

        if min_importance is not None:
            conditions.append("(importance >= ? OR pinned = 1)")
            params.append(min_importance)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT id, type, created_at, updated_at, sensitivity, title,
                   tags, source_workflow, file_path, importance, pinned
            FROM artifacts
            WHERE {where_clause}
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
        """
        params.append(limit)

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
                        "file_path": row["file_path"],
                        "importance": row["importance"],
                        "pinned": bool(row["pinned"])
                    })
                return results
        except Exception as e:
            print(f"Query current facts error: {e}")
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

    def get_fts_completeness(self) -> dict:
        """
        Check FTS text coverage for health reporting.

        Returns:
            {
                "fts_exists": bool,
                "total_fts_rows": int,
                "missing_text_count": int,  # rows with text='' or NULL
                "coverage_pct": float
            }
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if FTS table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_fts'"
                )
                if not cursor.fetchone():
                    return {"fts_exists": False, "total_fts_rows": 0, "missing_text_count": 0, "coverage_pct": 0.0}

                # Count total FTS rows
                total = conn.execute("SELECT COUNT(*) FROM artifact_fts").fetchone()[0]

                # Count rows with missing text
                missing = conn.execute(
                    "SELECT COUNT(*) FROM artifact_fts WHERE text IS NULL OR text = ''"
                ).fetchone()[0]

                coverage = ((total - missing) / total * 100) if total > 0 else 100.0

                return {
                    "fts_exists": True,
                    "total_fts_rows": total,
                    "missing_text_count": missing,
                    "coverage_pct": round(coverage, 1)
                }
        except Exception as e:
            return {"fts_exists": False, "error": str(e)}

    def get_embedding_stats(self) -> dict:
        """
        Check embedding/vector status for health reporting.

        Returns:
            {
                "vec_extension_available": bool,
                "vec_table_exists": bool,
                "embeddings_count": int,
                "artifacts_count": int,
                "coverage_pct": float,
                "embedding_dim": int or None
            }
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                artifacts_count = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]

                # Check vec extension
                vec_available = self._load_vec_extension(conn)

                # Check vec table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_vectors'"
                )
                vec_table_exists = cursor.fetchone() is not None

                if not vec_table_exists:
                    return {
                        "vec_extension_available": vec_available,
                        "vec_table_exists": False,
                        "embeddings_count": 0,
                        "artifacts_count": artifacts_count,
                        "coverage_pct": 0.0,
                        "embedding_dim": None
                    }

                # Count embeddings
                embeddings_count = conn.execute(
                    "SELECT COUNT(*) FROM artifact_vectors"
                ).fetchone()[0]

                coverage = (embeddings_count / artifacts_count * 100) if artifacts_count > 0 else 0.0

                # Try to get dimension from embedding_state or schema
                embedding_dim = None
                try:
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_state'"
                    )
                    if cursor.fetchone():
                        row = conn.execute(
                            "SELECT model FROM embedding_state LIMIT 1"
                        ).fetchone()
                        if row and "384" in str(row[0]):
                            embedding_dim = 384
                except Exception:
                    pass

                return {
                    "vec_extension_available": vec_available,
                    "vec_table_exists": vec_table_exists,
                    "embeddings_count": embeddings_count,
                    "artifacts_count": artifacts_count,
                    "coverage_pct": round(coverage, 1),
                    "embedding_dim": embedding_dim
                }
        except Exception as e:
            return {"vec_extension_available": False, "error": str(e)}

    # ========================================
    # Phase 1B: Hybrid Search Methods
    # ========================================

    def _load_vec_extension(self, conn) -> bool:
        """Load sqlite-vec extension if available."""
        if not _VEC_AVAILABLE:
            return False
        try:
            import sqlite_vec
            # Must enable extension loading before loading
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            return True
        except Exception as e:
            print(f"[WARN] Failed to load sqlite-vec: {e}")
            return False

    def _has_fts(self, conn) -> bool:
        """Check if FTS5 table exists."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_fts'"
        )
        return cursor.fetchone() is not None

    def _has_vectors(self, conn) -> bool:
        """Check if vector table exists and is usable."""
        if not self._load_vec_extension(conn):
            return False
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_vectors'"
        )
        return cursor.fetchone() is not None

    def upsert_embedding(
        self,
        artifact_id: str,
        embedding: list[float],
        content_hash: str,
        model_name: str = "bge-small-en-v1.5"
    ) -> bool:
        """
        Store or update an embedding for an artifact.

        Args:
            artifact_id: The artifact ID
            embedding: Vector embedding (list of floats)
            content_hash: Hash of content that was embedded
            model_name: Name of the embedding model used

        Returns:
            True on success, False if vectors not available
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if not self._has_vectors(conn):
                    return False

                # Update embedding state
                conn.execute("""
                    INSERT INTO embedding_state (artifact_id, content_hash, embedded_at, model)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        embedded_at = excluded.embedded_at,
                        model = excluded.model
                """, (
                    artifact_id,
                    content_hash,
                    utc_now_iso(),
                    model_name
                ))

                # Store embedding vector using sqlite-vec's serialize format
                # Note: vec0 virtual tables don't support UPSERT, so delete first
                import sqlite_vec
                vec_bytes = sqlite_vec.serialize_float32(embedding)

                conn.execute("DELETE FROM artifact_vectors WHERE artifact_id = ?", (artifact_id,))
                conn.execute("""
                    INSERT INTO artifact_vectors (artifact_id, embedding)
                    VALUES (?, ?)
                """, (artifact_id, vec_bytes))

                conn.commit()
                return True

        except Exception as e:
            print(f"Embedding upsert error: {e}")
            return False

    def delete_embedding(self, artifact_id: str) -> bool:
        """Delete embedding for an artifact."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM embedding_state WHERE artifact_id = ?", (artifact_id,))
                if self._has_vectors(conn):
                    conn.execute("DELETE FROM artifact_vectors WHERE artifact_id = ?", (artifact_id,))
                conn.commit()
                return True
        except Exception:
            return False

    def _escape_fts_query(self, query: str) -> str:
        """
        Escape special FTS5 characters in user query.

        FTS5 special chars: " : - * ( ) { } [ ] ^ ~
        Strategy: wrap each word in quotes for phrase-like matching
        """
        # Remove problematic characters that can't be in quoted phrases
        import re

        # Split into words, escape each, rejoin
        words = query.split()
        safe_words = []

        for word in words:
            # Remove FTS5 operators and special chars
            clean = re.sub(r'[":*(){}[\]^~\-]', ' ', word)
            clean = clean.strip()
            if clean:
                # Wrap in quotes for exact phrase matching
                safe_words.append(f'"{clean}"')

        if not safe_words:
            return '""'  # Empty query

        # Join with OR for flexible matching
        return " OR ".join(safe_words)

    def fts_search(
        self,
        query: str,
        artifact_type: Optional[str] = None,
        limit: int = 50
    ) -> list[dict]:
        """
        Full-text search using FTS5.

        Args:
            query: Search query (will be escaped for safety)
            artifact_type: Optional type filter
            limit: Max results

        Returns:
            List of {id, score, title} dicts sorted by relevance
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if not self._has_fts(conn):
                    # Fall back to LIKE search
                    return self._like_search(conn, query, artifact_type, limit)

                # Escape user query for FTS5 safety
                safe_query = self._escape_fts_query(query)

                # Use subquery pattern: first get FTS matches, then join for details
                sql = """
                    SELECT a.id, a.title, a.type, fts_matches.score
                    FROM (
                        SELECT id, bm25(artifact_fts) as score
                        FROM artifact_fts
                        WHERE artifact_fts MATCH ?
                    ) fts_matches
                    JOIN artifacts a ON a.id = fts_matches.id
                """
                params = [safe_query]

                if artifact_type:
                    sql += " WHERE a.type = ?"
                    params.append(artifact_type)

                sql += " ORDER BY fts_matches.score LIMIT ?"
                params.append(limit)

                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)

                results = []
                for row in cursor:
                    results.append({
                        "id": row["id"],
                        "title": row["title"],
                        "type": row["type"],
                        "score": abs(row["score"]),  # BM25 returns negative scores
                        "source": "fts"
                    })
                return results

        except Exception as e:
            print(f"FTS search error: {e}")
            return []

    def _like_search(
        self,
        conn,
        query: str,
        artifact_type: Optional[str],
        limit: int
    ) -> list[dict]:
        """Fallback LIKE search when FTS not available."""
        sql = """
            SELECT id, title, type
            FROM artifacts
            WHERE (title LIKE ? OR tags LIKE ?)
        """
        params = [f"%{query}%", f"%{query}%"]

        if artifact_type:
            sql += " AND type = ?"
            params.append(artifact_type)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)

        results = []
        for row in cursor:
            results.append({
                "id": row["id"],
                "title": row["title"],
                "type": row["type"],
                "score": 0.5,  # Default score for LIKE matches
                "source": "like"
            })
        return results

    def vector_search(
        self,
        query_embedding: list[float],
        artifact_type: Optional[str] = None,
        limit: int = 50
    ) -> list[dict]:
        """
        Vector similarity search using vec0 KNN queries.

        Uses sqlite-vec's optimized MATCH + k pattern for O(log n) search
        instead of O(n) brute force with vec_distance_cosine.

        Args:
            query_embedding: Query vector
            artifact_type: Optional type filter
            limit: Max results

        Returns:
            List of {id, score, title} dicts sorted by similarity
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if not self._has_vectors(conn):
                    return []

                # Serialize query vector using sqlite-vec format
                import sqlite_vec
                query_vec = sqlite_vec.serialize_float32(query_embedding)

                # Use vec0 KNN query pattern (MATCH + k) for O(log n) performance
                # First get KNN results from vec0, then join for metadata
                # Note: vec0 returns rowid and distance
                sql = """
                    SELECT v.artifact_id, v.distance
                    FROM artifact_vectors v
                    WHERE v.embedding MATCH ?
                      AND k = ?
                    ORDER BY v.distance
                """
                params = [query_vec, limit * 2]  # Get extra for type filter

                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)

                # Collect results and join with artifacts table
                knn_results = list(cursor)

                if not knn_results:
                    return []

                # Get artifact metadata
                artifact_ids = [r["artifact_id"] for r in knn_results]
                placeholders = ",".join("?" * len(artifact_ids))
                meta_sql = f"""
                    SELECT id, title, type
                    FROM artifacts
                    WHERE id IN ({placeholders})
                """
                cursor = conn.execute(meta_sql, artifact_ids)
                metadata = {row["id"]: dict(row) for row in cursor}

                results = []
                for row in knn_results:
                    aid = row["artifact_id"]
                    distance = row["distance"]

                    if aid not in metadata:
                        continue

                    meta = metadata[aid]

                    # Apply type filter if specified
                    if artifact_type and meta["type"] != artifact_type:
                        continue

                    # Convert distance to similarity score
                    # For cosine distance: similarity = 1 - distance
                    similarity = max(0, 1.0 - distance)

                    results.append({
                        "id": aid,
                        "title": meta["title"],
                        "type": meta["type"],
                        "score": similarity,
                        "source": "vector"
                    })

                    if len(results) >= limit:
                        break

                return results

        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        artifact_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 50,
        explain: bool = False
    ) -> dict:
        """
        Hybrid search combining vector and FTS results.

        Uses RRF (Reciprocal Rank Fusion) to merge results.
        Gracefully degrades: vector-only, FTS-only, or LIKE-only.

        Args:
            query: Text query for FTS
            query_embedding: Optional vector for semantic search
            artifact_type: Filter by type
            tags: Filter by tags (any match)
            limit: Max results
            explain: Include score breakdown in results

        Returns:
            {
                "results": [...],
                "mode": "hybrid" | "fts_only" | "vector_only" | "keyword_only",
                "fts_count": int,
                "vector_count": int
            }
        """
        from ranking_config import (
            RANKING,
            calculate_recency_boost,
            calculate_type_weight,
            calculate_confidence_boost,
            explain_score
        )

        # Get FTS results
        fts_results = self.fts_search(query, artifact_type, limit * 2)

        # Get vector results if embedding provided
        vector_results = []
        if query_embedding:
            vector_results = self.vector_search(query_embedding, artifact_type, limit * 2)

        # Determine search mode
        if vector_results and fts_results:
            mode = "hybrid"
        elif vector_results:
            mode = "vector_only"
        elif fts_results and fts_results[0].get("source") == "fts":
            mode = "fts_only"
        else:
            mode = "keyword_only"

        # RRF fusion
        rrf_k = RANKING["rrf_k"]
        scores = {}  # artifact_id -> combined_score

        # Score FTS results
        for rank, result in enumerate(fts_results):
            aid = result["id"]
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if aid not in scores:
                scores[aid] = {"rrf": 0, "fts_rank": None, "vec_rank": None, "meta": result}
            scores[aid]["rrf"] += RANKING["bm25_weight"] * rrf_score
            scores[aid]["fts_rank"] = rank + 1
            scores[aid]["fts_score"] = result["score"]

        # Score vector results
        for rank, result in enumerate(vector_results):
            aid = result["id"]
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if aid not in scores:
                scores[aid] = {"rrf": 0, "fts_rank": None, "vec_rank": None, "meta": result}
            scores[aid]["rrf"] += RANKING["vector_weight"] * rrf_score
            scores[aid]["vec_rank"] = rank + 1
            scores[aid]["vec_score"] = result["score"]

        # Get full artifact info for boosting
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            for aid, data in scores.items():
                # Get artifact details
                cursor = conn.execute(
                    "SELECT * FROM artifacts WHERE id = ?", (aid,)
                )
                row = cursor.fetchone()
                if not row:
                    continue

                # Apply boosts
                created_at = row["created_at"] or ""
                artifact_type = row["type"]

                recency_boost = calculate_recency_boost(created_at)
                type_weight = calculate_type_weight(artifact_type)

                # Calculate final score
                base_score = data["rrf"]
                final_score = base_score * type_weight + recency_boost

                result = {
                    "id": aid,
                    "type": artifact_type,
                    "title": row["title"],
                    "created_at": created_at,
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "file_path": row["file_path"],
                    "search_score": round(final_score, 4)
                }

                if explain:
                    result["score_components"] = {
                        "rrf_base": round(base_score, 4),
                        "type_weight": round(type_weight, 4),
                        "recency_boost": round(recency_boost, 4),
                        "fts_rank": data.get("fts_rank"),
                        "vec_rank": data.get("vec_rank"),
                        "fts_score": round(data.get("fts_score", 0), 4),
                        "vec_score": round(data.get("vec_score", 0), 4)
                    }
                    result["explain"] = explain_score(result["score_components"])

                results.append(result)

        # Sort by score and apply limit
        results.sort(key=lambda x: x["search_score"], reverse=True)

        # Filter by min score threshold
        min_score = RANKING["min_score_threshold"]
        results = [r for r in results if r["search_score"] >= min_score]

        # Apply tag filter if provided
        if tags:
            tag_set = set(tags)
            results = [r for r in results if tag_set & set(r.get("tags", []))]

        results = results[:limit]

        return {
            "results": results,
            "mode": mode,
            "fts_count": len(fts_results),
            "vector_count": len(vector_results),
            "total_candidates": len(scores)
        }

    def populate_fts_text(self, artifact_id: str, artifact: dict) -> bool:
        """
        Update the FTS text column with semantic content from artifact_to_text.

        Called during reindex to populate searchable text for each artifact.
        Triggers only store title/tags, not semantic text.
        """
        try:
            from embeddings import artifact_to_text

            text = artifact_to_text(artifact)
            if not text:
                text = ""

            # Truncate very long text (FTS performance)
            text = text[:2000]

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE artifact_fts SET text = ? WHERE id = ?",
                    (text, artifact_id)
                )
                conn.commit()
            return True
        except Exception as e:
            print(f"FTS text update error for {artifact_id}: {e}")
            return False

    def rebuild_fts(self) -> dict:
        """
        Rebuild FTS index with proper content.

        Drops and recreates FTS table, populates with:
        - id, title from artifacts table
        - tags as space-separated (not JSON)
        - text from artifact_to_text()

        Returns: {success, indexed_count, errors}
        """
        from embeddings import artifact_to_text

        result = {"success": False, "indexed_count": 0, "errors": []}

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Drop and recreate FTS table
                conn.execute("DROP TABLE IF EXISTS artifact_fts")
                conn.execute("""
                    CREATE VIRTUAL TABLE artifact_fts USING fts5(
                        id UNINDEXED,
                        title,
                        tags,
                        text
                    )
                """)

                # Get all artifacts
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT id, title, tags, file_path FROM artifacts")

                for row in cursor:
                    artifact_id = row["id"]
                    title = row["title"] or ""

                    # Convert JSON tags to space-separated
                    tags_json = row["tags"] or "[]"
                    try:
                        tags_list = json.loads(tags_json)
                        tags_str = " ".join(tags_list) if isinstance(tags_list, list) else ""
                    except Exception:
                        tags_str = ""

                    # Get semantic text from artifact file
                    text = ""
                    file_path = row["file_path"]
                    if file_path:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                artifact = json.load(f)
                            text = artifact_to_text(artifact)[:2000]
                        except Exception:
                            pass

                    # Insert into FTS
                    conn.execute(
                        "INSERT INTO artifact_fts (id, title, tags, text) VALUES (?, ?, ?, ?)",
                        (artifact_id, title, tags_str, text)
                    )
                    result["indexed_count"] += 1

                # Recreate triggers
                conn.execute("DROP TRIGGER IF EXISTS artifacts_ai")
                conn.execute("DROP TRIGGER IF EXISTS artifacts_ad")
                conn.execute("DROP TRIGGER IF EXISTS artifacts_au")

                conn.execute("""
                    CREATE TRIGGER artifacts_ai AFTER INSERT ON artifacts BEGIN
                        INSERT INTO artifact_fts(id, title, tags, text)
                        VALUES (
                            NEW.id,
                            NEW.title,
                            REPLACE(REPLACE(REPLACE(NEW.tags, '["', ''), '"]', ''), '","', ' '),
                            ''
                        );
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER artifacts_ad AFTER DELETE ON artifacts BEGIN
                        DELETE FROM artifact_fts WHERE id = OLD.id;
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER artifacts_au AFTER UPDATE ON artifacts BEGIN
                        DELETE FROM artifact_fts WHERE id = OLD.id;
                        INSERT INTO artifact_fts(id, title, tags, text)
                        VALUES (
                            NEW.id,
                            NEW.title,
                            REPLACE(REPLACE(REPLACE(NEW.tags, '["', ''), '"]', ''), '","', ' '),
                            ''
                        );
                    END
                """)

                conn.commit()
                result["success"] = True

        except Exception as e:
            result["errors"].append(str(e))

        return result

    def get_search_capabilities(self) -> dict:
        """
        Get current search capabilities.

        Returns:
            {
                "fts_available": bool,
                "vector_available": bool,
                "embedding_count": int,
                "mode": str
            }
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                fts = self._has_fts(conn)
                vec = self._has_vectors(conn)

                embedding_count = 0
                if vec:
                    cursor = conn.execute("SELECT COUNT(*) FROM embedding_state")
                    embedding_count = cursor.fetchone()[0]

                if vec and fts:
                    mode = "hybrid"
                elif vec:
                    mode = "vector_only"
                elif fts:
                    mode = "fts_only"
                else:
                    mode = "keyword_only"

                return {
                    "fts_available": fts,
                    "vector_available": vec,
                    "embedding_count": embedding_count,
                    "mode": mode
                }
        except Exception:
            return {
                "fts_available": False,
                "vector_available": False,
                "embedding_count": 0,
                "mode": "keyword_only"
            }
