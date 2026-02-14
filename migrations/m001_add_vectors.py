"""
Migration 001: Add vector embeddings and FTS5 tables.

Creates:
- artifact_fts: FTS5 table for full-text search
- artifact_vectors: vec0 table for vector search (if sqlite-vec available)
- embedding_state: tracks which artifacts have been embedded

Graceful degradation: If sqlite-vec is not available, only FTS5 is created.
Search will work in FTS-only mode.
"""

MIGRATION_ID = "001_add_vectors"
DEPENDS_ON = []


def _vec_available(conn) -> bool:
    """Check if sqlite-vec extension is available and loadable."""
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except Exception as e:
        print(f"[DEBUG] sqlite-vec not available: {e}")
        return False


def up(db_path: str) -> dict:
    """
    Apply migration.

    Returns:
        {
            "success": bool,
            "fts5_created": bool,
            "vec_created": bool,
            "message": str
        }
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    result = {"success": False, "fts5_created": False, "vec_created": False, "message": ""}

    try:
        # Check if migration already applied
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'"
        )
        if cursor.fetchone():
            cursor = conn.execute(
                "SELECT 1 FROM migrations WHERE id = ?", (MIGRATION_ID,)
            )
            if cursor.fetchone():
                result["success"] = True
                result["message"] = "Migration already applied"
                return result

        # Create migrations tracking table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL,
                details TEXT
            )
        """)

        # Create FTS5 table for full-text search
        # Using standalone FTS5 (not external content) for simplicity
        # Columns: id (for join), title, tags (space-separated), text (semantic content)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS artifact_fts USING fts5(
                id UNINDEXED,
                title,
                tags,
                text
            )
        """)
        result["fts5_created"] = True

        # Populate FTS5 from existing artifacts
        # Convert JSON tags to space-separated, leave text empty (will be populated by reindex)
        conn.execute("""
            INSERT OR IGNORE INTO artifact_fts(id, title, tags, text)
            SELECT
                id,
                title,
                REPLACE(REPLACE(REPLACE(tags, '["', ''), '"]', ''), '","', ' '),
                ''
            FROM artifacts
        """)

        # Create triggers to keep FTS in sync with artifacts table
        # Note: text column populated by Python reindex, not triggers (triggers can't call Python)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS artifacts_ai AFTER INSERT ON artifacts BEGIN
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
            CREATE TRIGGER IF NOT EXISTS artifacts_ad AFTER DELETE ON artifacts BEGIN
                DELETE FROM artifact_fts WHERE id = OLD.id;
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS artifacts_au AFTER UPDATE ON artifacts BEGIN
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

        # Create embedding state table (tracks what's been embedded)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_state (
                artifact_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                embedded_at TEXT NOT NULL,
                model TEXT NOT NULL
            )
        """)

        # Try to create vector table if sqlite-vec available
        vec_available = _vec_available(conn)
        if vec_available:
            try:
                # vec0 virtual table for 384-dimensional embeddings (all-MiniLM-L6-v2)
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS artifact_vectors USING vec0(
                        artifact_id TEXT PRIMARY KEY,
                        embedding FLOAT[384]
                    )
                """)
                result["vec_created"] = True
            except Exception as e:
                # Vec table creation failed, but FTS still works
                result["message"] = f"Vector table creation failed: {e}. FTS-only mode."
        else:
            result["message"] = "sqlite-vec not available. FTS-only mode enabled."

        # Record migration
        import json
        from datetime import datetime
        conn.execute(
            "INSERT INTO migrations (id, applied_at, details) VALUES (?, ?, ?)",
            (
                MIGRATION_ID,
                datetime.utcnow().isoformat() + "Z",
                json.dumps({
                    "fts5_created": result["fts5_created"],
                    "vec_created": result["vec_created"]
                })
            )
        )

        conn.commit()
        result["success"] = True
        if not result["message"]:
            result["message"] = "Migration applied successfully"

    except Exception as e:
        result["message"] = f"Migration failed: {e}"
        conn.rollback()
    finally:
        conn.close()

    return result


def down(db_path: str) -> dict:
    """
    Rollback migration.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    result = {"success": False, "message": ""}

    try:
        # Drop triggers
        conn.execute("DROP TRIGGER IF EXISTS artifacts_ai")
        conn.execute("DROP TRIGGER IF EXISTS artifacts_ad")
        conn.execute("DROP TRIGGER IF EXISTS artifacts_au")

        # Drop tables
        conn.execute("DROP TABLE IF EXISTS artifact_fts")
        conn.execute("DROP TABLE IF EXISTS artifact_vectors")
        conn.execute("DROP TABLE IF EXISTS embedding_state")

        # Remove migration record
        conn.execute("DELETE FROM migrations WHERE id = ?", (MIGRATION_ID,))

        conn.commit()
        result["success"] = True
        result["message"] = "Migration rolled back"

    except Exception as e:
        result["message"] = f"Rollback failed: {e}"
        conn.rollback()
    finally:
        conn.close()

    return result


def check_status(db_path: str) -> dict:
    """
    Check migration status.

    Returns:
        {
            "applied": bool,
            "fts5_available": bool,
            "vec_available": bool,
            "embedding_count": int
        }
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    status = {
        "applied": False,
        "fts5_available": False,
        "vec_available": False,
        "embedding_count": 0
    }

    try:
        # Check if migration was applied
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'"
        )
        if cursor.fetchone():
            cursor = conn.execute(
                "SELECT 1 FROM migrations WHERE id = ?", (MIGRATION_ID,)
            )
            status["applied"] = cursor.fetchone() is not None

        # Check FTS5
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_fts'"
        )
        status["fts5_available"] = cursor.fetchone() is not None

        # Check vec0
        status["vec_available"] = _vec_available(conn)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_vectors'"
        )
        if not cursor.fetchone():
            status["vec_available"] = False

        # Count embeddings
        if status["applied"]:
            cursor = conn.execute("SELECT COUNT(*) FROM embedding_state")
            status["embedding_count"] = cursor.fetchone()[0]

    except Exception:
        pass
    finally:
        conn.close()

    return status


if __name__ == "__main__":
    # Allow running migration directly
    import sys
    if len(sys.argv) < 2:
        print("Usage: python 001_add_vectors.py <db_path> [up|down|status]")
        sys.exit(1)

    db_path = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "up"

    if action == "up":
        result = up(db_path)
    elif action == "down":
        result = down(db_path)
    elif action == "status":
        result = check_status(db_path)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)

    import json
    print(json.dumps(result, indent=2))
