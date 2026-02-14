"""
Migration 002: Add temporal indexing columns and relations table.

Creates:
- Index columns for temporal queries (valid_from, valid_until, superseded_by)
- artifact_relations table for explicit relationships

Note: Temporal data lives in JSON (source of truth). These columns are
INDEX-ONLY for fast queries - extracted from JSON on upsert/reindex.
"""

MIGRATION_ID = "002_add_temporal"
DEPENDS_ON = ["001_add_vectors"]


def up(db_path: str) -> dict:
    """
    Apply migration.

    Returns:
        {
            "success": bool,
            "columns_added": list,
            "relations_table_created": bool,
            "message": str
        }
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    result = {
        "success": False,
        "columns_added": [],
        "relations_table_created": False,
        "message": ""
    }

    try:
        # Check if already applied via schema_migrations
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        )
        if cursor.fetchone():
            cursor = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE migration_id = ?", (MIGRATION_ID,)
            )
            if cursor.fetchone():
                result["success"] = True
                result["message"] = "Migration already applied"
                return result

        # Add temporal columns to artifacts table (index-only, extracted from JSON)
        # SQLite doesn't support ADD COLUMN IF NOT EXISTS, so check first
        cursor = conn.execute("PRAGMA table_info(artifacts)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        columns_to_add = [
            ("valid_from", "TEXT"),
            ("valid_until", "TEXT"),
            ("superseded_by", "TEXT"),
            ("importance", "REAL DEFAULT 0.5"),
            ("pinned", "INTEGER DEFAULT 0")
        ]

        for col_name, col_type in columns_to_add:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE artifacts ADD COLUMN {col_name} {col_type}")
                result["columns_added"].append(col_name)

        # Create indexes for temporal queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_valid_from ON artifacts(valid_from)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_valid_until ON artifacts(valid_until)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_superseded_by ON artifacts(superseded_by)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON artifacts(importance)")

        # Create artifact_relations table for explicit relationships
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifact_relations (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(source_id, target_id, relation)
            )
        """)
        result["relations_table_created"] = True

        # Create indexes for relation queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON artifact_relations(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON artifact_relations(target_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON artifact_relations(relation)")

        conn.commit()
        result["success"] = True
        result["message"] = f"Added columns: {result['columns_added']}, relations table created"

    except Exception as e:
        result["message"] = f"Migration failed: {e}"
        conn.rollback()
    finally:
        conn.close()

    return result


def down(db_path: str) -> dict:
    """
    Rollback migration.

    Note: SQLite doesn't support DROP COLUMN before 3.35.0
    This rollback drops the relations table but leaves the columns
    (they're harmless if unused).
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    result = {"success": False, "message": ""}

    try:
        # Drop indexes
        conn.execute("DROP INDEX IF EXISTS idx_valid_from")
        conn.execute("DROP INDEX IF EXISTS idx_valid_until")
        conn.execute("DROP INDEX IF EXISTS idx_superseded_by")
        conn.execute("DROP INDEX IF EXISTS idx_importance")
        conn.execute("DROP INDEX IF EXISTS idx_rel_source")
        conn.execute("DROP INDEX IF EXISTS idx_rel_target")
        conn.execute("DROP INDEX IF EXISTS idx_rel_type")

        # Drop relations table
        conn.execute("DROP TABLE IF EXISTS artifact_relations")

        # Remove migration record (best-effort)
        try:
            conn.execute("DELETE FROM schema_migrations WHERE migration_id = ?", (MIGRATION_ID,))
        except Exception:
            pass

        conn.commit()
        result["success"] = True
        result["message"] = "Migration rolled back (columns retained for SQLite compat)"

    except Exception as e:
        result["message"] = f"Rollback failed: {e}"
        conn.rollback()
    finally:
        conn.close()

    return result


def check_status(db_path: str) -> dict:
    """
    Check migration status.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    status = {
        "applied": False,
        "temporal_columns": [],
        "relations_table_exists": False
    }

    try:
        # Check schema_migrations
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        )
        if cursor.fetchone():
            cursor = conn.execute(
                "SELECT 1 FROM schema_migrations WHERE migration_id = ?", (MIGRATION_ID,)
            )
            status["applied"] = cursor.fetchone() is not None

        # Check columns
        cursor = conn.execute("PRAGMA table_info(artifacts)")
        cols = {row[1] for row in cursor.fetchall()}
        temporal_cols = ["valid_from", "valid_until", "superseded_by", "importance", "pinned"]
        status["temporal_columns"] = [c for c in temporal_cols if c in cols]

        # Check relations table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_relations'"
        )
        status["relations_table_exists"] = cursor.fetchone() is not None

    except Exception:
        pass
    finally:
        conn.close()

    return status


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python m002_add_temporal.py <db_path> [up|down|status]")
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

    print(json.dumps(result, indent=2))
