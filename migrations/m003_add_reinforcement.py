"""
Migration 003: Add reinforcement tracking columns.

Creates:
- last_reinforced_at column for temporal tracking
- reinforcement_count column for frequency tracking
- Index on last_reinforced_at for decay queries

Note: These are INDEX-ONLY columns - truth lives in JSON.
"""

MIGRATION_ID = "003_add_reinforcement"
DEPENDS_ON = ["002_add_temporal"]


def up(db_path: str) -> dict:
    """
    Apply migration.

    Returns:
        {
            "success": bool,
            "columns_added": list,
            "message": str
        }
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    result = {
        "success": False,
        "columns_added": [],
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

        # Check existing columns
        cursor = conn.execute("PRAGMA table_info(artifacts)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        columns_to_add = [
            ("last_reinforced_at", "TEXT"),
            ("reinforcement_count", "INTEGER DEFAULT 0"),
        ]

        for col_name, col_type in columns_to_add:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE artifacts ADD COLUMN {col_name} {col_type}")
                result["columns_added"].append(col_name)

        # Create index for decay queries (finding unreinforced facts)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_last_reinforced ON artifacts(last_reinforced_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reinforcement_count ON artifacts(reinforcement_count)")

        # Record migration
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (migration_id, applied_at) VALUES (?, datetime('now'))",
            (MIGRATION_ID,)
        )

        conn.commit()
        result["success"] = True
        result["message"] = f"Added columns: {result['columns_added']}"

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
    This rollback drops indexes but leaves columns.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    result = {"success": False, "message": ""}

    try:
        # Drop indexes
        conn.execute("DROP INDEX IF EXISTS idx_last_reinforced")
        conn.execute("DROP INDEX IF EXISTS idx_reinforcement_count")

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
        "columns": []
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
        target_cols = ["last_reinforced_at", "reinforcement_count"]
        status["columns"] = [c for c in target_cols if c in cols]

    except Exception:
        pass
    finally:
        conn.close()

    return status


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python m003_add_reinforcement.py <db_path> [up|down|status]")
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
