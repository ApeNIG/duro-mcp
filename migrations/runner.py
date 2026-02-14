"""
Migration runner for Duro.

Provides:
- schema_migrations table for tracking applied migrations
- Transaction-wrapped execution
- Checksum verification for drift detection
- Status reporting
"""

import hashlib
import importlib.util
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# Add parent dir to path for time_utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from time_utils import utc_now_iso


def _compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of migration file."""
    content = file_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def _ensure_schema_migrations(conn) -> None:
    """Create schema_migrations table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL,
            checksum TEXT NOT NULL,
            details TEXT
        )
    """)
    conn.commit()


def _load_migration(migration_path: Path):
    """Load a migration module from file path."""
    spec = importlib.util.spec_from_file_location(
        migration_path.stem,
        migration_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_applied_migrations(db_path: str) -> list[dict]:
    """Get list of applied migrations."""
    conn = sqlite3.connect(db_path)
    _ensure_schema_migrations(conn)

    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT migration_id, applied_at, checksum FROM schema_migrations ORDER BY applied_at"
    )

    result = []
    for row in cursor:
        result.append({
            "migration_id": row["migration_id"],
            "applied_at": row["applied_at"],
            "checksum": row["checksum"]
        })

    conn.close()
    return result


def get_pending_migrations(migrations_dir: Path, db_path: str) -> list[dict]:
    """Get list of migrations not yet applied."""
    applied = {m["migration_id"] for m in get_applied_migrations(db_path)}

    pending = []
    for migration_file in sorted(migrations_dir.glob("m[0-9][0-9][0-9]_*.py")):
        module = _load_migration(migration_file)
        migration_id = getattr(module, "MIGRATION_ID", migration_file.stem)

        if migration_id not in applied:
            pending.append({
                "migration_id": migration_id,
                "file": migration_file,
                "checksum": _compute_checksum(migration_file)
            })

    return pending


def run_migration(
    db_path: str,
    migration_path: Path,
    dry_run: bool = False
) -> dict:
    """
    Run a single migration.

    Args:
        db_path: Path to SQLite database
        migration_path: Path to migration file
        dry_run: If True, don't actually apply

    Returns:
        {success, migration_id, message, details}
    """
    result = {
        "success": False,
        "migration_id": None,
        "message": "",
        "details": {}
    }

    try:
        # Load migration module
        module = _load_migration(migration_path)
        migration_id = getattr(module, "MIGRATION_ID", migration_path.stem)
        result["migration_id"] = migration_id

        # Check if already applied
        conn = sqlite3.connect(db_path)
        _ensure_schema_migrations(conn)

        cursor = conn.execute(
            "SELECT checksum FROM schema_migrations WHERE migration_id = ?",
            (migration_id,)
        )
        existing = cursor.fetchone()

        if existing:
            current_checksum = _compute_checksum(migration_path)
            if existing[0] != current_checksum:
                result["message"] = f"Migration modified since applied (checksum mismatch)"
                result["details"]["old_checksum"] = existing[0]
                result["details"]["new_checksum"] = current_checksum
            else:
                result["success"] = True
                result["message"] = "Already applied"
            conn.close()
            return result

        if dry_run:
            result["success"] = True
            result["message"] = "Would apply (dry run)"
            conn.close()
            return result

        # Run migration within transaction
        conn.close()  # Close for migration to open its own connection

        if hasattr(module, "up"):
            migration_result = module.up(db_path)
            result["details"] = migration_result

            if not migration_result.get("success", False):
                result["message"] = migration_result.get("message", "Migration failed")
                return result

        # Record successful migration
        conn = sqlite3.connect(db_path)
        _ensure_schema_migrations(conn)

        import json
        conn.execute(
            """
            INSERT INTO schema_migrations (migration_id, applied_at, checksum, details)
            VALUES (?, ?, ?, ?)
            """,
            (
                migration_id,
                utc_now_iso(),
                _compute_checksum(migration_path),
                json.dumps(result["details"])
            )
        )
        conn.commit()
        conn.close()

        result["success"] = True
        result["message"] = "Applied successfully"

    except Exception as e:
        result["message"] = f"Error: {e}"

    return result


def run_all_pending(
    migrations_dir: Path,
    db_path: str,
    dry_run: bool = False
) -> dict:
    """
    Run all pending migrations in order.

    Returns:
        {success, applied, skipped, failed, details}
    """
    result = {
        "success": True,
        "applied": [],
        "skipped": [],
        "failed": [],
        "details": []
    }

    pending = get_pending_migrations(migrations_dir, db_path)

    for migration in pending:
        migration_result = run_migration(
            db_path,
            migration["file"],
            dry_run=dry_run
        )

        result["details"].append(migration_result)

        if migration_result["success"]:
            if "Already" in migration_result["message"]:
                result["skipped"].append(migration["migration_id"])
            else:
                result["applied"].append(migration["migration_id"])
        else:
            result["failed"].append(migration["migration_id"])
            result["success"] = False
            break  # Stop on first failure

    return result


def get_status(migrations_dir: Path, db_path: str) -> dict:
    """
    Get migration status report.

    Returns:
        {
            applied: [{migration_id, applied_at, checksum}],
            pending: [{migration_id, file, checksum}],
            modified: [{migration_id, old_checksum, new_checksum}]
        }
    """
    applied = get_applied_migrations(db_path)
    pending = get_pending_migrations(migrations_dir, db_path)

    # Check for modified migrations
    applied_map = {m["migration_id"]: m["checksum"] for m in applied}
    modified = []

    for migration_file in migrations_dir.glob("m[0-9][0-9][0-9]_*.py"):
        module = _load_migration(migration_file)
        migration_id = getattr(module, "MIGRATION_ID", migration_file.stem)

        if migration_id in applied_map:
            current_checksum = _compute_checksum(migration_file)
            if applied_map[migration_id] != current_checksum:
                modified.append({
                    "migration_id": migration_id,
                    "old_checksum": applied_map[migration_id],
                    "new_checksum": current_checksum
                })

    return {
        "applied": applied,
        "pending": pending,
        "modified": modified
    }
