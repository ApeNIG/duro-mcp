#!/usr/bin/env python3
"""
Audit Chain Migration Script v1.0.0

Repairs broken audit chain entries by:
1. Recalculating proper SHA-256 hashes
2. Fixing prev_hash linkage
3. Adding chain_version to all entries

GOVERNANCE: This script is a one-off migration tool. Once run, it should be
frozen/versioned. Re-running with different logic rewrites history.

This script creates:
- A backup of the original file
- A meta-audit record in audit_repairs.jsonl
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

# Force UTF-8 output for Windows compatibility
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_VERSION = "1.0.0"
AUDIT_LOG = Path(r"C:\Users\sibag\.agent\memory\logs\deletions.jsonl")
BACKUP_PATH = AUDIT_LOG.with_suffix(".jsonl.backup")
REPAIRS_LOG = AUDIT_LOG.parent / "audit_repairs.jsonl"


def _canonical_json(obj: dict) -> str:
    """
    Deterministic JSON serialization for hashing.
    MUST match runtime in artifacts.py exactly.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_full(s: str) -> str:
    """Full SHA-256 hex digest."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    """Compute SHA-256 of entire file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_entry_hash(entry: dict) -> str:
    """Compute the hash of an entry (excluding entry_hash itself)."""
    hashable = {k: v for k, v in entry.items() if k != "entry_hash"}
    return _sha256_full(_canonical_json(hashable))


def read_entries(path: Path) -> list[dict]:
    """Read JSONL file, returning only valid JSON objects."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip blank lines
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping invalid JSON on line {line_num}: {e}")
    return entries


def write_repair_record(
    backup_hash: str,
    repaired_hash: str,
    entry_count: int,
    entries_modified: int
):
    """Append a meta-audit record to audit_repairs.jsonl"""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "chain_migration",
        "script_version": SCRIPT_VERSION,
        "source_file": str(AUDIT_LOG),
        "backup_file": str(BACKUP_PATH),
        "backup_hash": backup_hash,
        "repaired_hash": repaired_hash,
        "entry_count": entry_count,
        "entries_modified": entries_modified,
        "migration_target": "chain_version_1",
        "notes": "Migrated legacy/broken chain to v1 with proper SHA-256 hashing"
    }

    with open(REPAIRS_LOG, 'a', encoding='utf-8') as f:
        f.write(_canonical_json(record) + "\n")

    return record


def repair_chain():
    """Repair the audit chain by fixing hashes and linkage."""

    # Pre-flight checks
    if not AUDIT_LOG.exists():
        print(f"ERROR: Audit log not found: {AUDIT_LOG}")
        return False

    # Record initial state for mtime/size check
    initial_stat = AUDIT_LOG.stat()
    initial_mtime = initial_stat.st_mtime
    initial_size = initial_stat.st_size

    # Read current entries
    entries = read_entries(AUDIT_LOG)
    if not entries:
        print("ERROR: No valid entries found in audit log")
        return False

    print(f"Found {len(entries)} valid entries")
    print("\n=== BEFORE REPAIR ===")
    for i, e in enumerate(entries, 1):
        prev = e.get('prev_hash')
        curr = e.get('entry_hash')
        ver = e.get('chain_version', '-')
        prev_display = prev[:16] + "..." if prev else "null"
        curr_display = curr[:16] + "..." if curr else "NONE"
        print(f"  Entry {i}: ver={ver} prev={prev_display} entry={curr_display}")

    # Create backup BEFORE any modifications
    with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    backup_hash = _file_hash(BACKUP_PATH)
    print(f"\nBackup saved: {BACKUP_PATH}")
    print(f"Backup hash: {backup_hash}")

    # Check if file was modified during read
    current_stat = AUDIT_LOG.stat()
    if current_stat.st_mtime != initial_mtime or current_stat.st_size != initial_size:
        print("ERROR: Audit log was modified during read. Aborting.")
        return False

    # Repair entries
    repaired = []
    prev_hash = None
    entries_modified = 0

    for i, entry in enumerate(entries):
        # Build new entry without old chain fields
        new_entry = {k: v for k, v in entry.items()
                     if k not in ('prev_hash', 'entry_hash', 'chain_version')}

        # Add chain fields
        new_entry['chain_version'] = 1
        new_entry['prev_hash'] = prev_hash  # None for first entry

        # Compute new entry_hash
        entry_hash = compute_entry_hash(new_entry)
        new_entry['entry_hash'] = entry_hash

        # Check if this entry actually changed
        old_hash = entry.get('entry_hash')
        old_prev = entry.get('prev_hash')
        if old_hash != entry_hash or old_prev != prev_hash:
            entries_modified += 1

        repaired.append(new_entry)
        prev_hash = entry_hash

    # Write repaired entries atomically (write to temp, then rename)
    temp_path = AUDIT_LOG.with_suffix(".jsonl.tmp")
    with open(temp_path, 'w', encoding='utf-8') as f:
        for e in repaired:
            f.write(_canonical_json(e) + "\n")

    # Final mtime check before overwrite
    current_stat = AUDIT_LOG.stat()
    if current_stat.st_mtime != initial_mtime:
        print("ERROR: Audit log was modified during repair. Aborting.")
        temp_path.unlink()
        return False

    # Atomic replace
    temp_path.replace(AUDIT_LOG)
    repaired_hash = _file_hash(AUDIT_LOG)

    print("\n=== AFTER REPAIR ===")
    for i, e in enumerate(repaired, 1):
        prev = e.get('prev_hash')
        curr = e.get('entry_hash', '')
        prev_display = prev[:16] + "..." if prev else "null"
        curr_display = curr[:16] + "..."
        print(f"  Entry {i}: ver=1 prev={prev_display} entry={curr_display}")

    # Write meta-audit record
    repair_record = write_repair_record(
        backup_hash=backup_hash,
        repaired_hash=repaired_hash,
        entry_count=len(entries),
        entries_modified=entries_modified
    )

    print(f"\n=== META-AUDIT RECORD ===")
    print(f"Repair log: {REPAIRS_LOG}")
    print(f"Repaired hash: {repaired_hash}")
    print(f"Entries modified: {entries_modified}/{len(entries)}")

    print("\n[OK] Audit chain repaired successfully")
    return True


if __name__ == "__main__":
    success = repair_chain()
    sys.exit(0 if success else 1)
