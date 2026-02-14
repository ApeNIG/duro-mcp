# Known Limitations

This document tracks known architectural limitations in Duro MCP.

---

## Audit Chain Concurrency (Multi-Process)

**Status:** Known limitation
**Severity:** Medium - data integrity risk under multi-process concurrent deletes
**Affected:** `artifacts.py` - `delete_artifact()` using `_AUDIT_CHAIN_LOCK`

### What It Protects

The `_AUDIT_CHAIN_LOCK` (threading.Lock) ensures atomic read-compute-append for audit entries:

```python
# In delete_artifact(), lines 1521-1533:
with _AUDIT_CHAIN_LOCK:
    prev_hash = _read_last_entry_hash(deletions_log)
    log_entry = {
        "prev_hash": prev_hash,
        # ... rest of entry
    }
    _append_with_retry(deletions_log, json.dumps(log_entry) + "\n")
```

### Symptoms (How It Fails)

- **Broken prev_hash chain:** Two entries with same prev_hash (fork)
- **Non-linear history:** Entry N+1's prev_hash doesn't match Entry N's hash
- **Intermittent verification failures:** `verify_chain=True` fails sometimes

### When It Triggers

Two processes calling `delete_artifact()` within the same ~50-100ms window:

```
Process A: reads prev_hash = "abc123"
Process B: reads prev_hash = "abc123"  <- RACE
Process A: appends entry with prev_hash="abc123"
Process B: appends entry with prev_hash="abc123"  <- FORK
```

### Detection

```python
# Query audit log with chain verification
result = duro_query_audit_log(verify_chain=True, limit=20)
# If chain_valid=False, there's a fork
```

The health check does NOT currently detect this automatically. Chain verification is opt-in.

### Mitigations (Ranked by Effort + Safety)

| Option | Effort | Safety | Notes |
|--------|--------|--------|-------|
| **1. SQLite audit table** | Medium | High | `BEGIN IMMEDIATE` transactions guarantee atomicity. Best default. Cross-platform. |
| **2. Single-writer architecture** | Medium | High | One process owns audit writes, others send via IPC. Clean but complex deployment. |
| **3. portalocker library** | Low | Medium | Cross-platform file locking. `pip install portalocker`. Still file-based. |
| **4. Accept the risk** | None | Low | Document that multi-process audit is unsupported. Current state. |

**Recommended:** Option 1 (SQLite table) if multi-process becomes a real use case.

### Non-Goals

We are NOT guaranteeing audit chain integrity under multi-process concurrent writes without implementing a proper cross-process synchronization mechanism (SQLite transactions or equivalent).

For single-process usage (typical MCP server), the current threading.Lock is sufficient.

### Current Mitigation

The `_append_with_retry()` function handles file contention by retrying on IOError with exponential backoff. This reduces the race window but does not eliminate it.

---

## FTS5 Text Population (Async Gap)

**Status:** Known limitation
**Severity:** Low - brief window where semantic search is incomplete
**Affected:** `artifacts.py` line 1368 calling `index.populate_fts_text()`

### What Happens

When an artifact is created via `_store_artifact()`:

```python
# Line 1366-1371 in artifacts.py:
try:
    self.index.populate_fts_text(artifact["id"], artifact)
except Exception as e:
    print(f"[WARN] Failed to populate FTS text: {e}", file=sys.stderr)
```

1. JSON file written (always succeeds first)
2. SQLite index updated (synchronous)
3. FTS text column populated (best-effort, may fail silently)

### Symptoms

- Artifact exists but `duro_semantic_search` doesn't find it by content
- Search finds it by title/tags (trigger-maintained) but not by semantic text

### Detection

```sql
-- Find artifacts missing FTS text
SELECT a.id FROM artifacts a
LEFT JOIN artifact_fts f ON a.id = f.id
WHERE f.text IS NULL OR f.text = ''
```

### Mitigation

- `rebuild_fts()` can fix all gaps
- Triggers maintain title/tags synchronously
- Only semantic text column is best-effort

---

## Vector Embeddings (External Dependency)

**Status:** Graceful degradation
**Severity:** Feature unavailable, not data loss
**Affected:** `embeddings.py`, `index.py`

### Dependencies

```python
# Required for vector search:
import sqlite_vec      # Native SQLite extension
from fastembed import TextEmbedding  # Embedding model
```

### Operating Modes

| Mode | Condition | Behavior |
|------|-----------|----------|
| **Full** | sqlite-vec + fastembed available | Vector + FTS5 hybrid search |
| **FTS-only** | sqlite-vec unavailable | Keyword search only |
| **Fallback** | Neither available | LIKE queries (slowest) |

### vec0 Schema

```sql
-- From migrations/m001_add_vectors.py, line 143:
CREATE VIRTUAL TABLE IF NOT EXISTS artifact_vectors USING vec0(
    artifact_id TEXT PRIMARY KEY,
    embedding FLOAT[384]
)
```

### vec0 KNN Query Pattern

```python
# From index.py vector_search(), line 559:
sql = """
    SELECT v.artifact_id, v.distance
    FROM artifact_vectors v
    WHERE v.embedding MATCH ?
      AND k = ?
    ORDER BY v.distance
"""
```

This uses vec0's optimized KNN index for O(log n) search, not O(n) brute force.

### Detection

Health check reports current mode:
```python
health = duro_health_check()
# health["search"]["vec_available"] = True/False
# health["search"]["fts_available"] = True/False
```

---

## Migration Tracking (Dual Tables)

**Status:** Technical debt
**Severity:** Confusing, not broken
**Affected:** `migrations/runner.py`, `m001_add_vectors.py`

### The Issue

Two migration tracking mechanisms exist:

| Table | Location | Purpose |
|-------|----------|---------|
| `schema_migrations` | runner.py line 27 | Proper tracking with checksums |
| `migrations` | m001_add_vectors.py line 61 | Legacy self-tracking |

### Why It Exists

m001 was written before runner.py existed. It tracks itself internally for idempotency.

### Schema (runner.py)

```sql
-- From migrations/runner.py, line 27:
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL,
    checksum TEXT NOT NULL,
    details TEXT
)
```

### Future

New migrations should use runner.py exclusively. m001 can remain as-is since it works correctly.
