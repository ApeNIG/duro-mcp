#!/usr/bin/env python3
"""
Duro MCP Server
A Model Context Protocol server for the Duro local AI agent system.

Exposes tools for:
- Loading/saving persistent memory
- Discovering and running skills
- Checking and applying rules
- Loading project context
"""

import asyncio
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from time_utils import utc_now, utc_now_iso
from typing import Any

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Local imports
from memory import DuroMemory
from skills import DuroSkills
from rules import DuroRules
from artifacts import ArtifactStore
from orchestrator import Orchestrator

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)

# Initialize modules
memory = DuroMemory(CONFIG)
skills = DuroSkills(CONFIG)
rules = DuroRules(CONFIG)

# Initialize artifact store (dual-store: files + SQLite index)
MEMORY_DIR = Path(CONFIG["paths"]["memory_dir"])
DB_PATH = MEMORY_DIR / "artifacts.db"
artifact_store = ArtifactStore(MEMORY_DIR, DB_PATH)

# Startup: ensure directories exist, seed core skills, and reindex
# This prevents "file exists but not indexed" ghost artifacts
def _startup_ensure_consistency():
    """Ensure artifact directories exist, core skills are seeded, and index is synced."""
    dirs_to_ensure = ["episodes", "evaluations", "skill_stats", "facts", "decisions", "logs"]
    for dir_name in dirs_to_ensure:
        (MEMORY_DIR / dir_name).mkdir(parents=True, exist_ok=True)

    # Seed core skill stats (idempotent - skips if already exists)
    core_skills = [
        ("web_research", "Web Research"),
        ("source_verification", "Source Verification"),
        ("summarization", "Summarization"),
        ("artifact_creation", "Artifact Creation"),
        ("planning", "Planning"),
    ]
    for skill_id, name in core_skills:
        created, _, _ = artifact_store.ensure_skill_stats(skill_id, name)
        # Only log if created (first run)
        if created:
            print(f"[INFO] Seeded skill_stats: {skill_id}", file=sys.stderr)

    # Reindex to ensure SQLite matches files on disk
    success_count, error_count = artifact_store.reindex()
    if error_count > 0:
        print(f"[WARN] Startup reindex: {success_count} indexed, {error_count} errors", file=sys.stderr)

    # Rebuild FTS to populate semantic text column
    # (Triggers can't call Python, so text column is empty after reindex)
    fts_result = artifact_store.index.rebuild_fts()
    if not fts_result.get("success"):
        print(f"[WARN] FTS rebuild failed: {fts_result}", file=sys.stderr)

_startup_ensure_consistency()


def _startup_health_check() -> dict:
    """
    Run health checks on Duro system components.
    Returns dict with check results for diagnostics.
    """
    checks = {}
    issues = []

    # 1. SQLite integrity check
    try:
        with sqlite3.connect(DB_PATH) as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            checks["sqlite_integrity"] = {
                "status": "ok" if result == "ok" else "error",
                "message": result
            }
            if result != "ok":
                issues.append(f"SQLite integrity check failed: {result}")
    except Exception as e:
        checks["sqlite_integrity"] = {"status": "error", "message": str(e)}
        issues.append(f"SQLite integrity check error: {e}")

    # 2. Index sync check - compare indexed count vs file count
    try:
        indexed_count = artifact_store.index.count()
        file_count = 0
        from schemas import TYPE_DIRECTORIES
        for type_name, dir_name in TYPE_DIRECTORIES.items():
            type_dir = MEMORY_DIR / dir_name
            if type_dir.exists():
                file_count += len(list(type_dir.glob("*.json")))

        drift = abs(indexed_count - file_count)
        sync_ok = drift < 5
        checks["index_sync"] = {
            "status": "ok" if sync_ok else "warning",
            "indexed": indexed_count,
            "files": file_count,
            "drift": drift
        }
        if not sync_ok:
            issues.append(f"Index drift: {drift} (indexed={indexed_count}, files={file_count})")
    except Exception as e:
        checks["index_sync"] = {"status": "error", "message": str(e)}
        issues.append(f"Index sync check error: {e}")

    # 3. Audit chain verification (last 10 entries)
    try:
        audit_result = artifact_store.query_audit_log(limit=10, verify_chain=True)
        chain_valid = audit_result.get("chain_valid")
        if chain_valid is None:
            checks["audit_chain"] = {"status": "ok", "message": "No audit entries (empty chain)"}
        elif chain_valid:
            checks["audit_chain"] = {
                "status": "ok",
                "message": f"Chain valid ({audit_result.get('total', 0)} entries verified)"
            }
        else:
            # Find first broken entry for diagnostics
            chain_details = audit_result.get("chain_details", [])
            first_broken = None
            for detail in chain_details:
                if detail.get("status") != "valid":
                    first_broken = detail
                    break

            checks["audit_chain"] = {
                "status": "error",
                "message": "Chain integrity broken - possible tampering",
                "first_broken_entry": first_broken.get("entry") if first_broken else None,
                "first_broken_timestamp": first_broken.get("timestamp") if first_broken else None,
                "first_broken_reason": first_broken.get("status") if first_broken else None
            }
            issues.append(f"Audit chain integrity broken at entry {first_broken.get('entry') if first_broken else '?'}")
    except Exception as e:
        checks["audit_chain"] = {"status": "error", "message": str(e)}
        issues.append(f"Audit chain check error: {e}")

    # 4. Disk space check
    try:
        total, used, free = shutil.disk_usage(MEMORY_DIR)
        free_gb = free / (1024 ** 3)
        disk_ok = free_gb > 1.0
        checks["disk_space"] = {
            "status": "ok" if disk_ok else "warning",
            "free_gb": round(free_gb, 2),
            "total_gb": round(total / (1024 ** 3), 2)
        }
        if not disk_ok:
            issues.append(f"Low disk space: {round(free_gb, 2)} GB free")
    except Exception as e:
        checks["disk_space"] = {"status": "error", "message": str(e)}
        issues.append(f"Disk space check error: {e}")

    # 5. FTS completeness check
    try:
        fts_stats = artifact_store.index.get_fts_completeness()
        if not fts_stats.get("fts_exists"):
            checks["fts_completeness"] = {
                "status": "warning",
                "message": "FTS table not created",
                "fts_exists": False
            }
            issues.append("FTS table not created - run migration")
        else:
            missing = fts_stats.get("missing_text_count", 0)
            coverage = fts_stats.get("coverage_pct", 100)
            # Warning if >10% missing, error if >50% missing
            if coverage < 50:
                status = "error"
            elif coverage < 90:
                status = "warning"
            else:
                status = "ok"

            checks["fts_completeness"] = {
                "status": status,
                "fts_exists": True,
                "total_fts_rows": fts_stats.get("total_fts_rows", 0),
                "missing_text_count": missing,
                "coverage_pct": coverage
            }
            if missing > 0:
                issues.append(f"FTS has {missing} rows missing semantic text ({coverage}% coverage)")
    except Exception as e:
        checks["fts_completeness"] = {"status": "error", "message": str(e)}
        issues.append(f"FTS completeness check error: {e}")

    # 6. Embedding/vector coverage check
    try:
        emb_stats = artifact_store.index.get_embedding_stats()
        vec_available = emb_stats.get("vec_extension_available", False)
        vec_table = emb_stats.get("vec_table_exists", False)

        if not vec_available:
            checks["embedding_coverage"] = {
                "status": "ok",  # Not an error - graceful degradation
                "message": "sqlite-vec not available - FTS-only mode",
                "vec_extension_available": False,
                "vec_table_exists": False
            }
        elif not vec_table:
            checks["embedding_coverage"] = {
                "status": "warning",
                "message": "sqlite-vec available but vector table not created",
                "vec_extension_available": True,
                "vec_table_exists": False
            }
            issues.append("Vector table not created - run migration")
        else:
            emb_count = emb_stats.get("embeddings_count", 0)
            art_count = emb_stats.get("artifacts_count", 0)
            coverage = emb_stats.get("coverage_pct", 0)

            # Warning if <50% coverage, error if <10%
            if art_count > 0 and coverage < 10:
                status = "warning"
            else:
                status = "ok"

            checks["embedding_coverage"] = {
                "status": status,
                "vec_extension_available": True,
                "vec_table_exists": True,
                "embeddings_count": emb_count,
                "artifacts_count": art_count,
                "coverage_pct": coverage,
                "embedding_dim": emb_stats.get("embedding_dim")
            }
            if art_count > 0 and coverage < 50:
                issues.append(f"Only {coverage}% of artifacts have embeddings ({emb_count}/{art_count})")
    except Exception as e:
        checks["embedding_coverage"] = {"status": "error", "message": str(e)}
        issues.append(f"Embedding coverage check error: {e}")

    # 7. Embedding queue depth with failed count and oldest age
    pending_dir = MEMORY_DIR / "pending_embeddings"
    failed_dir = MEMORY_DIR / "failed_embeddings"
    try:
        pending_count = 0
        oldest_pending_age_mins = 0
        failed_count = 0

        if pending_dir.exists():
            pending_files = list(pending_dir.glob("*.pending"))
            pending_count = len(pending_files)

            # Find oldest pending file age
            if pending_files:
                import os
                now = utc_now().timestamp()
                oldest_mtime = min(os.path.getmtime(f) for f in pending_files)
                oldest_pending_age_mins = int((now - oldest_mtime) / 60)

        if failed_dir.exists():
            failed_count = len(list(failed_dir.glob("*.failed")))

        # Queue is concerning if: >100 pending, or oldest >30 mins, or any failed
        queue_warning = pending_count > 100 or oldest_pending_age_mins > 30 or failed_count > 0
        queue_error = pending_count > 500 or oldest_pending_age_mins > 120

        if queue_error:
            status = "error"
        elif queue_warning:
            status = "warning"
        else:
            status = "ok"

        checks["embedding_queue"] = {
            "status": status,
            "pending": pending_count,
            "failed": failed_count,
            "oldest_pending_age_mins": oldest_pending_age_mins
        }

        if pending_count > 100:
            issues.append(f"Embedding queue backlog: {pending_count} pending")
        if oldest_pending_age_mins > 30:
            issues.append(f"Embedding queue stale: oldest item {oldest_pending_age_mins} mins old")
        if failed_count > 0:
            issues.append(f"Embedding queue has {failed_count} failed items")

    except Exception as e:
        checks["embedding_queue"] = {"status": "error", "message": str(e)}
        issues.append(f"Embedding queue check error: {e}")

    # Overall status
    has_errors = any(c.get("status") == "error" for c in checks.values())
    has_warnings = any(c.get("status") == "warning" for c in checks.values())

    return {
        "timestamp": utc_now_iso(),
        "overall": "error" if has_errors else ("warning" if has_warnings else "ok"),
        "checks": checks,
        "issues": issues
    }


# Run health check at startup and log any issues
_health_result = _startup_health_check()
if _health_result["issues"]:
    print(f"[WARN] Startup health check found issues:", file=sys.stderr)
    for issue in _health_result["issues"]:
        print(f"  - {issue}", file=sys.stderr)
else:
    print(f"[INFO] Startup health check passed", file=sys.stderr)


# Initialize orchestrator
orchestrator = Orchestrator(MEMORY_DIR, rules, skills, artifact_store)

# Create MCP server
server = Server("duro-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available Duro tools."""
    return [
        # Memory tools
        Tool(
            name="duro_load_context",
            description="Load full Duro context at session start. Returns soul, core memory, today's memory, and recent memories. Call this at the beginning of every session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_soul": {
                        "type": "boolean",
                        "description": "Include soul.md personality config",
                        "default": True
                    },
                    "recent_days": {
                        "type": "integer",
                        "description": "Number of days of recent memory to load",
                        "default": 3
                    }
                }
            }
        ),
        Tool(
            name="duro_save_memory",
            description="Save content to today's memory log. Use this to persist important information, learnings, or session notes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to save"
                    },
                    "section": {
                        "type": "string",
                        "description": "Section header for the entry",
                        "default": "Session Log"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="duro_save_learning",
            description="Save a specific learning or insight to memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "learning": {
                        "type": "string",
                        "description": "The learning or insight"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category (e.g., Technical, Process, User Preference)",
                        "default": "General"
                    }
                },
                "required": ["learning"]
            }
        ),
        Tool(
            name="duro_log_task",
            description="Log a completed task with its outcome.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Description of the task"
                    },
                    "outcome": {
                        "type": "string",
                        "description": "Result or outcome of the task"
                    }
                },
                "required": ["task", "outcome"]
            }
        ),
        Tool(
            name="duro_log_failure",
            description="Log a failure with lesson learned. Failures are valuable for building rules.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What was attempted"
                    },
                    "error": {
                        "type": "string",
                        "description": "What went wrong"
                    },
                    "lesson": {
                        "type": "string",
                        "description": "What to do differently next time"
                    }
                },
                "required": ["task", "error", "lesson"]
            }
        ),
        Tool(
            name="duro_compress_logs",
            description="Compress old memory logs into summaries. Archives raw logs and creates compact summaries for faster context loading. Run this periodically to keep context size manageable.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="duro_query_archive",
            description="Search or retrieve archived raw memory logs. Use when you need full detail from past sessions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Specific date to retrieve (YYYY-MM-DD format)"
                    },
                    "search": {
                        "type": "string",
                        "description": "Search query to find in archives"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 5
                    }
                }
            }
        ),
        Tool(
            name="duro_list_archives",
            description="List all available archived memory logs with their sizes.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # Skills tools
        Tool(
            name="duro_list_skills",
            description="List all available Duro skills with their metadata.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="duro_find_skills",
            description="Find skills matching keywords. Use this to discover which skill to use for a task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to search for"
                    }
                },
                "required": ["keywords"]
            }
        ),
        Tool(
            name="duro_run_skill",
            description="Execute a Duro skill by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name or ID of the skill to run"
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments to pass to the skill",
                        "default": {}
                    }
                },
                "required": ["skill_name"]
            }
        ),
        Tool(
            name="duro_get_skill_code",
            description="Get the source code of a skill for inspection or modification.",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name or ID of the skill"
                    }
                },
                "required": ["skill_name"]
            }
        ),

        # Rules tools
        Tool(
            name="duro_check_rules",
            description="Check which rules apply to a given task. Call this before starting work to get relevant constraints and guidance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Description of the task to check rules against"
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="duro_list_rules",
            description="List all Duro rules.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # Project tools
        Tool(
            name="duro_get_project",
            description="Load context for a specific project.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project"
                    }
                },
                "required": ["project_name"]
            }
        ),
        Tool(
            name="duro_list_projects",
            description="List all tracked projects.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # System tools
        Tool(
            name="duro_status",
            description="Get Duro system status and statistics.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="duro_health_check",
            description="Run health diagnostics on Duro system. Checks: SQLite integrity, index sync, audit chain, disk space, embedding queue. Use this to diagnose issues before they become problems.",
            inputSchema={
                "type": "object",
                "properties": {
                    "verbose": {
                        "type": "boolean",
                        "description": "Include detailed check information",
                        "default": False
                    }
                }
            }
        ),

        # Temporal tools (Phase 2)
        Tool(
            name="duro_supersede_fact",
            description="Mark an old fact as superseded by a new fact. Updates the old fact with valid_until and superseded_by. Use when information becomes outdated.",
            inputSchema={
                "type": "object",
                "properties": {
                    "old_fact_id": {
                        "type": "string",
                        "description": "The fact ID being superseded"
                    },
                    "new_fact_id": {
                        "type": "string",
                        "description": "The fact ID that replaces it"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional explanation for the supersession"
                    }
                },
                "required": ["old_fact_id", "new_fact_id"]
            }
        ),
        Tool(
            name="duro_get_related",
            description="Get artifacts related to a given artifact. Returns both explicit relations (supersedes, references) and optionally semantic neighbors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "The artifact to find relations for"
                    },
                    "relation_type": {
                        "type": "string",
                        "description": "Filter by relation type (e.g., 'supersedes', 'references')"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["outgoing", "incoming", "both"],
                        "description": "Direction of relations to include",
                        "default": "both"
                    }
                },
                "required": ["artifact_id"]
            }
        ),

        # Auto-capture & Proactive recall tools (Phase 3)
        Tool(
            name="duro_proactive_recall",
            description="Proactively recall relevant memories for current task context. Uses hot path classification + hybrid search to surface memories you might need. Call this at the start of complex tasks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Current task or conversation context to find relevant memories for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum memories to return",
                        "default": 10
                    },
                    "include_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to specific artifact types (e.g., ['fact', 'decision'])"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "If true, always search even if hot path classifier says no",
                        "default": False
                    }
                },
                "required": ["context"]
            }
        ),
        Tool(
            name="duro_extract_learnings",
            description="Auto-extract learnings, facts, and decisions from conversation text. Useful for capturing insights at session end or from tool outputs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Conversation or text to extract learnings from"
                    },
                    "auto_save": {
                        "type": "boolean",
                        "description": "If true, automatically save extracted items as artifacts",
                        "default": False
                    }
                },
                "required": ["text"]
            }
        ),

        # Decay & Maintenance tools (Phase 4)
        Tool(
            name="duro_apply_decay",
            description="Apply time-based confidence decay to unreinforced facts. Pinned facts are never decayed. Run with dry_run=true first to preview changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, calculate decay but don't modify facts",
                        "default": True
                    },
                    "min_importance": {
                        "type": "number",
                        "description": "Only decay facts with importance >= this value",
                        "default": 0
                    },
                    "include_stale_report": {
                        "type": "boolean",
                        "description": "Include list of stale high-importance facts",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="duro_reembed",
            description="Re-queue artifacts for embedding. Use after model upgrade, schema change, or to fix bad embeddings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific artifact IDs to re-embed (if not specified, uses filters)"
                    },
                    "artifact_type": {
                        "type": "string",
                        "description": "Re-embed all artifacts of this type"
                    },
                    "all": {
                        "type": "boolean",
                        "description": "Re-embed ALL artifacts (use with caution)",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="duro_maintenance_report",
            description="Generate a maintenance report for memory health. Shows: total facts, % pinned, % stale, top stale high-importance facts, embedding/FTS coverage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_stale_list": {
                        "type": "boolean",
                        "description": "Include list of top stale high-importance facts",
                        "default": True
                    },
                    "top_n_stale": {
                        "type": "integer",
                        "description": "Number of stale facts to list",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="duro_reinforce_fact",
            description="Reinforce a fact - mark it as recently used/confirmed. Resets decay clock and increments reinforcement count.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fact_id": {
                        "type": "string",
                        "description": "The fact ID to reinforce"
                    }
                },
                "required": ["fact_id"]
            }
        ),

        # Artifact tools (structured memory)
        Tool(
            name="duro_store_fact",
            description="Store a fact with source attribution. Facts are claims with evidence. High confidence (>=0.8) requires source_urls and evidence_type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The factual claim being recorded"
                    },
                    "source_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs supporting this fact"
                    },
                    "snippet": {
                        "type": "string",
                        "description": "Relevant excerpt or context"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0-1",
                        "default": 0.5
                    },
                    "evidence_type": {
                        "type": "string",
                        "enum": ["quote", "paraphrase", "inference", "none"],
                        "description": "How evidence supports the claim: quote (direct), paraphrase (reworded), inference (derived), none",
                        "default": "none"
                    },
                    "provenance": {
                        "type": "string",
                        "enum": ["web", "local_file", "user", "tool_output", "unknown"],
                        "description": "Where the fact came from",
                        "default": "unknown"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Searchable tags"
                    },
                    "workflow": {
                        "type": "string",
                        "description": "Source workflow name",
                        "default": "manual"
                    },
                    "sensitivity": {
                        "type": "string",
                        "enum": ["public", "internal", "sensitive"],
                        "default": "public"
                    }
                },
                "required": ["claim"]
            }
        ),
        Tool(
            name="duro_store_decision",
            description="Store a decision with rationale. Decisions capture choices made and why.",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "description": "The decision made"
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this decision was made"
                    },
                    "alternatives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Other options considered"
                    },
                    "context": {
                        "type": "string",
                        "description": "Situation context"
                    },
                    "reversible": {
                        "type": "boolean",
                        "description": "Whether decision can be undone",
                        "default": True
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Searchable tags"
                    },
                    "workflow": {
                        "type": "string",
                        "description": "Source workflow name",
                        "default": "manual"
                    },
                    "sensitivity": {
                        "type": "string",
                        "enum": ["public", "internal", "sensitive"],
                        "default": "internal"
                    }
                },
                "required": ["decision", "rationale"]
            }
        ),
        Tool(
            name="duro_validate_decision",
            description="Validate or reverse a decision based on evidence. Use this to record whether a decision worked out.",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {
                        "type": "string",
                        "description": "The decision ID to validate"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["validated", "reversed", "superseded"],
                        "description": "New status: validated (worked), reversed (didn't work), superseded (replaced)"
                    },
                    "episode_id": {
                        "type": "string",
                        "description": "Optional episode ID that provides evidence"
                    },
                    "result": {
                        "type": "string",
                        "enum": ["success", "partial", "failed"],
                        "description": "Episode result as evidence"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional context about the evidence"
                    }
                },
                "required": ["decision_id", "status"]
            }
        ),
        Tool(
            name="duro_link_decision",
            description="Link a decision to an episode where it was used/tested.",
            inputSchema={
                "type": "object",
                "properties": {
                    "decision_id": {
                        "type": "string",
                        "description": "The decision ID"
                    },
                    "episode_id": {
                        "type": "string",
                        "description": "The episode ID where this decision was applied"
                    }
                },
                "required": ["decision_id", "episode_id"]
            }
        ),
        Tool(
            name="duro_query_memory",
            description="Query artifacts from memory. SQLite-backed fast search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_type": {
                        "type": "string",
                        "enum": ["fact", "decision", "skill", "rule", "log", "episode", "evaluation", "skill_stats"],
                        "description": "Filter by artifact type"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (any match)"
                    },
                    "sensitivity": {
                        "type": "string",
                        "enum": ["public", "internal", "sensitive"],
                        "description": "Filter by sensitivity level"
                    },
                    "workflow": {
                        "type": "string",
                        "description": "Filter by source workflow"
                    },
                    "search_text": {
                        "type": "string",
                        "description": "Search in titles/content"
                    },
                    "since": {
                        "type": "string",
                        "description": "ISO date to filter from (e.g., 2026-02-01)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="duro_semantic_search",
            description="Semantic search across artifacts using hybrid vector + keyword matching. Falls back gracefully to keyword-only if embeddings unavailable. Returns ranked results with optional score breakdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "artifact_type": {
                        "type": "string",
                        "enum": ["fact", "decision", "episode", "evaluation", "skill_stats", "log"],
                        "description": "Filter by artifact type"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (any match)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 20
                    },
                    "explain": {
                        "type": "boolean",
                        "description": "Include score breakdown for debugging/tuning",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="duro_get_artifact",
            description="Retrieve full artifact by ID. Returns complete JSON envelope.",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "The artifact ID to retrieve"
                    }
                },
                "required": ["artifact_id"]
            }
        ),
        Tool(
            name="duro_list_artifacts",
            description="List recent artifacts, optionally filtered by type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_type": {
                        "type": "string",
                        "enum": ["fact", "decision", "skill", "rule", "log", "episode", "evaluation", "skill_stats"],
                        "description": "Filter by type"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="duro_reindex",
            description="Rebuild SQLite index from artifact files. Use if index gets out of sync.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="duro_run_migration",
            description="Run database migrations to add new features (e.g., vector search tables). Safe to run multiple times - migrations are idempotent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "migration_id": {
                        "type": "string",
                        "description": "Specific migration to run (e.g., '001_add_vectors'). If omitted, runs all pending migrations.",
                        "default": None
                    },
                    "action": {
                        "type": "string",
                        "enum": ["up", "status"],
                        "description": "up = apply migration, status = check status without applying",
                        "default": "status"
                    }
                }
            }
        ),
        Tool(
            name="duro_delete_artifact",
            description="Delete an artifact with audit logging. Requires a reason. Refuses to delete sensitive artifacts unless force=True.",
            inputSchema={
                "type": "object",
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "The artifact ID to delete"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Required: Explanation for why this artifact is being deleted"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Override sensitivity protection for sensitive artifacts. Use with caution.",
                        "default": False
                    }
                },
                "required": ["artifact_id", "reason"]
            }
        ),
        Tool(
            name="duro_query_audit_log",
            description="Query the audit log (deletions). Returns entries with optional integrity chain verification.",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type (e.g., 'delete')",
                        "enum": ["delete"]
                    },
                    "artifact_id": {
                        "type": "string",
                        "description": "Filter by specific artifact ID"
                    },
                    "search_text": {
                        "type": "string",
                        "description": "Search in reason field"
                    },
                    "since": {
                        "type": "string",
                        "description": "ISO date to filter from (e.g., 2026-02-01)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to return",
                        "default": 50
                    },
                    "verify_chain": {
                        "type": "boolean",
                        "description": "Verify integrity chain hashes",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="duro_log_audit_repair",
            description="Log an audit chain repair event. Use after manually fixing integrity chain issues.",
            inputSchema={
                "type": "object",
                "properties": {
                    "backup_path": {
                        "type": "string",
                        "description": "Path to the backup file"
                    },
                    "backup_hash": {
                        "type": "string",
                        "description": "SHA256 of backup file content"
                    },
                    "repaired_hash": {
                        "type": "string",
                        "description": "SHA256 of repaired file content"
                    },
                    "entries_before": {
                        "type": "integer",
                        "description": "Number of entries in backup"
                    },
                    "entries_after": {
                        "type": "integer",
                        "description": "Number of entries after repair"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the repair was needed"
                    }
                },
                "required": ["backup_path", "backup_hash", "repaired_hash", "entries_before", "entries_after", "reason"]
            }
        ),
        Tool(
            name="duro_query_repair_log",
            description="Query the audit repair log (meta-audit of chain fixes).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to return",
                        "default": 20
                    }
                }
            }
        ),

        # Orchestration tools
        Tool(
            name="duro_orchestrate",
            description="Route a task through the workflow selector. Checks rules, selects skill/tool, executes, logs run.",
            inputSchema={
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "What you want to do (e.g., 'store fact', 'store decision', 'delete artifact')"
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments for the task (e.g., claim, confidence, source_urls)",
                        "default": {}
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, show what would happen without executing",
                        "default": False
                    },
                    "sensitivity": {
                        "type": "string",
                        "enum": ["public", "internal", "sensitive"],
                        "description": "Override auto-detected sensitivity"
                    }
                },
                "required": ["intent"]
            }
        ),
        Tool(
            name="duro_list_runs",
            description="List recent orchestration runs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max runs to return",
                        "default": 20
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["success", "failed", "denied", "dry_run"],
                        "description": "Filter by outcome"
                    }
                }
            }
        ),
        Tool(
            name="duro_get_run",
            description="Get full details of a specific run.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "The run ID to retrieve"
                    }
                },
                "required": ["run_id"]
            }
        ),

        # Episode tools (Phase 1: Feedback Loop)
        Tool(
            name="duro_create_episode",
            description="Create a new episode to track goal-level work. Episodes capture: goal -> plan -> actions -> result -> evaluation. Use for tasks >3min, that use tools, or produce artifacts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "What this episode is trying to achieve"
                    },
                    "plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Planned steps to achieve the goal"
                    },
                    "context": {
                        "type": "object",
                        "description": "Context including domain, constraints, environment",
                        "properties": {
                            "domain": {"type": "string"},
                            "constraints": {"type": "array", "items": {"type": "string"}},
                            "environment": {"type": "object"}
                        }
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Searchable tags"
                    }
                },
                "required": ["goal"]
            }
        ),
        Tool(
            name="duro_add_episode_action",
            description="Add an action to an open episode. Actions are refs (run_id, tool, summary) not full outputs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "episode_id": {
                        "type": "string",
                        "description": "The episode ID to update"
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Run ID of the action (if orchestrated)"
                    },
                    "tool": {
                        "type": "string",
                        "description": "Tool used in this action"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what the action did"
                    }
                },
                "required": ["episode_id", "summary"]
            }
        ),
        Tool(
            name="duro_close_episode",
            description="Close an episode with its result. This marks the episode as complete and calculates duration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "episode_id": {
                        "type": "string",
                        "description": "The episode ID to close"
                    },
                    "result": {
                        "type": "string",
                        "enum": ["success", "partial", "failed"],
                        "description": "Final outcome of the episode"
                    },
                    "result_summary": {
                        "type": "string",
                        "description": "Brief summary of what was achieved"
                    },
                    "links": {
                        "type": "object",
                        "description": "References to artifacts created during this episode",
                        "properties": {
                            "facts_created": {"type": "array", "items": {"type": "string"}},
                            "decisions_created": {"type": "array", "items": {"type": "string"}},
                            "skills_used": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "required": ["episode_id", "result"]
            }
        ),
        Tool(
            name="duro_evaluate_episode",
            description="Create an evaluation for a closed episode. Rubric: outcome_quality, cost, correctness_risk, reusability, reproducibility (all 0-5 scale).",
            inputSchema={
                "type": "object",
                "properties": {
                    "episode_id": {
                        "type": "string",
                        "description": "The episode ID to evaluate"
                    },
                    "rubric": {
                        "type": "object",
                        "description": "Evaluation scores (0-5 scale)",
                        "properties": {
                            "outcome_quality": {"type": "object", "properties": {"score": {"type": "integer"}, "notes": {"type": "string"}}},
                            "cost": {"type": "object", "properties": {"duration_mins": {"type": "number"}, "tools_used": {"type": "integer"}, "tokens_bucket": {"type": "string", "enum": ["XS", "S", "M", "L", "XL"]}}},
                            "correctness_risk": {"type": "object", "properties": {"score": {"type": "integer"}, "notes": {"type": "string"}}},
                            "reusability": {"type": "object", "properties": {"score": {"type": "integer"}, "notes": {"type": "string"}}},
                            "reproducibility": {"type": "object", "properties": {"score": {"type": "integer"}, "notes": {"type": "string"}}}
                        }
                    },
                    "grade": {
                        "type": "string",
                        "description": "Overall grade (A+, A, B+, B, C, D, F)"
                    },
                    "memory_updates": {
                        "type": "object",
                        "description": "Artifacts to reinforce or decay",
                        "properties": {
                            "reinforce": {"type": "array", "items": {"type": "object"}},
                            "decay": {"type": "array", "items": {"type": "object"}}
                        }
                    },
                    "next_change": {
                        "type": "string",
                        "description": "What to do differently next time"
                    }
                },
                "required": ["episode_id", "rubric", "grade"]
            }
        ),
        Tool(
            name="duro_apply_evaluation",
            description="Apply memory updates from an evaluation. Reinforces/decays confidence on facts and skill stats. Deltas capped at +/-0.02, confidence range 0.05-0.99.",
            inputSchema={
                "type": "object",
                "properties": {
                    "evaluation_id": {
                        "type": "string",
                        "description": "The evaluation ID to apply"
                    }
                },
                "required": ["evaluation_id"]
            }
        ),
        Tool(
            name="duro_get_episode",
            description="Get full details of an episode including actions and links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "episode_id": {
                        "type": "string",
                        "description": "The episode ID to retrieve"
                    }
                },
                "required": ["episode_id"]
            }
        ),
        Tool(
            name="duro_list_episodes",
            description="List recent episodes, optionally filtered by status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "closed"],
                        "description": "Filter by episode status"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max episodes to return",
                        "default": 20
                    }
                }
            }
        ),
        Tool(
            name="duro_suggest_episode",
            description="Check if current work should become an episode. Returns suggestion based on: tools_used, duration >3min, or artifact production. Use this for auto-detection without full automation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tools_used": {
                        "type": "boolean",
                        "description": "Whether tools were used in this work"
                    },
                    "duration_mins": {
                        "type": "number",
                        "description": "Duration of work so far in minutes"
                    },
                    "artifacts_produced": {
                        "type": "boolean",
                        "description": "Whether any artifacts (facts, decisions, code) were produced"
                    },
                    "goal_summary": {
                        "type": "string",
                        "description": "Brief summary of what was being worked on"
                    }
                },
                "required": ["tools_used"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    try:
        # Memory tools
        if name == "duro_load_context":
            include_soul = arguments.get("include_soul", True)
            recent_days = arguments.get("recent_days", 3)

            # Auto-compress old logs before loading (keeps context lean)
            compress_results = memory.compress_old_logs()

            result = []

            # Soul (truncated if very long)
            if include_soul:
                soul = memory.load_soul()
                if soul:
                    if len(soul) > 2000:
                        result.append(f"## Soul Configuration\n{soul[:2000]}...")
                    else:
                        result.append(f"## Soul Configuration\n{soul}")

            # Core memory
            core = memory.load_core_memory()
            if core:
                result.append(f"## Core Memory\n{core}")

            # Today's memory (full raw log)
            today = memory.load_today_memory()
            if today:
                result.append(f"## Today's Memory\n{today}")

            # Recent memory (hybrid: summaries for old days)
            recent = memory.load_recent_memory(days=recent_days, use_summaries=True)
            today_date = datetime.now().strftime("%Y-%m-%d")

            # Filter out today (already shown above)
            older_days = {k: v for k, v in recent.items() if k != today_date}
            if older_days:
                result.append("## Recent Memory (Summaries)")
                for date, content in list(older_days.items()):
                    result.append(f"### {date}\n{content}")

            # Add stats footer
            stats = memory.get_memory_stats()
            if compress_results:
                compressed_dates = [d for d, s in compress_results.items() if "compressed" in s]
                if compressed_dates:
                    result.append(f"\n*Auto-compressed {len(compressed_dates)} old log(s). Use `duro_query_archive` to access full details.*")

            text = "\n\n".join(result) if result else "No context loaded."
            return [TextContent(type="text", text=text)]

        elif name == "duro_save_memory":
            content = arguments["content"]
            section = arguments.get("section", "Session Log")
            success = memory.save_to_today(content, section)
            text = f"Memory saved to today's log under '{section}'." if success else "Failed to save memory."
            return [TextContent(type="text", text=text)]

        elif name == "duro_save_learning":
            learning = arguments["learning"]
            category = arguments.get("category", "General")
            success = memory.save_learning(learning, category)
            text = f"Learning saved: {learning[:100]}..." if success else "Failed to save learning."
            return [TextContent(type="text", text=text)]

        elif name == "duro_log_task":
            task = arguments["task"]
            outcome = arguments["outcome"]
            success = memory.save_task_completed(task, outcome)
            text = "Task logged successfully." if success else "Failed to log task."
            return [TextContent(type="text", text=text)]

        elif name == "duro_log_failure":
            task = arguments["task"]
            error = arguments["error"]
            lesson = arguments["lesson"]
            success = memory.save_failure(task, error, lesson)
            text = "Failure logged with lesson." if success else "Failed to log failure."
            return [TextContent(type="text", text=text)]

        elif name == "duro_compress_logs":
            results = memory.compress_old_logs()
            if results:
                text = "## Memory Compression Results\n\n"
                for date, status in results.items():
                    text += f"- **{date}**: {status}\n"
                stats = memory.get_memory_stats()
                text += f"\n**Current stats:** {stats['active_logs']} active, {stats['summaries']} summaries, {stats['archived_logs']} archived"
            else:
                text = "No logs to compress. Only today's log is active."
            return [TextContent(type="text", text=text)]

        elif name == "duro_query_archive":
            date = arguments.get("date")
            search = arguments.get("search")
            limit = arguments.get("limit", 5)

            if date:
                # Retrieve specific archived log
                content = memory.load_archived_log(date)
                if content:
                    text = f"## Archived Log: {date}\n\n{content}"
                else:
                    text = f"No archived log found for {date}"
            elif search:
                # Search through archives
                results = memory.search_archives(search, limit)
                if results:
                    text = f"## Search Results: '{search}'\n\n"
                    for r in results:
                        text += f"### {r['date']}\n"
                        for match in r['matches']:
                            text += f"- Line {match['line_num']}: {match['text']}\n"
                        text += "\n"
                else:
                    text = f"No matches found for '{search}' in archives"
            else:
                text = "Please provide either 'date' or 'search' parameter"
            return [TextContent(type="text", text=text)]

        elif name == "duro_list_archives":
            archives = memory.list_available_archives()
            if archives:
                text = "## Available Archives\n\n"
                total_size = 0
                for a in archives:
                    text += f"- **{a['date']}**: {a['size_kb']} KB\n"
                    total_size += a['size_bytes']
                text += f"\n**Total:** {len(archives)} archives, {round(total_size/1024, 1)} KB"
            else:
                text = "No archives available. Run `duro_compress_logs` to archive old memory logs."
            return [TextContent(type="text", text=text)]

        # Skills tools
        elif name == "duro_list_skills":
            summary = skills.get_skills_summary()
            result = f"## Duro Skills ({summary['total_skills']} total)\n\n"
            result += f"**By tier:** Core: {summary['by_tier']['core']}, Tested: {summary['by_tier']['tested']}, Untested: {summary['by_tier']['untested']}\n\n"
            for s in summary["skills"]:
                result += f"- **{s['name']}** [{s['tier']}]: {s['description']}\n"
            return [TextContent(type="text", text=result)]

        elif name == "duro_find_skills":
            keywords = arguments["keywords"]
            matches = skills.find_skills(keywords)
            if matches:
                result = f"## Skills matching {keywords}\n\n"
                for s in matches:
                    result += f"- **{s['name']}** [{s.get('tier')}]: {s.get('description', '')}\n"
            else:
                result = f"No skills found matching: {keywords}"
            return [TextContent(type="text", text=result)]

        elif name == "duro_run_skill":
            skill_name = arguments["skill_name"]
            args = arguments.get("args", {})
            success, output = skills.run_skill(skill_name, args)
            text = f"**Skill execution {'succeeded' if success else 'failed'}**\n\n{output}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_get_skill_code":
            skill_name = arguments["skill_name"]
            code = skills.get_skill_code(skill_name)
            if code:
                text = f"```python\n{code}\n```"
            else:
                text = f"Skill '{skill_name}' not found."
            return [TextContent(type="text", text=text)]

        # Rules tools
        elif name == "duro_check_rules":
            task_desc = arguments["task_description"]
            applicable = rules.check_rules(task_desc)
            formatted = rules.format_rules_for_context(applicable)

            # Mark rules as used
            for r in applicable:
                rules.apply_rule(r["rule"]["id"])

            return [TextContent(type="text", text=formatted)]

        elif name == "duro_list_rules":
            summary = rules.get_rules_summary()
            result = f"## Duro Rules ({summary['total_rules']} total)\n\n"
            result += f"**Hard rules:** {summary['hard_rules']} | **Soft rules:** {summary['soft_rules']}\n\n"
            for r in summary["rules"]:
                result += f"- **{r['name']}** [{r['type']}]: triggers on {r['keywords']}\n"
            return [TextContent(type="text", text=result)]

        # Project tools
        elif name == "duro_get_project":
            project_name = arguments["project_name"]
            projects_dir = Path(CONFIG["paths"]["projects_dir"])
            project_dir = projects_dir / project_name

            if not project_dir.exists():
                return [TextContent(type="text", text=f"Project '{project_name}' not found.")]

            # Load project files
            result = f"## Project: {project_name}\n\n"

            # Look for common project files
            for filename in ["README.md", "PRODUCTION_WORKFLOW.md", "CHARACTER_BIBLE.md", "setup.py"]:
                file_path = project_dir / filename
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8")[:1000]
                    result += f"### {filename}\n{content}...\n\n"

            return [TextContent(type="text", text=result)]

        elif name == "duro_list_projects":
            projects_dir = Path(CONFIG["paths"]["projects_dir"])
            if projects_dir.exists():
                projects = [d.name for d in projects_dir.iterdir() if d.is_dir()]
                result = f"## Projects ({len(projects)})\n\n" + "\n".join(f"- {p}" for p in projects)
            else:
                result = "No projects directory found."
            return [TextContent(type="text", text=result)]

        # System tools
        elif name == "duro_status":
            mem_stats = memory.get_memory_stats()
            skill_summary = skills.get_skills_summary()
            rule_summary = rules.get_rules_summary()
            artifact_stats = artifact_store.get_stats()

            result = f"""## Duro System Status

**Memory**
- Memory files: {mem_stats['total_memory_files']}
- Core memory: {'Yes' if mem_stats['core_memory_exists'] else 'No'}
- Today's log: {'Yes' if mem_stats['today_file_exists'] else 'No'}

**Artifacts (Structured Memory)**
- Total artifacts: {artifact_stats['total_artifacts']}
- By type: {artifact_stats['by_type']}
- By sensitivity: {artifact_stats['by_sensitivity']}

**Skills**
- Total skills: {skill_summary['total_skills']}
- Core: {skill_summary['by_tier']['core']} | Tested: {skill_summary['by_tier']['tested']} | Untested: {skill_summary['by_tier']['untested']}

**Rules**
- Total rules: {rule_summary['total_rules']}
- Hard: {rule_summary['hard_rules']} | Soft: {rule_summary['soft_rules']}

**Status:** Operational
**Timestamp:** {datetime.now().isoformat()}
"""
            return [TextContent(type="text", text=result)]

        elif name == "duro_health_check":
            verbose = arguments.get("verbose", False)
            health = _startup_health_check()

            # Build output
            status_icons = {"ok": "✅", "warning": "⚠️", "error": "❌"}
            overall_icon = status_icons.get(health["overall"], "❓")

            lines = [f"## Duro Health Check {overall_icon}\n"]
            lines.append(f"**Timestamp:** {health['timestamp']}")
            lines.append(f"**Overall Status:** {health['overall'].upper()}\n")

            lines.append("### Checks\n")
            for check_name, check_data in health["checks"].items():
                icon = status_icons.get(check_data.get("status"), "❓")
                status = check_data.get("status", "unknown")
                lines.append(f"- **{check_name}**: {icon} {status}")

                if verbose:
                    # Show detailed info for each check
                    for key, value in check_data.items():
                        if key != "status":
                            lines.append(f"  - {key}: {value}")

            if health["issues"]:
                lines.append("\n### Issues Found\n")
                for issue in health["issues"]:
                    lines.append(f"- {issue}")

            # Verbose mode: add artifact types breakdown for drift debugging
            if verbose:
                lines.append("\n### Artifact Types Breakdown\n")
                from schemas import TYPE_DIRECTORIES
                for type_name, dir_name in TYPE_DIRECTORIES.items():
                    type_dir = MEMORY_DIR / dir_name
                    if type_dir.exists():
                        file_count = len(list(type_dir.glob("*.json")))
                        indexed_count = artifact_store.index.count(type_name)
                        drift = abs(file_count - indexed_count)
                        drift_marker = " ⚠️" if drift > 0 else ""
                        lines.append(f"- **{type_name}**: {file_count} files, {indexed_count} indexed{drift_marker}")
                    else:
                        lines.append(f"- **{type_name}**: (dir not created)")

                # Add search capabilities
                lines.append("\n### Search Capabilities\n")
                search_caps = artifact_store.index.get_search_capabilities()
                lines.append(f"- **Mode:** {search_caps['mode']}")
                lines.append(f"- **FTS5:** {'✅' if search_caps['fts_available'] else '❌'}")
                lines.append(f"- **Vector Search:** {'✅' if search_caps['vector_available'] else '❌'}")
                lines.append(f"- **Embeddings:** {search_caps['embedding_count']}")

                # Add embedding model status
                from embeddings import get_embedding_status
                emb_status = get_embedding_status()
                lines.append(f"- **Embedding Model:** {emb_status['model_name'] or 'Not available'}")
                lines.append(f"- **Model Loaded:** {'✅' if emb_status['model_loaded'] else '❌'}")

            if not health["issues"] and not verbose:
                lines.append("\n*All systems operational. Use verbose=true for details.*")

            return [TextContent(type="text", text="\n".join(lines))]

        # Temporal tools (Phase 2)
        elif name == "duro_supersede_fact":
            old_fact_id = arguments["old_fact_id"]
            new_fact_id = arguments["new_fact_id"]
            reason = arguments.get("reason")

            success, msg = artifact_store.supersede_fact(old_fact_id, new_fact_id, reason)

            if success:
                text = f"✅ {msg}"
            else:
                text = f"❌ {msg}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_get_related":
            artifact_id = arguments["artifact_id"]
            relation_type = arguments.get("relation_type")
            direction = arguments.get("direction", "both")

            relations = artifact_store.index.get_relations(
                artifact_id,
                direction=direction,
                relation_type=relation_type
            )

            if not relations:
                text = f"No relations found for '{artifact_id}'."
            else:
                lines = [f"## Relations for {artifact_id}\n"]
                for rel in relations:
                    dir_icon = "→" if rel["direction"] == "outgoing" else "←"
                    other_id = rel["target_id"] if rel["direction"] == "outgoing" else rel["source_id"]
                    lines.append(f"- {dir_icon} **{rel['relation']}** {other_id}")
                    if rel.get("metadata"):
                        lines.append(f"  - {rel['metadata']}")
                text = "\n".join(lines)
            return [TextContent(type="text", text=text)]

        # Auto-capture & Proactive recall tools (Phase 3)
        elif name == "duro_proactive_recall":
            from proactive import ProactiveRecall

            context = arguments["context"]
            limit = arguments.get("limit", 10)
            include_types = arguments.get("include_types")
            force = arguments.get("force", False)

            # Create ProactiveRecall instance
            recall = ProactiveRecall(artifact_store, artifact_store.index)
            result = recall.recall(
                context=context,
                limit=limit,
                include_types=include_types,
                force=force
            )

            if not result.triggered:
                text = f"**Proactive Recall:** No relevant memories found.\n\nReason: {result.reason}"
            else:
                lines = [f"## Proactive Recall Results\n"]
                lines.append(f"**Categories matched:** {', '.join(result.categories_matched) or 'none'}")
                lines.append(f"**Search mode:** {result.search_mode}")
                lines.append(f"**Recall time:** {result.recall_time_ms}ms\n")

                if result.memories:
                    lines.append("### Relevant Memories\n")
                    for i, mem in enumerate(result.memories, 1):
                        score = round(mem.get("relevance_score", 0), 3)
                        lines.append(f"**{i}. [{mem['type']}]** {mem['summary'][:200]}")
                        lines.append(f"   - ID: `{mem['id']}`")
                        lines.append(f"   - Score: {score} | Tags: {', '.join(mem.get('tags', []))}\n")
                else:
                    lines.append("*No memories met the relevance threshold.*")

                text = "\n".join(lines)
            return [TextContent(type="text", text=text)]

        elif name == "duro_extract_learnings":
            from proactive import extract_learnings_from_text

            text_input = arguments["text"]
            auto_save = arguments.get("auto_save", False)

            result = extract_learnings_from_text(
                text=text_input,
                artifact_store=artifact_store if auto_save else None,
                auto_save=auto_save
            )

            lines = ["## Extracted Learnings\n"]
            lines.append(f"**Total items found:** {result['count']}")
            if auto_save:
                lines.append(f"**Auto-saved:** {len(result['saved_ids'])} artifacts\n")
            else:
                lines.append("*(Use auto_save=true to persist these)*\n")

            if result["learnings"]:
                lines.append("### Learnings\n")
                for i, learning in enumerate(result["learnings"], 1):
                    lines.append(f"{i}. {learning}")
                lines.append("")

            if result["facts"]:
                lines.append("### Facts\n")
                for fact in result["facts"]:
                    conf = fact.get("confidence", 0.5)
                    lines.append(f"- **{fact['claim'][:150]}** (confidence: {conf})")
                lines.append("")

            if result["decisions"]:
                lines.append("### Decisions\n")
                for dec in result["decisions"]:
                    lines.append(f"- **{dec['decision'][:100]}**")
                    lines.append(f"  - Rationale: {dec.get('rationale', '')[:100]}")
                lines.append("")

            if result['count'] == 0:
                lines.append("*No learnings, facts, or decisions detected in the text.*")

            text = "\n".join(lines)
            return [TextContent(type="text", text=text)]

        # Decay & Maintenance tools (Phase 4)
        elif name == "duro_apply_decay":
            from decay import apply_batch_decay, DecayConfig, DEFAULT_DECAY_CONFIG

            dry_run = arguments.get("dry_run", True)
            min_importance = arguments.get("min_importance", 0)
            include_stale_report = arguments.get("include_stale_report", True)

            # Load all facts
            facts = artifact_store.query(artifact_type="fact", limit=10000)
            full_facts = [artifact_store.get_artifact(f["id"]) for f in facts]
            full_facts = [f for f in full_facts if f is not None]

            # Filter by importance if specified
            if min_importance > 0:
                full_facts = [f for f in full_facts if f.get("data", {}).get("importance", 0.5) >= min_importance]

            # Apply decay
            result = apply_batch_decay(full_facts, DEFAULT_DECAY_CONFIG, dry_run=dry_run)

            # Save changes if not dry run
            if not dry_run:
                for fact in full_facts:
                    artifact_store._save_artifact(fact)

            lines = ["## Decay Results\n"]
            lines.append(f"**Mode:** {'DRY RUN' if dry_run else 'APPLIED'}")
            lines.append(f"**Total facts:** {result.total_facts}")
            lines.append(f"**Decayed:** {result.decayed_count}")
            lines.append(f"**Skipped (pinned):** {result.skipped_pinned}")
            lines.append(f"**Skipped (grace period):** {result.skipped_grace_period}")
            lines.append(f"**Skipped (reinforcement):** {result.skipped_reinforcement}")
            lines.append(f"**Now stale:** {result.stale_count}\n")

            if include_stale_report and result.stale_count > 0:
                lines.append("### Stale Facts (top 10)\n")
                stale = [r for r in result.results if r.get("stale")][:10]
                for s in stale:
                    lines.append(f"- `{s['id']}` conf: {s['new_confidence']:.3f} (was {s['old_confidence']:.3f})")

            text = "\n".join(lines)
            return [TextContent(type="text", text=text)]

        elif name == "duro_reembed":
            from embedding_worker import queue_for_embedding

            artifact_ids = arguments.get("artifact_ids")
            artifact_type = arguments.get("artifact_type")
            all_artifacts = arguments.get("all", False)

            queued = []

            if artifact_ids:
                # Specific IDs
                for aid in artifact_ids:
                    queue_for_embedding(aid)
                    queued.append(aid)
            elif artifact_type:
                # All of a type
                results = artifact_store.query(artifact_type=artifact_type, limit=10000)
                for r in results:
                    queue_for_embedding(r["id"])
                    queued.append(r["id"])
            elif all_artifacts:
                # Everything
                results = artifact_store.query(limit=10000)
                for r in results:
                    queue_for_embedding(r["id"])
                    queued.append(r["id"])
            else:
                return [TextContent(type="text", text="No artifacts specified. Use artifact_ids, artifact_type, or all=true.")]

            text = f"## Re-embed Queued\n\n**Queued:** {len(queued)} artifacts for re-embedding.\n\nThe embedding worker will process these in the background."
            return [TextContent(type="text", text=text)]

        elif name == "duro_maintenance_report":
            from decay import generate_maintenance_report, DEFAULT_DECAY_CONFIG

            include_stale_list = arguments.get("include_stale_list", True)
            top_n_stale = arguments.get("top_n_stale", 10)

            # Load all facts
            facts = artifact_store.query(artifact_type="fact", limit=10000)
            full_facts = [artifact_store.get_artifact(f["id"]) for f in facts]
            full_facts = [f for f in full_facts if f is not None]

            # Generate report
            report = generate_maintenance_report(full_facts, DEFAULT_DECAY_CONFIG, top_n_stale)

            # Get embedding/FTS coverage
            fts_stats = artifact_store.index.get_fts_completeness()
            emb_stats = artifact_store.index.get_embedding_stats()

            lines = ["## Maintenance Report\n"]
            lines.append("### Fact Health\n")
            lines.append(f"- **Total facts:** {report.total_facts}")
            lines.append(f"- **Pinned:** {report.pinned_count} ({report.pinned_pct}%)")
            lines.append(f"- **Stale:** {report.stale_count} ({report.stale_pct}%)")
            lines.append(f"- **Avg confidence:** {report.avg_confidence}")
            lines.append(f"- **Avg importance:** {report.avg_importance}")
            lines.append(f"- **Avg reinforcement count:** {report.avg_reinforcement_count}")
            lines.append(f"- **Oldest unreinforced:** {report.oldest_unreinforced_days} days\n")

            lines.append("### Index Coverage\n")
            fts_coverage = fts_stats.get("coverage_pct", 0)
            emb_coverage = emb_stats.get("coverage_pct", 0)
            lines.append(f"- **FTS coverage:** {fts_coverage}%")
            lines.append(f"- **Embedding coverage:** {emb_coverage}%\n")

            if include_stale_list and report.top_stale_high_importance:
                lines.append("### Top Stale High-Importance Facts\n")
                for fact in report.top_stale_high_importance:
                    lines.append(f"- `{fact['id']}` imp={fact['importance']}, conf={fact['confidence']:.3f}")
                    lines.append(f"  - {fact['claim'][:80]}...")
                    lines.append(f"  - Inactive: {fact['days_inactive']} days")

            text = "\n".join(lines)
            return [TextContent(type="text", text=text)]

        elif name == "duro_reinforce_fact":
            from decay import reinforce_fact

            fact_id = arguments["fact_id"]
            fact = artifact_store.get_artifact(fact_id)

            if not fact:
                return [TextContent(type="text", text=f"Fact not found: {fact_id}")]

            if fact.get("type") != "fact":
                return [TextContent(type="text", text=f"Artifact {fact_id} is not a fact (type: {fact.get('type')})")]

            # Reinforce
            updated_fact = reinforce_fact(fact)
            artifact_store._save_artifact(updated_fact)

            data = updated_fact.get("data", {})
            text = f"## Fact Reinforced\n\n- **ID:** `{fact_id}`\n- **Reinforcement count:** {data.get('reinforcement_count', 0)}\n- **Last reinforced:** {data.get('last_reinforced_at')}"
            return [TextContent(type="text", text=text)]

        # Artifact tools
        elif name == "duro_store_fact":
            success, artifact_id, path = artifact_store.store_fact(
                claim=arguments["claim"],
                source_urls=arguments.get("source_urls"),
                snippet=arguments.get("snippet"),
                confidence=arguments.get("confidence", 0.5),
                tags=arguments.get("tags"),
                workflow=arguments.get("workflow", "manual"),
                sensitivity=arguments.get("sensitivity", "public"),
                evidence_type=arguments.get("evidence_type", "none"),
                provenance=arguments.get("provenance", "unknown")
            )
            if success:
                text = f"Fact stored successfully.\n- ID: {artifact_id}\n- Path: {path}"
            else:
                text = f"Failed to store fact: {path}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_store_decision":
            success, artifact_id, path = artifact_store.store_decision(
                decision=arguments["decision"],
                rationale=arguments["rationale"],
                alternatives=arguments.get("alternatives"),
                context=arguments.get("context"),
                reversible=arguments.get("reversible", True),
                tags=arguments.get("tags"),
                workflow=arguments.get("workflow", "manual"),
                sensitivity=arguments.get("sensitivity", "internal")
            )
            if success:
                text = f"Decision stored successfully.\n- ID: {artifact_id}\n- Path: {path}"
            else:
                text = f"Failed to store decision: {path}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_validate_decision":
            success, message = artifact_store.validate_decision(
                decision_id=arguments["decision_id"],
                status=arguments["status"],
                episode_id=arguments.get("episode_id"),
                result=arguments.get("result"),
                notes=arguments.get("notes")
            )
            if success:
                # Get updated decision to show new confidence
                decision = artifact_store.get_artifact(arguments["decision_id"])
                confidence = decision["data"]["outcome"]["confidence"] if decision else "?"
                status_icon = {"validated": "✓", "reversed": "✗", "superseded": "→"}.get(arguments["status"], "?")
                text = f"Decision validated.\n- ID: `{arguments['decision_id']}`\n- Status: {status_icon} {arguments['status']}\n- Confidence: {confidence}"
                if arguments.get("episode_id"):
                    text += f"\n- Evidence: episode `{arguments['episode_id']}`"
            else:
                text = f"Failed to validate decision: {message}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_link_decision":
            success, message = artifact_store.link_decision_to_episode(
                decision_id=arguments["decision_id"],
                episode_id=arguments["episode_id"]
            )
            if success:
                text = f"Decision linked to episode.\n- Decision: `{arguments['decision_id']}`\n- Episode: `{arguments['episode_id']}`"
            else:
                text = f"Failed to link decision: {message}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_query_memory":
            results = artifact_store.query(
                artifact_type=arguments.get("artifact_type"),
                tags=arguments.get("tags"),
                sensitivity=arguments.get("sensitivity"),
                workflow=arguments.get("workflow"),
                search_text=arguments.get("search_text"),
                since=arguments.get("since"),
                limit=arguments.get("limit", 50)
            )
            if results:
                text = f"## Query Results ({len(results)} found)\n\n"
                for r in results:
                    text += f"- **{r['id']}** [{r['type']}]: {r['title']}\n"
                    text += f"  Tags: {r['tags']} | Created: {r['created_at'][:10]}\n"
            else:
                text = "No artifacts found matching query."
            return [TextContent(type="text", text=text)]

        elif name == "duro_semantic_search":
            query = arguments["query"]
            artifact_type = arguments.get("artifact_type")
            tags = arguments.get("tags")
            limit = arguments.get("limit", 20)
            explain = arguments.get("explain", False)

            # Get query embedding if available
            from embeddings import embed_text, is_embedding_available
            query_embedding = None
            if is_embedding_available():
                query_embedding = embed_text(query)

            # Run hybrid search
            search_result = artifact_store.index.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                artifact_type=artifact_type,
                tags=tags,
                limit=limit,
                explain=explain
            )

            results = search_result["results"]
            mode = search_result["mode"]

            text = f"## Semantic Search Results\n\n"
            text += f"**Mode:** {mode} | **Query:** \"{query}\"\n"
            text += f"**Candidates:** {search_result['total_candidates']} (FTS: {search_result['fts_count']}, Vec: {search_result['vector_count']})\n\n"

            if results:
                for r in results:
                    score = r["search_score"]
                    text += f"- **{r['id']}** [{r['type']}] (score: {score:.3f})\n"
                    text += f"  {r['title']}\n"
                    if explain and "score_components" in r:
                        sc = r["score_components"]
                        text += f"  _Components: rrf={sc['rrf_base']:.3f}, type={sc['type_weight']:.2f}, recency={sc['recency_boost']:.3f}_\n"
                        if r.get("explain"):
                            text += f"  _Explain: {r['explain']}_\n"
                    text += "\n"
            else:
                text += "No results found.\n"

            return [TextContent(type="text", text=text)]

        elif name == "duro_get_artifact":
            artifact_id = arguments["artifact_id"]
            artifact = artifact_store.get_artifact(artifact_id)
            if artifact:
                text = f"## Artifact: {artifact_id}\n\n```json\n{json.dumps(artifact, indent=2)}\n```"
            else:
                text = f"Artifact '{artifact_id}' not found."
            return [TextContent(type="text", text=text)]

        elif name == "duro_list_artifacts":
            results = artifact_store.list_artifacts(
                artifact_type=arguments.get("artifact_type"),
                limit=arguments.get("limit", 50)
            )
            if results:
                text = f"## Artifacts ({len(results)} listed)\n\n"
                for r in results:
                    text += f"- **{r['id']}** [{r['type']}]: {r['title']}\n"
            else:
                text = "No artifacts found."
            return [TextContent(type="text", text=text)]

        elif name == "duro_reindex":
            success_count, error_count = artifact_store.reindex()
            text = f"Reindex complete.\n- Indexed: {success_count}\n- Errors: {error_count}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_run_migration":
            action = arguments.get("action", "status")
            migration_id = arguments.get("migration_id")  # Optional: specific migration

            from migrations.runner import get_status, run_all_pending, run_migration

            migrations_dir = Path(__file__).parent / "migrations"
            db_path = str(MEMORY_DIR / "artifacts.db")

            if action == "status":
                status = get_status(migrations_dir, db_path)
                text = "## Migration Status\n\n"

                if status["applied"]:
                    text += "### Applied\n"
                    for m in status["applied"]:
                        text += f"- ✅ **{m['migration_id']}** ({m['applied_at'][:10]})\n"
                else:
                    text += "No migrations applied yet.\n"

                if status["pending"]:
                    text += "\n### Pending\n"
                    for m in status["pending"]:
                        text += f"- ⏳ **{m['migration_id']}**\n"

                if status["modified"]:
                    text += "\n### ⚠️ Modified Since Applied\n"
                    for m in status["modified"]:
                        text += f"- **{m['migration_id']}** (checksum changed)\n"

            elif action == "up":
                if migration_id:
                    # Run specific migration
                    migration_path = migrations_dir / f"m{migration_id}.py"
                    if not migration_path.exists():
                        migration_path = migrations_dir / f"{migration_id}.py"
                    if not migration_path.exists():
                        return [TextContent(type="text", text=f"Migration not found: {migration_id}")]

                    result = run_migration(db_path, migration_path)
                    text = f"## Migration: {result['migration_id']}\n\n"
                    text += f"- **Success:** {'✅' if result['success'] else '❌'}\n"
                    text += f"- **Message:** {result['message']}\n"
                    if result.get("details"):
                        text += f"- **Details:** {result['details']}\n"
                else:
                    # Run all pending
                    result = run_all_pending(migrations_dir, db_path)
                    text = "## Migration Run\n\n"
                    text += f"- **Success:** {'✅' if result['success'] else '❌'}\n"
                    if result["applied"]:
                        text += f"- **Applied:** {', '.join(result['applied'])}\n"
                    if result["skipped"]:
                        text += f"- **Skipped:** {', '.join(result['skipped'])}\n"
                    if result["failed"]:
                        text += f"- **Failed:** {', '.join(result['failed'])}\n"
            else:
                text = f"Unknown action: {action}. Use 'status' or 'up'."

            return [TextContent(type="text", text=text)]

        elif name == "duro_delete_artifact":
            artifact_id = arguments["artifact_id"]
            reason = arguments["reason"]
            force = arguments.get("force", False)

            success, message = artifact_store.delete_artifact(
                artifact_id=artifact_id,
                reason=reason,
                force=force
            )

            if success:
                text = f"Deleted successfully.\n- ID: {artifact_id}\n- {message}"
            else:
                text = f"Delete failed: {message}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_query_audit_log":
            result = artifact_store.query_audit_log(
                event_type=arguments.get("event_type"),
                artifact_id=arguments.get("artifact_id"),
                search_text=arguments.get("search_text"),
                since=arguments.get("since"),
                limit=arguments.get("limit", 50),
                verify_chain=arguments.get("verify_chain", False)
            )

            lines = [f"## Audit Log Query Results\n"]
            lines.append(f"**Total entries:** {result['total']}")

            if result.get("chain_valid") is not None:
                chain_status = "✅ Valid" if result["chain_valid"] else "❌ BROKEN - possible tampering"
                lines.append(f"**Integrity chain:** {chain_status}")

                # Show per-link details if available
                chain_details = result.get("chain_details", [])
                if chain_details:
                    lines.append("\n### Chain Verification\n")
                    lines.append("| # | timestamp | entry_hash | prev_hash | link |")
                    lines.append("|---|-----------|------------|-----------|------|")
                    for detail in chain_details:
                        entry_num = detail.get("entry", "?")
                        timestamp = detail.get("timestamp", "?")
                        entry_hash_len = detail.get("entry_hash_len", 0)
                        prev_hash_len = detail.get("prev_hash_len", 0)
                        link_ok = detail.get("link_ok")
                        status = detail.get("status", "unknown")

                        if link_ok is True:
                            link_icon = "✅"
                        elif link_ok is False:
                            link_icon = "❌"
                        elif status == "legacy":
                            link_icon = "⚪"
                        else:
                            link_icon = "⚠️"

                        lines.append(f"| {entry_num} | {timestamp} | {entry_hash_len} | {prev_hash_len} | {link_icon} |")

                    # Also show any broken links with details
                    broken = [d for d in chain_details if d.get("status") == "broken"]
                    if broken:
                        lines.append("\n**Broken links:**")
                        for d in broken:
                            lines.append(f"- Entry {d['entry']}: {d['message']}")

            if result.get("error"):
                lines.append(f"**Error:** {result['error']}")

            if result["entries"]:
                lines.append("\n### Entries\n")
                for entry in result["entries"]:
                    ts = entry.get("timestamp", "?")[:19]
                    aid = entry.get("artifact_id", "?")
                    reason = entry.get("reason", "")
                    force = " [FORCE]" if entry.get("force_used") else ""
                    lines.append(f"- `{ts}` | `{aid}` | {reason}{force}")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "duro_log_audit_repair":
            success = artifact_store.log_audit_repair(
                backup_path=arguments["backup_path"],
                backup_hash=arguments["backup_hash"],
                repaired_hash=arguments["repaired_hash"],
                entries_before=arguments["entries_before"],
                entries_after=arguments["entries_after"],
                reason=arguments["reason"]
            )
            if success:
                text = "Audit repair logged successfully."
            else:
                text = "Failed to log audit repair."
            return [TextContent(type="text", text=text)]

        elif name == "duro_query_repair_log":
            limit = arguments.get("limit", 20)
            logs_dir = Path(CONFIG["paths"]["memory_dir"]) / "logs"
            repairs_log = logs_dir / "audit_repairs.jsonl"

            if not repairs_log.exists():
                return [TextContent(type="text", text="No repair log found (no repairs recorded).")]

            entries = []
            try:
                with open(repairs_log, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
            except Exception as e:
                return [TextContent(type="text", text=f"Error reading repair log: {e}")]

            entries = entries[-limit:][::-1]  # Most recent first

            lines = [f"## Audit Repair Log ({len(entries)} entries)\n"]
            for entry in entries:
                ts = entry.get("timestamp", "?")[:19]
                reason = entry.get("reason", "?")
                before = entry.get("entries_before", "?")
                after = entry.get("entries_after", "?")
                lines.append(f"- `{ts}` | {before}→{after} entries | {reason}")

            return [TextContent(type="text", text="\n".join(lines))]

        # Orchestration tools
        elif name == "duro_orchestrate":
            intent = arguments["intent"]
            args = arguments.get("args", {})
            dry_run = arguments.get("dry_run", False)
            sensitivity = arguments.get("sensitivity")

            result = orchestrator.orchestrate(
                intent=intent,
                args=args,
                dry_run=dry_run,
                sensitivity=sensitivity
            )

            # Format output
            lines = ["## Orchestration Result\n"]
            lines.append(f"**Run ID:** `{result['run_id']}`")
            lines.append(f"**Intent:** {result['intent']}")
            lines.append(f"**Plan:** {result['plan']} ({result['plan_type']})")

            if result['rules_applied']:
                lines.append(f"**Rules Applied:** {', '.join(result['rules_applied'])}")

            if result['constraints']:
                lines.append(f"**Constraints:** {json.dumps(result['constraints'])}")

            outcome_icon = {
                "success": "✅",
                "failed": "❌",
                "denied": "🚫",
                "dry_run": "👁️"
            }.get(result['outcome'], "❓")

            lines.append(f"**Outcome:** {outcome_icon} {result['outcome']}")

            if result['error']:
                lines.append(f"**Error:** {result['error']}")

            if result['artifacts_created']:
                lines.append(f"**Artifacts Created:** {', '.join(result['artifacts_created'])}")

            lines.append(f"**Duration:** {result['duration_ms']}ms")
            lines.append(f"**Run Log:** `{result['run_path']}`")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "duro_list_runs":
            limit = arguments.get("limit", 20)
            outcome = arguments.get("outcome")

            runs = orchestrator.list_runs(limit=limit, outcome=outcome)

            if not runs:
                return [TextContent(type="text", text="No runs found.")]

            lines = [f"## Recent Runs ({len(runs)} shown)\n"]
            lines.append("| Run ID | Intent | Outcome | Duration |")
            lines.append("|--------|--------|---------|----------|")

            for run in runs:
                run_id = run.get("run_id", "?")[:25]
                intent = run.get("intent", "?")
                outcome = run.get("outcome", "?")
                duration = run.get("duration_ms", 0)

                outcome_icon = {"success": "✅", "failed": "❌", "denied": "🚫", "dry_run": "👁️"}.get(outcome, "❓")
                lines.append(f"| `{run_id}` | {intent} | {outcome_icon} | {duration}ms |")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "duro_get_run":
            run_id = arguments["run_id"]
            run = orchestrator.get_run(run_id)

            if not run:
                return [TextContent(type="text", text=f"Run '{run_id}' not found.")]

            text = f"## Run: {run_id}\n\n```json\n{json.dumps(run, indent=2)}\n```"
            return [TextContent(type="text", text=text)]

        # Episode tools
        elif name == "duro_create_episode":
            goal = arguments["goal"]
            plan = arguments.get("plan", [])
            context = arguments.get("context", {})
            tags = arguments.get("tags", [])

            success, episode_id, path = artifact_store.store_episode(
                goal=goal,
                plan=plan,
                context=context,
                tags=tags
            )

            if success:
                text = f"Episode created successfully.\n- ID: `{episode_id}`\n- Goal: {goal[:100]}...\n- Status: open"
            else:
                text = f"Failed to create episode: {path}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_add_episode_action":
            episode_id = arguments["episode_id"]
            action = {
                "run_id": arguments.get("run_id"),
                "tool": arguments.get("tool"),
                "summary": arguments["summary"]
            }

            success, message = artifact_store.update_episode(episode_id, {"action": action})

            if success:
                text = f"Action added to episode `{episode_id}`.\n- Tool: {action.get('tool', 'N/A')}\n- Summary: {action['summary'][:100]}"
            else:
                text = f"Failed to add action: {message}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_close_episode":
            episode_id = arguments["episode_id"]
            result = arguments["result"]
            result_summary = arguments.get("result_summary", "")
            links = arguments.get("links", {})

            updates = {
                "status": "closed",
                "result": result,
                "result_summary": result_summary
            }
            if links:
                updates["links"] = links

            success, message = artifact_store.update_episode(episode_id, updates)

            if success:
                # Get updated episode to show duration
                episode = artifact_store.get_artifact(episode_id)
                duration = episode["data"].get("duration_mins", "?")
                result_icon = {"success": "✅", "partial": "⚠️", "failed": "❌"}.get(result, "❓")
                text = f"Episode closed.\n- ID: `{episode_id}`\n- Result: {result_icon} {result}\n- Duration: {duration} mins"
            else:
                text = f"Failed to close episode: {message}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_evaluate_episode":
            episode_id = arguments["episode_id"]
            rubric = arguments["rubric"]
            grade = arguments["grade"]
            memory_updates = arguments.get("memory_updates", {"reinforce": [], "decay": []})
            next_change = arguments.get("next_change")

            success, eval_id, path = artifact_store.store_evaluation(
                episode_id=episode_id,
                rubric=rubric,
                grade=grade,
                memory_updates=memory_updates,
                next_change=next_change
            )

            if success:
                text = f"Evaluation created.\n- ID: `{eval_id}`\n- Episode: `{episode_id}`\n- Grade: {grade}\n- Memory updates pending: {len(memory_updates.get('reinforce', []))} reinforce, {len(memory_updates.get('decay', []))} decay"
            else:
                text = f"Failed to create evaluation: {path}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_apply_evaluation":
            evaluation_id = arguments["evaluation_id"]

            success, message, applied = artifact_store.apply_evaluation(evaluation_id)

            if success:
                lines = [f"Evaluation applied.\n- ID: `{evaluation_id}`"]
                if applied.get("reinforced"):
                    lines.append(f"\n**Reinforced ({len(applied['reinforced'])}):**")
                    for r in applied["reinforced"]:
                        lines.append(f"  - `{r['id']}` ({r['type']}): +{r['delta']:.3f} → {r['new_confidence']:.3f}")
                if applied.get("decayed"):
                    lines.append(f"\n**Decayed ({len(applied['decayed'])}):**")
                    for d in applied["decayed"]:
                        lines.append(f"  - `{d['id']}` ({d['type']}): {d['delta']:.3f} → {d['new_confidence']:.3f}")
                if applied.get("errors"):
                    lines.append(f"\n**Errors ({len(applied['errors'])}):**")
                    for e in applied["errors"]:
                        lines.append(f"  - `{e['id']}`: {e['error']}")
                text = "\n".join(lines)
            else:
                text = f"Failed to apply evaluation: {message}"
            return [TextContent(type="text", text=text)]

        elif name == "duro_get_episode":
            episode_id = arguments["episode_id"]
            episode = artifact_store.get_artifact(episode_id)

            if not episode:
                return [TextContent(type="text", text=f"Episode '{episode_id}' not found.")]

            text = f"## Episode: {episode_id}\n\n```json\n{json.dumps(episode, indent=2)}\n```"
            return [TextContent(type="text", text=text)]

        elif name == "duro_list_episodes":
            status = arguments.get("status")
            limit = arguments.get("limit", 20)

            # Query episodes via index
            results = artifact_store.query(artifact_type="episode", limit=limit)

            # Filter by status if specified (post-query since index doesn't have status field)
            if status:
                filtered = []
                for r in results:
                    ep = artifact_store.get_artifact(r["id"])
                    if ep and ep["data"].get("status") == status:
                        filtered.append(r)
                results = filtered[:limit]

            if not results:
                return [TextContent(type="text", text=f"No episodes found{' with status ' + status if status else ''}.")]

            lines = [f"## Episodes ({len(results)} found)\n"]
            lines.append("| ID | Goal | Status | Duration |")
            lines.append("|-----|------|--------|----------|")

            for r in results:
                ep = artifact_store.get_artifact(r["id"])
                if ep:
                    ep_id = ep["id"][:20]
                    goal = ep["data"].get("goal", "")[:40]
                    ep_status = ep["data"].get("status", "?")
                    duration = ep["data"].get("duration_mins", "-")
                    status_icon = {"open": "🔵", "closed": "✅"}.get(ep_status, "❓")
                    lines.append(f"| `{ep_id}` | {goal} | {status_icon} {ep_status} | {duration} |")

            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "duro_suggest_episode":
            tools_used = arguments.get("tools_used", False)
            duration_mins = arguments.get("duration_mins", 0)
            artifacts_produced = arguments.get("artifacts_produced", False)
            goal_summary = arguments.get("goal_summary", "")

            # Criteria for suggesting an episode
            should_create = False
            reasons = []

            if tools_used:
                should_create = True
                reasons.append("tools were used")
            if duration_mins >= 3:
                should_create = True
                reasons.append(f"duration is {duration_mins:.1f}min (>=3min)")
            if artifacts_produced:
                should_create = True
                reasons.append("artifacts were produced")

            if should_create:
                text = f"**Episode Suggested** ✨\n\nThis looks like an episode because: {', '.join(reasons)}.\n\n"
                text += f"Goal: {goal_summary[:100] if goal_summary else 'Not specified'}\n\n"
                text += "To create: `duro_create_episode(goal=\"{goal}\")`"
            else:
                text = "**No Episode Needed**\n\nThis work doesn't meet episode criteria (tools used, >3min, or artifact production)."

            return [TextContent(type="text", text=text)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


async def main():
    """Run the Duro MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
