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
import sys
from datetime import datetime
from pathlib import Path
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

_startup_ensure_consistency()

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

            context = memory.load_full_context()

            result = []
            if include_soul and context.get("soul"):
                result.append(f"## Soul Configuration\n{context['soul'][:2000]}...")

            if context.get("core_memory"):
                result.append(f"## Core Memory\n{context['core_memory']}")

            if context.get("today_memory"):
                result.append(f"## Today's Memory\n{context['today_memory']}")

            recent = context.get("recent_memories", {})
            if recent:
                result.append("## Recent Memory")
                for date, content in list(recent.items())[:recent_days]:
                    result.append(f"### {date}\n{content[:500]}...")

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
