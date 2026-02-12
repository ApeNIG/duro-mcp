"""
Artifact schemas for Duro memory system.
JSON Schema definitions for all artifact types.
"""

from typing import Any

# Base artifact envelope schema
ARTIFACT_ENVELOPE_SCHEMA = {
    "type": "object",
    "required": ["id", "type", "version", "created_at", "sensitivity", "tags", "source", "data"],
    "properties": {
        "id": {
            "type": "string",
            "pattern": "^[a-z]+_[0-9]{8}_[0-9]{6}_[a-z0-9]{6}$",
            "description": "Unique identifier: {type}_{YYYYMMDD}_{HHMMSS}_{random6}"
        },
        "type": {
            "type": "string",
            "enum": ["fact", "decision", "skill", "rule", "log"],
            "description": "Artifact type"
        },
        "version": {
            "type": "string",
            "pattern": "^[0-9]+\\.[0-9]+$",
            "description": "Schema version"
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp"
        },
        "updated_at": {
            "type": ["string", "null"],
            "format": "date-time",
            "description": "Last update timestamp or null"
        },
        "sensitivity": {
            "type": "string",
            "enum": ["public", "internal", "sensitive"],
            "description": "Data sensitivity level"
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Searchable tags"
        },
        "source": {
            "type": "object",
            "required": ["workflow"],
            "properties": {
                "workflow": {"type": "string"},
                "run_id": {"type": ["string", "null"]},
                "tool_trace_path": {"type": ["string", "null"]}
            }
        },
        "data": {
            "type": "object",
            "description": "Type-specific payload"
        }
    }
}

# Fact artifact data schema
FACT_DATA_SCHEMA = {
    "type": "object",
    "required": ["claim"],
    "properties": {
        "claim": {
            "type": "string",
            "description": "The factual claim being recorded"
        },
        "source_urls": {
            "type": "array",
            "items": {"type": "string", "format": "uri"},
            "description": "URLs supporting this fact"
        },
        "snippet": {
            "type": "string",
            "description": "Relevant excerpt or context"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score 0-1"
        },
        "verified": {
            "type": "boolean",
            "default": False,
            "description": "Whether fact has been verified"
        },
        "evidence_type": {
            "type": "string",
            "enum": ["quote", "paraphrase", "inference", "none"],
            "default": "none",
            "description": "How the evidence supports the claim: quote (direct), paraphrase (reworded), inference (derived), none (unverified)"
        },
        "provenance": {
            "type": "string",
            "enum": ["web", "local_file", "user", "tool_output", "unknown"],
            "default": "unknown",
            "description": "Where the fact came from: web (URL), local_file, user (stated), tool_output, unknown"
        }
    }
}

# Decision artifact data schema
DECISION_DATA_SCHEMA = {
    "type": "object",
    "required": ["decision", "rationale"],
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
            "default": True,
            "description": "Whether decision can be undone"
        },
        "outcome": {
            "type": "object",
            "description": "Decision outcome tracking",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["unverified", "validated", "reversed", "superseded"],
                    "description": "Current status of the decision"
                },
                "verified_at": {
                    "type": ["string", "null"],
                    "format": "date-time",
                    "description": "When the decision was validated/reversed"
                },
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "episode_id": {"type": "string"},
                            "result": {"type": "string", "enum": ["success", "partial", "failed"]},
                            "notes": {"type": "string"}
                        }
                    },
                    "description": "Episodes that tested this decision"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 0.99,
                    "description": "Confidence in this decision (can be reinforced/decayed)"
                }
            }
        },
        "episodes_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Episode IDs where this decision was applied"
        }
    }
}

# Skill artifact data schema
SKILL_DATA_SCHEMA = {
    "type": "object",
    "required": ["name", "description", "steps"],
    "properties": {
        "name": {
            "type": "string",
            "description": "Skill name/identifier"
        },
        "description": {
            "type": "string",
            "description": "What this skill does"
        },
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered steps to execute"
        },
        "tools_required": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tools needed for this skill"
        },
        "tier": {
            "type": "string",
            "enum": ["core", "tested", "untested"],
            "default": "untested"
        },
        "usage_count": {
            "type": "integer",
            "default": 0
        },
        "last_used": {
            "type": ["string", "null"],
            "format": "date-time"
        },
        # Operational metadata (reliability layer)
        "requires_network": {
            "type": "boolean",
            "default": False,
            "description": "Whether skill needs internet connectivity"
        },
        "timeout_seconds": {
            "type": "integer",
            "default": 300,
            "description": "Max execution time before timeout (default 5 min)"
        },
        "expected_runtime_seconds": {
            "type": ["integer", "null"],
            "description": "Typical execution time for planning purposes"
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Python packages or system dependencies required"
        },
        "side_effects": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["writes_file", "reads_file", "network_request", "modifies_state", "external_api", "creates_process"]
            },
            "description": "What side effects this skill produces"
        },
        "fallback_workflow": {
            "type": ["object", "null"],
            "description": "What to do if skill fails",
            "properties": {
                "on_timeout": {"type": "string", "description": "Action on timeout"},
                "on_network_error": {"type": "string", "description": "Action on network failure"},
                "on_dependency_missing": {"type": "string", "description": "Action if dependency not installed"},
                "alternative_skill": {"type": ["string", "null"], "description": "Fallback skill to try"}
            }
        },
        "pre_checks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Checks to run before execution (e.g., 'network_available', 'dependency_installed:edge-tts')"
        }
    }
}

# Skill metadata schema for skill index entries (extended)
SKILL_INDEX_ENTRY_SCHEMA = {
    "type": "object",
    "required": ["id", "name", "path", "description"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "path": {"type": "string"},
        "description": {"type": "string"},
        "tier": {"type": "string", "enum": ["core", "tested", "untested"]},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "usage_count": {"type": "integer"},
        "success_rate": {"type": "number"},
        "last_used": {"type": ["string", "null"]},
        # Operational fields
        "requires_network": {"type": "boolean", "default": False},
        "timeout_seconds": {"type": "integer", "default": 300},
        "expected_runtime_seconds": {"type": ["integer", "null"]},
        "dependencies": {"type": "array", "items": {"type": "string"}},
        "side_effects": {"type": "array", "items": {"type": "string"}},
        "fallback_skill": {"type": ["string", "null"]}
    }
}

# Rule artifact data schema
RULE_DATA_SCHEMA = {
    "type": "object",
    "required": ["name", "rule_type", "content"],
    "properties": {
        "name": {
            "type": "string",
            "description": "Rule name"
        },
        "rule_type": {
            "type": "string",
            "enum": ["hard", "soft"],
            "description": "Hard rules are mandatory, soft are guidance"
        },
        "content": {
            "type": "string",
            "description": "The actual rule content"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keywords that trigger this rule"
        },
        "applies_to": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Contexts where rule applies"
        },
        "exceptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "When rule doesn't apply"
        }
    }
}

# Log artifact data schema
LOG_DATA_SCHEMA = {
    "type": "object",
    "required": ["event_type", "message"],
    "properties": {
        "event_type": {
            "type": "string",
            "enum": ["task_start", "task_complete", "task_fail", "learning", "error", "info"],
            "description": "Type of log event"
        },
        "message": {
            "type": "string",
            "description": "Log message"
        },
        "task": {
            "type": ["string", "null"],
            "description": "Related task if applicable"
        },
        "outcome": {
            "type": ["string", "null"],
            "description": "Outcome for task events"
        },
        "error": {
            "type": ["string", "null"],
            "description": "Error details if applicable"
        },
        "lesson": {
            "type": ["string", "null"],
            "description": "Lesson learned from failure"
        },
        "duration_ms": {
            "type": ["integer", "null"],
            "description": "Duration in milliseconds"
        }
    }
}

# Episode artifact data schema (Phase 1: Feedback Loop)
EPISODE_DATA_SCHEMA = {
    "type": "object",
    "required": ["goal"],
    "properties": {
        "goal": {
            "type": "string",
            "description": "What this episode is trying to achieve"
        },
        "status": {
            "type": "string",
            "enum": ["open", "closed"],
            "default": "open",
            "description": "Whether episode is still active"
        },
        "plan": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Planned steps to achieve the goal"
        },
        "context": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Domain of work (e.g., research, coding, design)"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints like 'use official sources', 'timebox 20m'"
                },
                "environment": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "tools_available": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "tool": {"type": "string"},
                    "summary": {"type": "string"},
                    "timestamp": {"type": "string"}
                }
            },
            "description": "Actions taken during this episode (refs, not full outputs)"
        },
        "result": {
            "type": ["string", "null"],
            "enum": ["success", "partial", "failed", None],
            "description": "Final outcome of the episode"
        },
        "result_summary": {
            "type": ["string", "null"],
            "description": "Brief summary of what was achieved"
        },
        "links": {
            "type": "object",
            "properties": {
                "facts_created": {"type": "array", "items": {"type": "string"}},
                "decisions_created": {"type": "array", "items": {"type": "string"}},
                "skills_used": {"type": "array", "items": {"type": "string"}},
                "evaluation_id": {"type": ["string", "null"]}
            },
            "description": "References to artifacts created/used during this episode"
        },
        "started_at": {
            "type": "string",
            "format": "date-time",
            "description": "When the episode started"
        },
        "closed_at": {
            "type": ["string", "null"],
            "format": "date-time",
            "description": "When the episode was closed"
        },
        "duration_mins": {
            "type": ["number", "null"],
            "description": "Duration in minutes"
        }
    }
}

# Evaluation artifact data schema (Phase 1: Feedback Loop)
EVALUATION_DATA_SCHEMA = {
    "type": "object",
    "required": ["episode_id", "rubric"],
    "properties": {
        "episode_id": {
            "type": "string",
            "description": "ID of the episode being evaluated"
        },
        "rubric": {
            "type": "object",
            "properties": {
                "outcome_quality": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 5},
                        "max": {"type": "integer", "default": 5},
                        "notes": {"type": "string"}
                    }
                },
                "cost": {
                    "type": "object",
                    "properties": {
                        "duration_mins": {"type": "number"},
                        "tools_used": {"type": "integer"},
                        "tokens_bucket": {
                            "type": "string",
                            "enum": ["XS", "S", "M", "L", "XL"],
                            "description": "XS<1k, S<5k, M<20k, L<50k, XL>50k"
                        }
                    }
                },
                "correctness_risk": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 5},
                        "max": {"type": "integer", "default": 5},
                        "notes": {"type": "string"}
                    },
                    "description": "Higher = riskier (more likely to contain errors)"
                },
                "reusability": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 5},
                        "max": {"type": "integer", "default": 5},
                        "notes": {"type": "string"}
                    }
                },
                "reproducibility": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 5},
                        "max": {"type": "integer", "default": 5},
                        "notes": {"type": "string"}
                    },
                    "description": "Could another agent reproduce this with same inputs?"
                }
            }
        },
        "grade": {
            "type": "string",
            "description": "Overall grade (A+, A, B+, B, C, D, F)"
        },
        "memory_updates": {
            "type": "object",
            "properties": {
                "reinforce": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["fact", "decision", "skill"]},
                            "id": {"type": "string"},
                            "delta": {"type": "number", "description": "Confidence/stat delta (capped ±0.02)"}
                        }
                    }
                },
                "decay": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["fact", "decision", "skill"]},
                            "id": {"type": "string"},
                            "delta_confidence": {"type": "number", "description": "Negative delta (capped -0.02)"}
                        }
                    }
                }
            }
        },
        "applied": {
            "type": "boolean",
            "default": False,
            "description": "Whether memory_updates have been applied"
        },
        "applied_at": {
            "type": ["string", "null"],
            "format": "date-time",
            "description": "When updates were applied"
        },
        "next_change": {
            "type": ["string", "null"],
            "description": "What to do differently next time"
        }
    }
}

# Skill stats schema (for tracking skill performance)
SKILL_STATS_SCHEMA = {
    "type": "object",
    "required": ["skill_id", "name"],
    "properties": {
        "skill_id": {"type": "string"},
        "name": {"type": "string"},
        "total_uses": {"type": "integer", "default": 0},
        "successes": {"type": "integer", "default": 0},
        "failures": {"type": "integer", "default": 0},
        "success_rate": {"type": "number", "minimum": 0, "maximum": 1},
        "avg_duration_ms": {"type": ["number", "null"]},
        "last_used": {"type": ["string", "null"]},
        "last_outcome": {"type": ["string", "null"]},
        "confidence": {
            "type": "number",
            "minimum": 0.05,
            "maximum": 0.99,
            "default": 0.5,
            "description": "Confidence in skill effectiveness (capped 0.05-0.99)"
        }
    }
}

# Map type to data schema
DATA_SCHEMAS = {
    "fact": FACT_DATA_SCHEMA,
    "decision": DECISION_DATA_SCHEMA,
    "skill": SKILL_DATA_SCHEMA,
    "rule": RULE_DATA_SCHEMA,
    "log": LOG_DATA_SCHEMA,
    "episode": EPISODE_DATA_SCHEMA,
    "evaluation": EVALUATION_DATA_SCHEMA,
    "skill_stats": SKILL_STATS_SCHEMA
}

# Map type to storage directory
TYPE_DIRECTORIES = {
    "fact": "facts",
    "decision": "decisions",
    "skill": "skills",
    "rule": "rules",
    "log": "logs",
    "episode": "episodes",
    "evaluation": "evaluations",
    "skill_stats": "skill_stats"
}


def validate_artifact(artifact: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate an artifact against schemas.
    Returns (is_valid, error_message).
    """
    # Check required envelope fields
    required = ["id", "type", "version", "created_at", "sensitivity", "tags", "source", "data"]
    for field in required:
        if field not in artifact:
            return False, f"Missing required field: {field}"

    # Check type is valid
    artifact_type = artifact.get("type")
    if artifact_type not in DATA_SCHEMAS:
        return False, f"Invalid artifact type: {artifact_type}"

    # Check id format (allow ep_, eval_, ss_ prefixes for episode, evaluation, skill_stats)
    # skill_stats can use deterministic IDs like ss_web_research (no timestamp required)
    id_val = artifact.get("id", "")
    prefix_map = {
        "episode": "ep_",
        "evaluation": "eval_",
        "skill_stats": "ss_"
    }
    expected_prefix = prefix_map.get(artifact_type, f"{artifact_type}_")
    if not id_val.startswith(expected_prefix):
        return False, f"ID must start with '{expected_prefix}'"

    # Check sensitivity
    if artifact.get("sensitivity") not in ["public", "internal", "sensitive"]:
        return False, "Invalid sensitivity level"

    # Check tags is a list
    if not isinstance(artifact.get("tags"), list):
        return False, "Tags must be a list"

    # Check source has workflow
    source = artifact.get("source", {})
    if not isinstance(source, dict) or "workflow" not in source:
        return False, "Source must have workflow field"

    # Check data is present and is dict
    data = artifact.get("data")
    if not isinstance(data, dict):
        return False, "Data must be an object"

    # Type-specific data validation
    data_schema = DATA_SCHEMAS[artifact_type]
    required_data = data_schema.get("properties", {})

    # Check required data fields based on schema
    if artifact_type == "fact" and "claim" not in data:
        return False, "Fact must have 'claim' field"
    elif artifact_type == "decision" and ("decision" not in data or "rationale" not in data):
        return False, "Decision must have 'decision' and 'rationale' fields"
    elif artifact_type == "skill" and ("name" not in data or "description" not in data or "steps" not in data):
        return False, "Skill must have 'name', 'description', and 'steps' fields"
    elif artifact_type == "rule" and ("name" not in data or "rule_type" not in data or "content" not in data):
        return False, "Rule must have 'name', 'rule_type', and 'content' fields"
    elif artifact_type == "log" and ("event_type" not in data or "message" not in data):
        return False, "Log must have 'event_type' and 'message' fields"
    elif artifact_type == "episode" and "goal" not in data:
        return False, "Episode must have 'goal' field"
    elif artifact_type == "evaluation" and ("episode_id" not in data or "rubric" not in data):
        return False, "Evaluation must have 'episode_id' and 'rubric' fields"
    elif artifact_type == "skill_stats" and ("skill_id" not in data or "name" not in data):
        return False, "Skill stats must have 'skill_id' and 'name' fields"

    return True, ""


def apply_backward_compat_defaults(artifact: dict[str, Any]) -> dict[str, Any]:
    """
    Apply safe defaults for missing fields on existing artifacts.
    This ensures old artifacts work with new code that expects new fields.

    Defaults:
    - missing freshness -> "stable"
    - missing decay_rate -> 0
    - missing last_verified -> created_at
    - missing outcome on decisions -> auto-create with status="unverified"
    """
    artifact_type = artifact.get("type")
    data = artifact.get("data", {})

    if artifact_type == "fact":
        # Freshness defaults
        if "freshness" not in data:
            data["freshness"] = "stable"
        if "decay_rate" not in data:
            data["decay_rate"] = 0
        if "last_verified" not in data:
            data["last_verified"] = artifact.get("created_at")
        # Confidence tracking
        if "stored_confidence" not in data and "confidence" in data:
            data["stored_confidence"] = data["confidence"]
        if "usage_count" not in data:
            data["usage_count"] = 0
        if "reinforcement_count" not in data:
            data["reinforcement_count"] = 0

    elif artifact_type == "decision":
        # Outcome defaults
        if "outcome" not in data or data["outcome"] is None:
            data["outcome"] = {"status": "unverified", "verified_at": None}
        elif isinstance(data["outcome"], str):
            # Convert old string outcome to new object format
            data["outcome"] = {"status": data["outcome"], "verified_at": None}

    elif artifact_type == "skill_stats":
        # Ensure confidence is capped
        if "confidence" in data:
            data["confidence"] = max(0.05, min(0.99, data["confidence"]))
        else:
            data["confidence"] = 0.5

    artifact["data"] = data
    return artifact
