# Duro Orchestration Phase 1: Workflow Selector

## Overview

A lightweight routing layer that:
1. Receives a task intent
2. Checks applicable rules
3. Selects and executes the appropriate skill
4. Logs execution consistently

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     duro_orchestrate                        │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │  Intent  │ -> │  Rules   │ -> │  Skill   │ -> │  Log  │ │
│  │  Parse   │    │  Check   │    │  Execute │    │  Run  │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### WorkflowRun (logged to `runs/` directory)
```json
{
  "run_id": "run_20260210_220000_abc123",
  "timestamp": "2026-02-10T22:00:00Z",
  "intent": "store fact about X",
  "intent_type": "store_fact",
  "rules_checked": ["rule_005"],
  "rules_applied": ["Fact Verification Requirements"],
  "skill_selected": "verify_and_store_fact",
  "skill_args": {"claim": "...", "confidence": 0.9},
  "outcome": "success",
  "artifacts_created": ["fact_20260210_..."],
  "duration_ms": 1234,
  "error": null
}
```

## Intent Types & Routing

| Intent Pattern | Intent Type | Rules to Check | Default Skill |
|----------------|-------------|----------------|---------------|
| "store fact", "record fact" | `store_fact` | rule_005 | → route by confidence |
| "remember", "note" | `store_note` | none | `duro_store_fact` (low conf) |
| "decide", "chose" | `store_decision` | none | `duro_store_decision` |
| "generate audio", "tts" | `generate_audio` | rule_002, rule_003 | `generate_tts` |

### Fact Storage Routing Logic
```python
def route_fact_storage(claim: str, confidence: float, sources: list) -> str:
    """Route to appropriate fact storage skill."""
    if confidence >= 0.8:
        if not sources:
            # Rule 005 requires verification for high confidence
            return "verify_and_store_fact"  # Will web search first
        else:
            return "duro_store_fact"  # Already has sources
    else:
        # Low confidence = notes, no verification needed
        return "duro_store_fact"
```

## New MCP Tool

### `duro_orchestrate`
```python
Tool(
    name="duro_orchestrate",
    description="Route a task through the workflow selector. Checks rules, selects skill, executes, logs.",
    inputSchema={
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "What you want to do (e.g., 'store a verified fact about X')"
            },
            "args": {
                "type": "object",
                "description": "Arguments for the task",
                "default": {}
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, show what would happen without executing",
                "default": False
            }
        },
        "required": ["intent"]
    }
)
```

### Example Usage

**Input:**
```json
{
  "intent": "store fact",
  "args": {
    "claim": "Claude Opus 4.5 was released in early 2026",
    "confidence": 0.9
  }
}
```

**Orchestrator Flow:**
1. Parse intent → `store_fact`
2. Check rules → rule_005 applies (high confidence fact)
3. Check args → confidence 0.9, no sources
4. Route → `verify_and_store_fact` (will web search first)
5. Execute skill
6. Log run to `runs/run_20260210_...json`
7. Return result

**Output:**
```json
{
  "run_id": "run_20260210_220000_abc123",
  "skill_used": "verify_and_store_fact",
  "rules_applied": ["Fact Verification Requirements"],
  "outcome": "success",
  "artifacts": ["fact_20260210_220005_xyz789"]
}
```

## Implementation Plan

### Phase 1a: Core Router (minimal)
```
orchestrator.py
├── parse_intent(text) -> IntentType
├── check_rules(intent, args) -> list[Rule]
├── select_skill(intent, args, rules) -> SkillName
└── execute(skill, args) -> Result
```

### Phase 1b: Run Logging
```
memory/runs/
├── run_20260210_220000_abc123.json
├── run_20260210_220100_def456.json
└── ...
```

### Phase 1c: MCP Integration
- Add `duro_orchestrate` tool
- Add `duro_list_runs` tool (query run history)
- Add `duro_get_run` tool (get run details)

## File Structure

```
duro-mcp/
├── orchestrator.py      # NEW: Workflow selector
├── artifacts.py         # Existing
├── skills.py           # Existing
├── rules.py            # Existing
└── duro_mcp_server.py  # Add new tools

memory/
├── runs/               # NEW: Run logs
│   └── *.json
├── facts/
├── decisions/
└── logs/
```

## Key Design Decisions

1. **Thin Layer**: Orchestrator doesn't add new logic, just routes existing pieces
2. **Run = Audit**: Every orchestrated action gets a run log (queryable history)
3. **Dry Run**: Can preview what would happen before executing
4. **Idempotent**: Same intent + args = same routing (deterministic)

## Next Steps

1. Create `orchestrator.py` with core routing logic
2. Add `runs/` directory to schema
3. Implement `duro_orchestrate` MCP tool
4. Test with fact storage routing (most complex case)
5. Add `duro_list_runs` for run history queries
