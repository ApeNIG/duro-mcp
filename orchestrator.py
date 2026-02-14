"""
Duro Orchestrator - Workflow selector and run logger.

The orchestrator is a thin routing layer that:
1. Normalizes intent
2. Checks rules
3. Selects an action plan
4. Executes (or dry-runs)
5. Writes a run log linking everything together

One new concept: a "run" - the auditable trace of an orchestrated action.
"""

import json
import random
import re
import string
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from time_utils import utc_now, utc_now_iso
from pathlib import Path
from typing import Any, Optional

# Version info for run logs
SERVER_BUILD = "1.1.0"
SCHEMA_VERSION = "1.1"

# Stop conditions
MAX_TOOL_CALLS = 10
MAX_RETRIES = 3
MAX_SECONDS = 60
SKILL_TIMEOUT_SECONDS = 60

# External tool mapping: capability -> (server, tool_name)
# This is a tiny mapping, not a full registry
EXTERNAL_TOOL_MAP = {
    "search": ("superagi", "web_search"),
    "read": ("superagi", "read_webpage"),
}


@dataclass
class RuleDecision:
    """A decision made based on a rule."""
    rule_id: str
    severity: str  # hard, soft
    decision: str  # ALLOW, CONSTRAIN, DENY
    notes: str = ""


@dataclass
class Plan:
    """The selected action plan."""
    selected: str  # skill or tool name
    type: str  # "skill" or "tool"
    reason: str
    constraints: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    """A single tool call during execution."""
    name: str
    ok: bool
    ms: int
    error: Optional[str] = None


@dataclass
class RunLog:
    """Complete run log structure."""
    run_id: str
    started_at: str
    finished_at: Optional[str]
    intent: str
    intent_normalized: str
    args: dict
    dry_run: bool
    sensitivity: str

    rules_checked: bool
    rules_applicable: list
    rules_decisions: list

    plan_selected: Optional[str]
    plan_type: Optional[str]
    plan_reason: Optional[str]
    plan_constraints: dict

    tool_calls: list
    outcome: str  # pending, success, failed, denied, dry_run
    error: Optional[str]
    duration_ms: int

    artifacts_created: list
    artifact_paths: list
    notes: list

    server_build: str = SERVER_BUILD
    schema_version: str = SCHEMA_VERSION


def generate_run_id() -> str:
    """Generate unique run ID."""
    now = utc_now()
    date_part = now.strftime("%Y%m%d")
    time_part = now.strftime("%H%M%S")
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"run_{date_part}_{time_part}_{random_part}"


def normalize_intent(intent: str) -> str:
    """
    Normalize intent string to a canonical form.
    Returns one of: store_fact, store_decision, delete_artifact, unknown
    """
    intent_lower = intent.lower().strip()

    # store_fact patterns
    if any(p in intent_lower for p in ["store fact", "save fact", "record fact", "add fact"]):
        return "store_fact"
    if any(p in intent_lower for p in ["remember", "note that", "note:"]):
        return "store_fact"  # Low-confidence note

    # store_decision patterns
    if any(p in intent_lower for p in ["store decision", "record decision", "decided", "chose"]):
        return "store_decision"

    # delete_artifact patterns
    if any(p in intent_lower for p in ["delete artifact", "remove artifact", "delete fact", "delete decision"]):
        return "delete_artifact"

    return "unknown"


def detect_sensitivity(args: dict, default: str = "internal") -> str:
    """
    Auto-detect sensitivity based on args content.
    If args contain PII-like patterns, upgrade to internal/sensitive.
    """
    args_str = json.dumps(args).lower()

    # Patterns that suggest sensitive data
    sensitive_patterns = [
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # phone
        r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',  # SSN pattern
        r'password', r'secret', r'api[_-]?key', r'token',
        r'credit[_\s]?card', r'ssn', r'social[_\s]?security'
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, args_str):
            return "sensitive"

    return default


class Orchestrator:
    """
    The workflow selector and executor.
    Routes intents through rules to skills, logs everything.
    """

    def __init__(self, memory_dir: Path, rules_module, skills_module, artifact_store, external_tools=None):
        self.memory_dir = Path(memory_dir)
        self.runs_dir = self.memory_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.rules = rules_module
        self.skills = skills_module
        self.artifacts = artifact_store

        # External tools (from other MCP servers like superagi)
        # Dict of capability_name -> callable
        self.external_tools = external_tools or {}

    def _build_tools_dict(self, run_id: str) -> dict:
        """
        Build the tools dict that skills receive.
        Skills call tools["search"], tools["store_fact"], etc.
        They never see server names or implementation details.
        """
        tools = {}

        # Search capability (from superagi or similar)
        if "search" in self.external_tools:
            tools["search"] = self.external_tools["search"]
        else:
            # Stub that returns empty results
            tools["search"] = lambda q, **kw: {"results": [], "error": "search not configured"}

        # Read webpage capability
        if "read" in self.external_tools:
            tools["read"] = self.external_tools["read"]
        else:
            tools["read"] = lambda url, **kw: {"content": "", "error": "read not configured"}

        # Internal Duro tools - wrapped to track calls
        def store_fact_wrapper(**kwargs):
            success, artifact_id, path = self.artifacts.store_fact(
                claim=kwargs.get("claim", ""),
                source_urls=kwargs.get("source_urls"),
                snippet=kwargs.get("snippet"),
                confidence=kwargs.get("confidence", 0.5),
                tags=kwargs.get("tags"),
                workflow=kwargs.get("workflow", run_id),
                sensitivity=kwargs.get("sensitivity", "public"),
                evidence_type=kwargs.get("evidence_type", "none"),
                provenance=kwargs.get("provenance", "unknown")
            )
            return {"success": success, "artifact_id": artifact_id, "path": path}

        def store_decision_wrapper(**kwargs):
            success, artifact_id, path = self.artifacts.store_decision(
                decision=kwargs.get("decision", ""),
                rationale=kwargs.get("rationale", ""),
                alternatives=kwargs.get("alternatives"),
                context=kwargs.get("context"),
                reversible=kwargs.get("reversible", True),
                tags=kwargs.get("tags"),
                workflow=kwargs.get("workflow", run_id),
                sensitivity=kwargs.get("sensitivity", "internal")
            )
            return {"success": success, "artifact_id": artifact_id, "path": path}

        def log_wrapper(msg, **kwargs):
            # Simple logging to memory
            return {"logged": True, "message": msg}

        tools["store_fact"] = store_fact_wrapper
        tools["store_decision"] = store_decision_wrapper
        tools["log"] = log_wrapper

        return tools

    def set_external_tools(self, tools: dict):
        """Set external tool callables (e.g., from superagi MCP)."""
        self.external_tools = tools

    def orchestrate(
        self,
        intent: str,
        args: dict,
        dry_run: bool = False,
        sensitivity: Optional[str] = None
    ) -> dict:
        """
        Main entry point. Route intent through rules to skill, execute, log.

        Returns dict with run_id, outcome, artifacts, etc.
        """
        start_time = time.time()
        run_id = generate_run_id()

        # Normalize
        intent_normalized = normalize_intent(intent)

        # Detect sensitivity
        final_sensitivity = sensitivity or detect_sensitivity(args)

        # Initialize run log
        run = RunLog(
            run_id=run_id,
            started_at=utc_now_iso(),
            finished_at=None,
            intent=intent,
            intent_normalized=intent_normalized,
            args=args,
            dry_run=dry_run,
            sensitivity=final_sensitivity,
            rules_checked=False,
            rules_applicable=[],
            rules_decisions=[],
            plan_selected=None,
            plan_type=None,
            plan_reason=None,
            plan_constraints={},
            tool_calls=[],
            outcome="pending",
            error=None,
            duration_ms=0,
            artifacts_created=[],
            artifact_paths=[],
            notes=[]
        )

        try:
            # Step 1: Check rules
            run = self._apply_rules(run, intent_normalized, args)

            # Check for DENY
            denies = [d for d in run.rules_decisions if d["decision"] == "DENY"]
            if denies:
                run.outcome = "denied"
                run.error = f"Denied by rule: {denies[0]['rule_id']}"
                run.notes.append(f"Rule {denies[0]['rule_id']} blocked execution")
                return self._finalize_run(run, start_time)

            # Step 2: Select plan
            run = self._select_plan(run, intent_normalized, args)

            if run.plan_selected is None:
                run.outcome = "failed"
                run.error = f"No plan available for intent: {intent_normalized}"
                return self._finalize_run(run, start_time)

            # Step 3: Execute or dry-run
            if dry_run:
                run.outcome = "dry_run"
                run.notes.append(f"Would execute: {run.plan_selected}")
            else:
                run = self._execute_plan(run, args)

        except Exception as e:
            run.outcome = "failed"
            run.error = str(e)

        return self._finalize_run(run, start_time)

    def _apply_rules(self, run: RunLog, intent: str, args: dict) -> RunLog:
        """Check rules and record decisions."""
        run.rules_checked = True

        # Build task description for rule matching
        task_desc = f"{intent}: {json.dumps(args)[:200]}"

        # Get applicable rules
        applicable = self.rules.check_rules(task_desc)
        run.rules_applicable = [r["rule"]["name"] for r in applicable]

        # Make decisions based on rules
        for match in applicable:
            rule = match["rule"]
            rule_id = rule.get("id", "unknown")
            severity = rule.get("type", "soft")

            # Determine decision based on rule + args
            decision = self._evaluate_rule(rule, intent, args)
            run.rules_decisions.append({
                "rule_id": rule_id,
                "severity": severity,
                "decision": decision["decision"],
                "notes": decision["notes"]
            })

        return run

    def _evaluate_rule(self, rule: dict, intent: str, args: dict) -> dict:
        """
        Evaluate a single rule against intent and args.
        Returns {"decision": "ALLOW|CONSTRAIN|DENY", "notes": "..."}
        """
        rule_id = rule.get("id", "")

        # Rule 005: Fact Verification Requirements
        if rule_id == "rule_005" and intent == "store_fact":
            confidence = args.get("confidence", 0.5)
            has_sources = bool(args.get("source_urls"))

            if confidence >= 0.8 and not has_sources:
                return {
                    "decision": "CONSTRAIN",
                    "notes": "High confidence requires sources; routing to verify_and_store_fact"
                }

        # Rule: Stop Conditions (rule about retries/errors)
        if "stop" in rule_id.lower():
            return {"decision": "ALLOW", "notes": "Stop conditions checked at execution time"}

        # Default: allow
        return {"decision": "ALLOW", "notes": "No constraints"}

    def _select_plan(self, run: RunLog, intent: str, args: dict) -> RunLog:
        """Select the skill/tool to execute based on intent and constraints."""

        # Check if any rule constrained us
        constraints = {}
        for dec in run.rules_decisions:
            if dec["decision"] == "CONSTRAIN":
                constraints["rule_constrained"] = True

        # ROUTING TABLE
        if intent == "store_fact":
            confidence = args.get("confidence", 0.5)
            has_sources = bool(args.get("source_urls"))

            if confidence >= 0.8 and not has_sources:
                run.plan_selected = "verify_and_store_fact"
                run.plan_type = "skill"
                run.plan_reason = "High confidence without sources requires verification"
                run.plan_constraints = {"require_sources": True}
            else:
                run.plan_selected = "duro_store_fact"
                run.plan_type = "tool"
                run.plan_reason = "Direct storage (low confidence or sources provided)"
                run.plan_constraints = {}

        elif intent == "store_decision":
            run.plan_selected = "duro_store_decision"
            run.plan_type = "tool"
            run.plan_reason = "Direct decision storage"
            run.plan_constraints = {}

        elif intent == "delete_artifact":
            artifact_id = args.get("artifact_id", "")
            # Check if sensitive
            artifact = self.artifacts.get_artifact(artifact_id) if artifact_id else None
            if artifact and artifact.get("sensitivity") == "sensitive":
                if not args.get("force"):
                    run.plan_constraints = {"requires_force": True}
                    run.notes.append("Sensitive artifact requires force=True")

            run.plan_selected = "duro_delete_artifact"
            run.plan_type = "tool"
            run.plan_reason = "Delete with audit logging"

        else:
            run.plan_selected = None
            run.plan_reason = f"Unknown intent: {intent}"

        return run

    def _execute_plan(self, run: RunLog, args: dict) -> RunLog:
        """Execute the selected plan."""

        if run.plan_type == "skill":
            # Execute via skills module with new interface
            call_start = time.time()
            try:
                # Build tools dict for the skill
                tools = self._build_tools_dict(run.run_id)

                # Build context
                context = {
                    "run_id": run.run_id,
                    "constraints": run.plan_constraints,
                    "sensitivity": run.sensitivity,
                    "max_sources": 3,
                    "max_pages": 2,
                }

                # Execute with tools interface
                success, result = self.skills.run_skill_with_tools(
                    skill_name=run.plan_selected,
                    args=args,
                    tools=tools,
                    context=context,
                    timeout_seconds=SKILL_TIMEOUT_SECONDS
                )

                call_ms = int((time.time() - call_start) * 1000)

                run.tool_calls.append({
                    "name": run.plan_selected,
                    "ok": success,
                    "ms": call_ms,
                    "error": result.get("error") if not success else None
                })

                if success:
                    run.outcome = "success"
                    # Extract artifact IDs from result
                    if result.get("artifact_id"):
                        run.artifacts_created.append(result["artifact_id"])
                    if result.get("artifacts_created"):
                        run.artifacts_created.extend(result["artifacts_created"])
                else:
                    # Check for timeout - apply degraded fallback
                    if result.get("timeout") and run.intent_normalized == "store_fact":
                        run = self._degraded_fallback_store_fact(run, args, call_ms)
                    else:
                        run.outcome = "failed"
                        run.error = result.get("error", "Unknown skill error")[:500]

            except Exception as e:
                call_ms = int((time.time() - call_start) * 1000)
                run.tool_calls.append({
                    "name": run.plan_selected,
                    "ok": False,
                    "ms": call_ms,
                    "error": str(e)[:200]
                })
                # Try degraded fallback for store_fact
                if run.intent_normalized == "store_fact":
                    run = self._degraded_fallback_store_fact(run, args, call_ms)
                else:
                    run.outcome = "failed"
                    run.error = str(e)

        elif run.plan_type == "tool":
            # Execute via artifact store directly
            call_start = time.time()
            try:
                if run.plan_selected == "duro_store_fact":
                    # Add provenance
                    success, artifact_id, path = self.artifacts.store_fact(
                        claim=args.get("claim", ""),
                        source_urls=args.get("source_urls"),
                        snippet=args.get("snippet"),
                        confidence=args.get("confidence", 0.5),
                        tags=args.get("tags"),
                        workflow="duro_orchestrate",
                        sensitivity=run.sensitivity,
                        evidence_type=args.get("evidence_type", "none"),
                        provenance=args.get("provenance", "unknown")
                    )
                    call_ms = int((time.time() - call_start) * 1000)

                    run.tool_calls.append({
                        "name": "duro_store_fact",
                        "ok": success,
                        "ms": call_ms
                    })

                    if success:
                        run.outcome = "success"
                        run.artifacts_created.append(artifact_id)
                        run.artifact_paths.append(path)
                    else:
                        run.outcome = "failed"
                        run.error = path  # Error message in path slot

                elif run.plan_selected == "duro_store_decision":
                    success, artifact_id, path = self.artifacts.store_decision(
                        decision=args.get("decision", ""),
                        rationale=args.get("rationale", ""),
                        alternatives=args.get("alternatives"),
                        context=args.get("context"),
                        reversible=args.get("reversible", True),
                        tags=args.get("tags"),
                        workflow="duro_orchestrate",
                        sensitivity=run.sensitivity
                    )
                    call_ms = int((time.time() - call_start) * 1000)

                    run.tool_calls.append({
                        "name": "duro_store_decision",
                        "ok": success,
                        "ms": call_ms
                    })

                    if success:
                        run.outcome = "success"
                        run.artifacts_created.append(artifact_id)
                        run.artifact_paths.append(path)
                    else:
                        run.outcome = "failed"
                        run.error = path

                elif run.plan_selected == "duro_delete_artifact":
                    success, message = self.artifacts.delete_artifact(
                        artifact_id=args.get("artifact_id", ""),
                        reason=args.get("reason", "Orchestrated deletion"),
                        force=args.get("force", False)
                    )
                    call_ms = int((time.time() - call_start) * 1000)

                    run.tool_calls.append({
                        "name": "duro_delete_artifact",
                        "ok": success,
                        "ms": call_ms
                    })

                    if success:
                        run.outcome = "success"
                        run.notes.append(message)
                    else:
                        run.outcome = "failed"
                        run.error = message

                else:
                    run.outcome = "failed"
                    run.error = f"Unknown tool: {run.plan_selected}"

            except Exception as e:
                call_ms = int((time.time() - call_start) * 1000)
                run.tool_calls.append({
                    "name": run.plan_selected,
                    "ok": False,
                    "ms": call_ms,
                    "error": str(e)[:200]
                })
                run.outcome = "failed"
                run.error = str(e)

        return run

    def _degraded_fallback_store_fact(self, run: RunLog, args: dict, original_ms: int) -> RunLog:
        """
        Fallback when skill verification fails/timeouts.
        Stores fact as unverified with capped confidence.
        Outcome is 'degraded_success' - not a hard failure.
        """
        run.notes.append("Verification failed/timeout - storing as unverified")

        call_start = time.time()
        try:
            # Cap confidence at 0.5, mark as unverified
            capped_confidence = min(args.get("confidence", 0.5), 0.5)

            # Add needs_verification tag
            tags = list(args.get("tags") or [])
            if "needs_verification" not in tags:
                tags.append("needs_verification")

            success, artifact_id, path = self.artifacts.store_fact(
                claim=args.get("claim", ""),
                source_urls=None,  # No verified sources
                snippet=None,
                confidence=capped_confidence,
                tags=tags,
                workflow=f"{run.run_id}_degraded",
                sensitivity=run.sensitivity,
                evidence_type="none",
                provenance="unknown"
            )

            call_ms = int((time.time() - call_start) * 1000)

            run.tool_calls.append({
                "name": "duro_store_fact_degraded",
                "ok": success,
                "ms": call_ms,
                "note": "Degraded fallback"
            })

            if success:
                run.outcome = "degraded_success"
                run.artifacts_created.append(artifact_id)
                run.artifact_paths.append(path)
                run.notes.append(f"Stored as unverified with confidence={capped_confidence}")
            else:
                run.outcome = "failed"
                run.error = f"Degraded fallback also failed: {path}"

        except Exception as e:
            run.outcome = "failed"
            run.error = f"Degraded fallback exception: {str(e)}"

        return run

    def _finalize_run(self, run: RunLog, start_time: float) -> dict:
        """Finalize and write run log."""
        run.finished_at = utc_now_iso()
        run.duration_ms = int((time.time() - start_time) * 1000)

        # Write run log
        run_path = self.runs_dir / f"{run.run_id}.json"
        run_dict = self._run_to_dict(run)

        try:
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(run_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            run.notes.append(f"Failed to write run log: {e}")

        # Return summary
        return {
            "run_id": run.run_id,
            "run_path": str(run_path),
            "intent": run.intent_normalized,
            "plan": run.plan_selected,
            "plan_type": run.plan_type,
            "rules_applied": run.rules_applicable,
            "constraints": run.plan_constraints,
            "outcome": run.outcome,
            "error": run.error,
            "artifacts_created": run.artifacts_created,
            "duration_ms": run.duration_ms,
            "dry_run": run.dry_run
        }

    def _run_to_dict(self, run: RunLog) -> dict:
        """Convert RunLog to nested dict matching spec."""
        return {
            "run_id": run.run_id,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "intent": run.intent,
            "intent_normalized": run.intent_normalized,
            "args": run.args,
            "dry_run": run.dry_run,
            "sensitivity": run.sensitivity,

            "rules": {
                "checked": run.rules_checked,
                "applicable": run.rules_applicable,
                "decisions": run.rules_decisions
            },

            "plan": {
                "selected": run.plan_selected,
                "type": run.plan_type,
                "reason": run.plan_reason,
                "constraints": run.plan_constraints
            },

            "execution": {
                "tool_calls": run.tool_calls,
                "outcome": run.outcome,
                "error": run.error,
                "duration_ms": run.duration_ms
            },

            "results": {
                "artifacts_created": run.artifacts_created,
                "artifact_paths": run.artifact_paths,
                "notes": run.notes
            },

            "meta": {
                "server_build": run.server_build,
                "schema_version": run.schema_version
            }
        }

    def get_run(self, run_id: str) -> Optional[dict]:
        """Retrieve a run log by ID."""
        run_path = self.runs_dir / f"{run_id}.json"
        if not run_path.exists():
            return None
        try:
            with open(run_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def list_runs(self, limit: int = 20, outcome: Optional[str] = None) -> list:
        """List recent runs."""
        runs = []
        for run_file in sorted(self.runs_dir.glob("run_*.json"), reverse=True):
            if len(runs) >= limit:
                break
            try:
                with open(run_file, "r", encoding="utf-8") as f:
                    run = json.load(f)
                    if outcome and run.get("execution", {}).get("outcome") != outcome:
                        continue
                    runs.append({
                        "run_id": run.get("run_id"),
                        "intent": run.get("intent_normalized"),
                        "outcome": run.get("execution", {}).get("outcome"),
                        "started_at": run.get("started_at"),
                        "duration_ms": run.get("execution", {}).get("duration_ms")
                    })
            except Exception:
                continue
        return runs
