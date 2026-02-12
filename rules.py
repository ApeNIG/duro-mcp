"""
Duro Rules Module
Handles rule matching, application, and tracking.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DuroRules:
    def __init__(self, config: dict):
        self.rules_dir = Path(config["paths"]["rules_dir"])
        self.index_file = self.rules_dir / config["files"]["rules_index"]
        self._index_cache = None

    def _load_index(self, force_reload: bool = False) -> dict:
        """Load the rules index."""
        if self._index_cache is None or force_reload:
            if self.index_file.exists():
                self._index_cache = json.loads(self.index_file.read_text(encoding="utf-8"))
            else:
                self._index_cache = {"rules": [], "version": "1.0"}
        return self._index_cache

    def _save_index(self, index: dict) -> bool:
        """Save the rules index."""
        self.index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")
        self._index_cache = index
        return True

    def list_rules(self) -> List[Dict]:
        """List all rules."""
        index = self._load_index()
        return index.get("rules", [])

    def get_rule(self, rule_id: str) -> Optional[Dict]:
        """Get a specific rule by ID or name."""
        rules = self.list_rules()
        for rule in rules:
            if rule["id"] == rule_id or rule["name"] == rule_id:
                return rule
        return None

    def load_rule_content(self, rule: Dict) -> Optional[Dict]:
        """Load the full content of a rule from its file."""
        rule_file = self.rules_dir / rule["file"]
        if rule_file.exists():
            return json.loads(rule_file.read_text(encoding="utf-8"))
        return None

    def check_rules(self, task_description: str) -> List[Dict]:
        """
        Check which rules apply to a given task.
        Returns list of applicable rules with their full content.
        """
        rules = self.list_rules()
        applicable = []
        task_lower = task_description.lower()

        for rule in rules:
            keywords = rule.get("trigger_keywords", [])
            for kw in keywords:
                if kw.lower() in task_lower:
                    # Load full rule content
                    content = self.load_rule_content(rule)
                    applicable.append({
                        "rule": rule,
                        "content": content,
                        "matched_keyword": kw
                    })
                    break

        # Sort by priority (1 = highest) and type (hard before soft)
        type_priority = {"hard": 0, "soft": 1}
        applicable.sort(key=lambda r: (
            r["rule"].get("priority", 99),
            type_priority.get(r["rule"].get("type", "soft"), 2)
        ))

        return applicable

    def apply_rule(self, rule_id: str) -> bool:
        """Mark a rule as used and update stats."""
        index = self._load_index(force_reload=True)
        for rule in index.get("rules", []):
            if rule["id"] == rule_id or rule["name"] == rule_id:
                rule["usage_count"] = rule.get("usage_count", 0) + 1
                rule["last_used"] = datetime.now().strftime("%Y-%m-%d")
                self._save_index(index)
                return True
        return False

    def get_hard_rules(self) -> List[Dict]:
        """Get all hard (always enforced) rules."""
        rules = self.list_rules()
        return [r for r in rules if r.get("type") == "hard"]

    def get_soft_rules(self) -> List[Dict]:
        """Get all soft (preference) rules."""
        rules = self.list_rules()
        return [r for r in rules if r.get("type") == "soft"]

    def format_rules_for_context(self, rules: List[Dict]) -> str:
        """Format applicable rules as context string."""
        if not rules:
            return "No specific rules apply to this task."

        lines = ["## Applicable Rules\n"]
        for r in rules:
            rule = r["rule"]
            content = r.get("content", {})

            rule_type = "MUST" if rule.get("type") == "hard" else "SHOULD"
            lines.append(f"### [{rule_type}] {rule['name']}")
            lines.append(f"*Triggered by: {r.get('matched_keyword', 'N/A')}*\n")

            if content:
                if "description" in content:
                    lines.append(f"**What:** {content['description']}")
                if "action" in content:
                    lines.append(f"**Do:** {content['action']}")
                if "avoid" in content:
                    lines.append(f"**Avoid:** {content['avoid']}")
                if "reason" in content:
                    lines.append(f"**Why:** {content['reason']}")

            lines.append("")

        return "\n".join(lines)

    def get_rules_summary(self) -> Dict:
        """Get summary of rules system."""
        rules = self.list_rules()
        return {
            "total_rules": len(rules),
            "hard_rules": len([r for r in rules if r.get("type") == "hard"]),
            "soft_rules": len([r for r in rules if r.get("type") == "soft"]),
            "rules": [{"name": r["name"], "type": r.get("type"), "keywords": r.get("trigger_keywords", [])} for r in rules]
        }
