"""
Duro Skills Module
Handles skill discovery, lookup, and execution.

Skill Interface Convention:
- Each skill module exposes:
  - SKILL_META (dict): name, description, tier, requires (list of capabilities)
  - run(args: dict, tools: dict, context: dict) -> dict

The orchestrator builds the tools dict with capability wrappers.
Skills never see server names - they just call tools["search"], tools["read"], etc.
"""

import os
import json
import importlib.util
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any


class DuroSkills:
    def __init__(self, config: dict):
        self.skills_dir = Path(config["paths"]["skills_dir"])
        self.index_file = self.skills_dir / config["files"]["skills_index"]
        self.log_usage = config["settings"].get("log_skill_usage", True)
        self._index_cache = None

    def _load_index(self, force_reload: bool = False) -> dict:
        """Load the skills index."""
        if self._index_cache is None or force_reload:
            if self.index_file.exists():
                self._index_cache = json.loads(self.index_file.read_text(encoding="utf-8"))
            else:
                self._index_cache = {"skills": [], "version": "1.0"}
        return self._index_cache

    def _save_index(self, index: dict) -> bool:
        """Save the skills index."""
        self.index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")
        self._index_cache = index
        return True

    def list_skills(self) -> List[Dict]:
        """List all available skills."""
        index = self._load_index()
        return index.get("skills", [])

    def get_skill(self, skill_name: str) -> Optional[Dict]:
        """Get a specific skill by name."""
        skills = self.list_skills()
        for skill in skills:
            if skill["name"] == skill_name or skill["id"] == skill_name:
                return skill
        return None

    def find_skills(self, keywords: List[str]) -> List[Dict]:
        """Find skills matching keywords."""
        skills = self.list_skills()
        matches = []

        for skill in skills:
            skill_keywords = skill.get("keywords", [])
            skill_name = skill.get("name", "").lower()
            skill_desc = skill.get("description", "").lower()

            for kw in keywords:
                kw_lower = kw.lower()
                if (kw_lower in skill_keywords or
                    kw_lower in skill_name or
                    kw_lower in skill_desc):
                    matches.append(skill)
                    break

        # Sort by tier priority and usage count
        tier_priority = {"core": 0, "tested": 1, "untested": 2}
        matches.sort(key=lambda s: (
            tier_priority.get(s.get("tier", "untested"), 3),
            -s.get("usage_count", 0)
        ))

        return matches

    def get_skill_path(self, skill: Dict) -> Path:
        """Get the full path to a skill's Python file."""
        return self.skills_dir / skill["path"]

    def run_skill(self, skill_name: str, args: Dict = None) -> Tuple[bool, str]:
        """
        Execute a skill by name.
        Returns (success, output).
        """
        skill = self.get_skill(skill_name)
        if not skill:
            return False, f"Skill '{skill_name}' not found"

        skill_path = self.get_skill_path(skill)
        if not skill_path.exists():
            return False, f"Skill file not found: {skill_path}"

        try:
            # Build command
            cmd = [sys.executable, str(skill_path)]

            # Add arguments if provided
            if args:
                for key, value in args.items():
                    cmd.extend([f"--{key}", str(value)])

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(self.skills_dir)
            )

            # Update usage stats
            self._update_skill_stats(skill_name, success=(result.returncode == 0))

            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, f"Error: {result.stderr}"

        except subprocess.TimeoutExpired:
            self._update_skill_stats(skill_name, success=False)
            return False, "Skill execution timed out (5 min limit)"
        except Exception as e:
            self._update_skill_stats(skill_name, success=False)
            return False, f"Execution error: {str(e)}"

    def run_skill_with_tools(
        self,
        skill_name: str,
        args: Dict,
        tools: Dict[str, Callable],
        context: Dict,
        timeout_seconds: int = 60
    ) -> Tuple[bool, Dict]:
        """
        Execute a skill using the new interface with tools dict.

        Args:
            skill_name: Name of the skill to run
            args: Arguments for the skill
            tools: Dict of capability name -> callable wrapper
            context: Run context (run_id, constraints, etc.)
            timeout_seconds: Max execution time

        Returns:
            (success, result_dict)
        """
        skill = self.get_skill(skill_name)
        if not skill:
            return False, {"error": f"Skill '{skill_name}' not found"}

        skill_path = self.get_skill_path(skill)
        if not skill_path.exists():
            return False, {"error": f"Skill file not found: {skill_path}"}

        try:
            # Load skill module dynamically
            spec = importlib.util.spec_from_file_location(skill_name, skill_path)
            if spec is None or spec.loader is None:
                return False, {"error": f"Cannot load skill module: {skill_path}"}

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check for required interface
            if not hasattr(module, 'run'):
                return False, {"error": f"Skill '{skill_name}' missing run() function"}

            # Check required capabilities
            required = getattr(module, 'REQUIRES', [])
            missing = [cap for cap in required if cap not in tools]
            if missing:
                return False, {"error": f"Missing capabilities: {missing}"}

            # Execute with timeout
            result = {"error": "Timeout"}
            exception_holder = [None]

            def execute():
                nonlocal result
                try:
                    result = module.run(args, tools, context)
                except Exception as e:
                    exception_holder[0] = e

            thread = threading.Thread(target=execute)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                # Timeout - thread still running
                self._update_skill_stats(skill_name, success=False)
                return False, {"error": f"Skill timed out after {timeout_seconds}s", "timeout": True}

            if exception_holder[0]:
                self._update_skill_stats(skill_name, success=False)
                return False, {"error": str(exception_holder[0])}

            # Check result
            success = result.get("success", False) if isinstance(result, dict) else False
            self._update_skill_stats(skill_name, success=success)

            return success, result

        except Exception as e:
            self._update_skill_stats(skill_name, success=False)
            return False, {"error": f"Skill execution error: {str(e)}"}

    def get_skill_meta(self, skill_name: str) -> Optional[Dict]:
        """Get SKILL_META from a skill module without executing it."""
        skill = self.get_skill(skill_name)
        if not skill:
            return None

        skill_path = self.get_skill_path(skill)
        if not skill_path.exists():
            return None

        try:
            spec = importlib.util.spec_from_file_location(skill_name, skill_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            return getattr(module, 'SKILL_META', None)
        except Exception:
            return None

    def _update_skill_stats(self, skill_name: str, success: bool) -> None:
        """Update usage statistics for a skill."""
        if not self.log_usage:
            return

        index = self._load_index(force_reload=True)
        for skill in index.get("skills", []):
            if skill["name"] == skill_name or skill["id"] == skill_name:
                skill["usage_count"] = skill.get("usage_count", 0) + 1
                # Update success rate
                total = skill["usage_count"]
                current_rate = skill.get("success_rate", 1.0)
                if success:
                    skill["success_rate"] = ((current_rate * (total - 1)) + 1) / total
                else:
                    skill["success_rate"] = (current_rate * (total - 1)) / total
                skill["last_used"] = datetime.now().isoformat()
                break

        self._save_index(index)

    def get_skill_code(self, skill_name: str) -> Optional[str]:
        """Get the source code of a skill."""
        skill = self.get_skill(skill_name)
        if not skill:
            return None

        skill_path = self.get_skill_path(skill)
        if skill_path.exists():
            return skill_path.read_text(encoding="utf-8")
        return None

    def get_skills_summary(self) -> Dict:
        """Get a summary of the skills system."""
        skills = self.list_skills()
        return {
            "total_skills": len(skills),
            "by_tier": {
                "core": len([s for s in skills if s.get("tier") == "core"]),
                "tested": len([s for s in skills if s.get("tier") == "tested"]),
                "untested": len([s for s in skills if s.get("tier") == "untested"])
            },
            "skills": [{"name": s["name"], "tier": s.get("tier"), "description": s.get("description", "")} for s in skills]
        }
