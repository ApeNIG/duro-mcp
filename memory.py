"""
Duro Memory Module
Handles loading and saving persistent memory across sessions.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DuroMemory:
    def __init__(self, config: dict):
        self.memory_dir = Path(config["paths"]["memory_dir"])
        self.core_memory_file = self.memory_dir / config["files"]["memory_core"]
        self.agent_root = Path(config["paths"]["agent_root"])
        self.soul_file = self.agent_root / config["files"]["soul"]

    def get_today_file(self) -> Path:
        """Get path to today's memory log file."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.memory_dir / f"{today}.md"

    def load_soul(self) -> str:
        """Load the soul configuration."""
        if self.soul_file.exists():
            return self.soul_file.read_text(encoding="utf-8")
        return ""

    def load_core_memory(self) -> str:
        """Load the core long-term memory."""
        if self.core_memory_file.exists():
            return self.core_memory_file.read_text(encoding="utf-8")
        return ""

    def load_today_memory(self) -> str:
        """Load today's session memory."""
        today_file = self.get_today_file()
        if today_file.exists():
            return today_file.read_text(encoding="utf-8")
        return ""

    def load_recent_memory(self, days: int = 3) -> Dict[str, str]:
        """Load memory from recent days."""
        memories = {}
        for i in range(days):
            date = datetime.now()
            from datetime import timedelta
            date = date - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            file_path = self.memory_dir / f"{date_str}.md"
            if file_path.exists():
                memories[date_str] = file_path.read_text(encoding="utf-8")
        return memories

    def load_full_context(self) -> Dict[str, any]:
        """Load complete context for session start."""
        return {
            "soul": self.load_soul(),
            "core_memory": self.load_core_memory(),
            "today_memory": self.load_today_memory(),
            "recent_memories": self.load_recent_memory(days=3),
            "timestamp": datetime.now().isoformat()
        }

    def save_to_today(self, content: str, section: str = "Session Log") -> bool:
        """Append content to today's memory file."""
        today_file = self.get_today_file()
        timestamp = datetime.now().strftime("%H:%M")

        entry = f"\n### [{timestamp}] {section}\n{content}\n"

        if today_file.exists():
            existing = today_file.read_text(encoding="utf-8")
            today_file.write_text(existing + entry, encoding="utf-8")
        else:
            header = f"# Memory Log - {datetime.now().strftime('%Y-%m-%d')}\n"
            today_file.write_text(header + entry, encoding="utf-8")

        return True

    def save_learning(self, learning: str, category: str = "General") -> bool:
        """Save a specific learning to today's memory."""
        return self.save_to_today(f"**Learning ({category}):** {learning}", "Learnings")

    def save_task_completed(self, task: str, outcome: str) -> bool:
        """Log a completed task."""
        content = f"**Task:** {task}\n**Outcome:** {outcome}"
        return self.save_to_today(content, "Task Completed")

    def save_failure(self, task: str, error: str, lesson: str) -> bool:
        """Log a failure with lesson learned."""
        content = f"**Task:** {task}\n**Error:** {error}\n**Lesson:** {lesson}"
        return self.save_to_today(content, "Failure Logged")

    def update_core_memory(self, section: str, content: str) -> bool:
        """Update a section in core memory."""
        if not self.core_memory_file.exists():
            return False

        core = self.core_memory_file.read_text(encoding="utf-8")

        # Simple append if section doesn't exist
        if f"## {section}" not in core:
            core += f"\n## {section}\n{content}\n"

        self.core_memory_file.write_text(core, encoding="utf-8")
        return True

    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage."""
        memory_files = list(self.memory_dir.glob("*.md"))
        return {
            "total_memory_files": len(memory_files),
            "core_memory_exists": self.core_memory_file.exists(),
            "today_file_exists": self.get_today_file().exists(),
            "memory_dir": str(self.memory_dir)
        }
