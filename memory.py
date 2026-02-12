"""
Duro Memory Module
Handles loading and saving persistent memory across sessions.

Hybrid Memory System:
- Today: Full raw logs (all detail preserved)
- Yesterday+: Compressed daily summaries (~500 tokens each)
- Archive: Raw logs stored in archive/ folder (queryable when needed)
"""

import os
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DuroMemory:
    def __init__(self, config: dict):
        self.memory_dir = Path(config["paths"]["memory_dir"])
        self.core_memory_file = self.memory_dir / config["files"]["memory_core"]
        self.agent_root = Path(config["paths"]["agent_root"])
        self.soul_file = self.agent_root / config["files"]["soul"]

        # Archive and summary directories
        self.archive_dir = self.memory_dir / "archive"
        self.summaries_dir = self.memory_dir / "summaries"

        # Ensure directories exist
        self.archive_dir.mkdir(exist_ok=True)
        self.summaries_dir.mkdir(exist_ok=True)

    def get_today_file(self) -> Path:
        """Get path to today's memory log file."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.memory_dir / f"{today}.md"

    def get_summary_file(self, date_str: str) -> Path:
        """Get path to a day's summary file."""
        return self.summaries_dir / f"{date_str}-summary.md"

    def get_archive_file(self, date_str: str) -> Path:
        """Get path to archived raw log."""
        return self.archive_dir / f"{date_str}.md"

    def summarize_day_content(self, raw_content: str, date_str: str) -> str:
        """
        Extract key information from a raw daily log to create a compressed summary.
        Preserves: learnings, task outcomes, failures, key decisions
        Removes: verbose debugging, repeated actions, session noise
        """
        lines = raw_content.split('\n')
        summary_parts = []

        # Header
        summary_parts.append(f"# Summary: {date_str}")

        # Extract learnings
        learnings = []
        for line in lines:
            if '**Learning' in line or 'Learning:' in line:
                # Clean up the learning text
                learning = re.sub(r'\*\*Learning \([^)]*\):\*\*\s*', '', line)
                learning = re.sub(r'Learning:\s*', '', learning)
                if learning.strip():
                    learnings.append(f"- {learning.strip()}")

        if learnings:
            summary_parts.append("\n## Learnings")
            summary_parts.extend(learnings[:10])  # Max 10 learnings

        # Extract completed tasks
        tasks = []
        in_task_block = False
        current_task = {}

        for i, line in enumerate(lines):
            if '### [' in line and 'Task Completed' in line:
                in_task_block = True
                current_task = {'time': line}
            elif in_task_block:
                if '**Task:**' in line:
                    current_task['task'] = line.replace('**Task:**', '').strip()
                elif '**Outcome:**' in line:
                    current_task['outcome'] = line.replace('**Outcome:**', '').strip()
                    tasks.append(f"- {current_task.get('task', 'Unknown')}: {current_task.get('outcome', 'Done')}")
                    in_task_block = False
                    current_task = {}

        if tasks:
            summary_parts.append("\n## Tasks Completed")
            summary_parts.extend(tasks[:15])  # Max 15 tasks

        # Extract failures/lessons
        failures = []
        in_failure_block = False
        current_failure = {}

        for i, line in enumerate(lines):
            if 'Failure Logged' in line:
                in_failure_block = True
                current_failure = {}
            elif in_failure_block:
                if '**Task:**' in line:
                    current_failure['task'] = line.replace('**Task:**', '').strip()
                elif '**Lesson:**' in line:
                    current_failure['lesson'] = line.replace('**Lesson:**', '').strip()
                    failures.append(f"- {current_failure.get('task', 'Task')}: {current_failure.get('lesson', 'Lesson learned')}")
                    in_failure_block = False
                    current_failure = {}

        if failures:
            summary_parts.append("\n## Failures & Lessons")
            summary_parts.extend(failures[:5])  # Max 5 failures

        # Add stats
        summary_parts.append(f"\n## Stats")
        summary_parts.append(f"- Original size: {len(raw_content)} chars")
        summary_parts.append(f"- Total learnings: {len(learnings)}")
        summary_parts.append(f"- Tasks completed: {len(tasks)}")
        summary_parts.append(f"- Failures logged: {len(failures)}")

        return '\n'.join(summary_parts)

    def compress_old_logs(self) -> Dict[str, str]:
        """
        Compress logs older than today:
        1. Create summary for each day
        2. Move raw log to archive
        3. Delete original from memory_dir

        Returns dict of {date: status}
        """
        today = datetime.now().strftime("%Y-%m-%d")
        results = {}

        # Find all date-based log files (YYYY-MM-DD.md format)
        for log_file in self.memory_dir.glob("????-??-??.md"):
            date_str = log_file.stem

            # Skip today's log
            if date_str == today:
                results[date_str] = "skipped (today)"
                continue

            # Skip if already processed
            summary_file = self.get_summary_file(date_str)
            archive_file = self.get_archive_file(date_str)

            if summary_file.exists() and archive_file.exists():
                results[date_str] = "already processed"
                continue

            try:
                # Read raw content
                raw_content = log_file.read_text(encoding="utf-8")

                # Create summary
                summary = self.summarize_day_content(raw_content, date_str)
                summary_file.write_text(summary, encoding="utf-8")

                # Archive raw log
                archive_file.write_text(raw_content, encoding="utf-8")

                # Remove original (now we have summary + archive)
                log_file.unlink()

                results[date_str] = f"compressed ({len(raw_content)} -> {len(summary)} chars)"

            except Exception as e:
                results[date_str] = f"error: {str(e)}"

        return results

    def load_archived_log(self, date_str: str) -> Optional[str]:
        """Load raw archived log for a specific date."""
        archive_file = self.get_archive_file(date_str)
        if archive_file.exists():
            return archive_file.read_text(encoding="utf-8")
        return None

    def load_day_summary(self, date_str: str) -> Optional[str]:
        """Load summary for a specific date."""
        summary_file = self.get_summary_file(date_str)
        if summary_file.exists():
            return summary_file.read_text(encoding="utf-8")
        return None

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

    def load_recent_memory(self, days: int = 3, use_summaries: bool = True) -> Dict[str, str]:
        """
        Load memory from recent days.

        Hybrid approach (when use_summaries=True):
        - Today: Full raw log
        - Yesterday+: Compressed summaries (if available), otherwise raw log
        """
        today = datetime.now().strftime("%Y-%m-%d")
        memories = {}

        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")

            if date_str == today or not use_summaries:
                # Today: always use raw log
                file_path = self.memory_dir / f"{date_str}.md"
                if file_path.exists():
                    memories[date_str] = file_path.read_text(encoding="utf-8")
            else:
                # Older days: prefer summary, fallback to raw, then archive
                summary = self.load_day_summary(date_str)
                if summary:
                    memories[date_str] = summary
                else:
                    # Check raw log in memory_dir
                    file_path = self.memory_dir / f"{date_str}.md"
                    if file_path.exists():
                        memories[date_str] = file_path.read_text(encoding="utf-8")
                    else:
                        # Check archive (but only load a snippet)
                        archived = self.load_archived_log(date_str)
                        if archived:
                            memories[date_str] = f"[Archived - {len(archived)} chars available via duro_query_archive]"

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
        memory_files = list(self.memory_dir.glob("????-??-??.md"))
        summary_files = list(self.summaries_dir.glob("*-summary.md"))
        archive_files = list(self.archive_dir.glob("*.md"))

        return {
            "active_logs": len(memory_files),
            "summaries": len(summary_files),
            "archived_logs": len(archive_files),
            "core_memory_exists": self.core_memory_file.exists(),
            "today_file_exists": self.get_today_file().exists(),
            "memory_dir": str(self.memory_dir),
            "archive_dir": str(self.archive_dir),
            "summaries_dir": str(self.summaries_dir)
        }

    def list_available_archives(self) -> List[Dict[str, any]]:
        """List all archived logs with metadata."""
        archives = []
        for archive_file in sorted(self.archive_dir.glob("*.md"), reverse=True):
            date_str = archive_file.stem
            size = archive_file.stat().st_size
            archives.append({
                "date": date_str,
                "size_bytes": size,
                "size_kb": round(size / 1024, 1)
            })
        return archives

    def search_archives(self, query: str, limit: int = 5) -> List[Dict[str, any]]:
        """Search through archived logs for specific content."""
        results = []
        query_lower = query.lower()

        for archive_file in sorted(self.archive_dir.glob("*.md"), reverse=True):
            content = archive_file.read_text(encoding="utf-8")
            if query_lower in content.lower():
                # Find matching lines
                matches = []
                for i, line in enumerate(content.split('\n')):
                    if query_lower in line.lower():
                        matches.append({
                            "line_num": i + 1,
                            "text": line[:200]  # Truncate long lines
                        })

                results.append({
                    "date": archive_file.stem,
                    "matches": matches[:5]  # Max 5 matches per file
                })

                if len(results) >= limit:
                    break

        return results
