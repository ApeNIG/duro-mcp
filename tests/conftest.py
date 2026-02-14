"""
Pytest configuration for Duro MCP tests.

Ensures the parent directory is in the path for imports.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
