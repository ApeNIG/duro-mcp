"""
Duro database migrations.
Each migration is a Python file with:
  - MIGRATION_ID: unique identifier
  - DEPENDS_ON: list of migration IDs this depends on
  - up(): function to apply migration
  - down(): function to rollback (optional)
"""

# Export runner functions for stable imports
from .runner import (
    get_applied_migrations,
    get_pending_migrations,
    run_migration,
    run_all_pending,
    get_status,
)
