"""
Duro database migrations.
Each migration is a Python file with:
  - MIGRATION_ID: unique identifier
  - DEPENDS_ON: list of migration IDs this depends on
  - up(): function to apply migration
  - down(): function to rollback (optional)
"""
