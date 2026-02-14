"""
Centralized time utilities for Duro.

All datetime operations should use these helpers to ensure:
- Consistent timezone handling (always UTC, always aware)
- Python 3.13+ compatibility (no deprecated utcnow())
- Consistent ISO format with Z suffix
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """
    Return current UTC time as timezone-aware datetime.

    Use this instead of datetime.utcnow() which is deprecated in Python 3.12+
    and will be removed in a future version.
    """
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """
    Return current UTC time as ISO string with Z suffix.

    Format: 2026-02-14T12:34:56.789012Z
    """
    return utc_now().isoformat().replace("+00:00", "Z")


def parse_iso_datetime(iso_string: str) -> datetime:
    """
    Parse ISO datetime string to timezone-aware datetime.

    Handles both Z suffix and +00:00 offset.
    Always returns timezone-aware datetime in UTC.
    """
    if iso_string.endswith("Z"):
        iso_string = iso_string[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso_string)
    if dt.tzinfo is None:
        # Assume UTC for naive datetimes
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def days_since(iso_string: str) -> int:
    """
    Calculate days since a given ISO datetime string.

    Returns integer number of days (floor).
    """
    created = parse_iso_datetime(iso_string)
    now = utc_now()
    return (now - created).days


def normalize_iso_z(iso_string: str) -> str:
    """
    Normalize any ISO datetime string to consistent Z-suffix format.

    Handles:
    - 2026-02-14T00:00:00Z → unchanged
    - 2026-02-14T00:00:00+00:00 → 2026-02-14T00:00:00Z
    - 2026-02-14T00:00:00 (naive) → 2026-02-14T00:00:00Z (assumed UTC)

    Use this to ensure DB text comparisons work correctly.
    """
    dt = parse_iso_datetime(iso_string)
    return dt.isoformat().replace("+00:00", "Z")
