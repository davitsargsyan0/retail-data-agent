"""User-preference store — the prototype slice of the user-level learning loop.

Architecture §5.4: each manager has a presentation preference (Manager A wants
tables, Manager B wants bullet points). **Production:** a Firestore document per
``user_id``. **Prototype:** a single ``data/prefs.json`` keyed ``user_id → format``,
following the same ``data/`` convention as the saved-reports store.

Only the report *format* is modelled here (``table`` | ``bullets``, default
``table``); the shape generalises to richer profiles (verbosity, favourite
metrics) without touching callers. Reads default gracefully, and a corrupt or
missing file never blocks a turn — the agent just falls back to ``table``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

Format = str  # one of VALID_FORMATS; kept a plain str for the state's dict[str, str]

VALID_FORMATS: Final[tuple[str, ...]] = ("table", "bullets")
DEFAULT_FORMAT: Final[str] = "table"


def parse_format(text: str) -> str | None:
    """Extract a requested format from a preference command, or ``None``.

    Deterministic: ``bullets`` wins if the user mentions bullets, else ``table``
    if they mention tables. Used by the ``set_preference`` node to turn
    "remember I prefer bullets" into a stored value.
    """
    lowered = text.lower()
    if "bullet" in lowered:
        return "bullets"
    if "table" in lowered:
        return "table"
    return None


class PrefsStore:
    """File-backed ``user_id → format`` store (``data/prefs.json``)."""

    def __init__(self, root: Path) -> None:
        self._path = root / "prefs.json"

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}

    def get_format(self, user_id: str) -> str:
        """The user's preferred format, or the default if unset/invalid."""
        fmt = self._load().get(user_id, DEFAULT_FORMAT)
        return fmt if fmt in VALID_FORMATS else DEFAULT_FORMAT

    def set_format(self, user_id: str, fmt: str) -> None:
        """Persist the user's preferred format. Rejects unknown formats."""
        if fmt not in VALID_FORMATS:
            raise ValueError(f"unknown format {fmt!r}; expected one of {VALID_FORMATS}")
        data = self._load()
        data[user_id] = fmt
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
