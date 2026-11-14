"""Task prompts, kept out of the Python so they can be reviewed and versioned.

`prompts/system.md` holds the ROLE / GOAL / RULES / PROCEDURE / OUTPUT FORMAT
sections required by the brief. `prompts/tasks.md` holds the per-tool
instructions, split into `## SECTION` blocks.
"""

from __future__ import annotations

import re
from pathlib import Path

_VERSION_RE = re.compile(r"<!--\s*version:\s*([^\s]+)\s*-->")
_TASK_RE = re.compile(r"^##\s+([A-Z0-9_]+)\s*$", re.MULTILINE)


def load_system_prompt(path: str | Path) -> tuple[str, str]:
    """Return (prompt, version)."""
    text = Path(path).read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    return text.strip(), (match.group(1) if match else "unversioned")


def load_task_prompts(path: str | Path) -> dict[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    matches = list(_TASK_RE.finditer(text))
    tasks: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        tasks[match.group(1)] = text[start:end].strip()
    return tasks
