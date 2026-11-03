"""Wall-clock timings of one HumanScore run, written to ``<session>/timings.json``.

``hmb.cli`` wraps each step of a run (frame plan, reconstruction engines, scoring)
in :meth:`RunTimings.stage`.  Every finished stage is appended to the file right
away, so an interrupted run still shows how far it got and where the time went.
A stage served from cache is marked ``cache_hit`` so reused work is not mistaken
for computation.

``timings.json``::

    {"command": "run-all", "started_at": "<UTC ISO time>", "complete": true,
     "wall_seconds": 163.2,
     "stages": [{"stage": "frame_plan", "kind": "stage", "cache_hit": false,
                 "status": "ok", "seconds": 0.4}, ...]}
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import time

from .layout import write_manifest


class RunTimings:
    """Collects stage timings of one command and keeps ``timings.json`` up to date."""

    def __init__(self, session_dir: Path, command: str):
        self.path = session_dir / "timings.json"
        self.started = time.perf_counter()
        self.payload = {
            "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stages": [],
            "complete": False,
        }

    @contextmanager
    def stage(self, name: str, *, cache_hit: bool = False, group: bool = False):
        """Time the enclosed block as stage ``name``; yields its entry so callers can add fields.

        An exception marks the stage ``"error"`` (with its message) and propagates.
        ``group`` marks a stage that encloses other recorded stages (do not add its seconds to theirs).
        """
        started = time.perf_counter()
        entry = {"stage": name, "cache_hit": cache_hit, "status": "ok", "kind": "group" if group else "stage"}
        try:
            yield entry
        except BaseException as exc:
            entry.update(status="error", error=str(exc))
            raise
        finally:
            entry["seconds"] = time.perf_counter() - started
            self.payload["stages"].append(entry)
            self.save()

    def save(self):
        """Write the current state (with the elapsed wall time) to ``timings.json``."""
        self.payload["wall_seconds"] = time.perf_counter() - self.started
        write_manifest(self.path, self.payload)

    def finish(self, success: bool):
        """Mark the run complete (``success``) or failed, save, and return the payload."""
        self.payload["complete"] = success
        self.save()
        return self.payload
