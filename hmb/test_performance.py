"""Tests of the run-time profile (hmb.performance) and of the content identity used by the caches."""
import json
import os

import pytest

from hmb.performance import RunTimings
from hmb.result_cache import file_identity


def test_content_identity_survives_touch_but_detects_same_size_edits(tmp_path):
    path = tmp_path / "input.npz"
    path.write_bytes(b"abcd")
    identity = file_identity(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert file_identity(path) == identity
    path.write_bytes(b"efgh")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert file_identity(path) != identity
    assert file_identity(tmp_path / "missing.npz")["missing"] is True


def test_profile_records_failures_and_distinguishes_cached_work(tmp_path):
    profile = RunTimings(tmp_path, "test")
    with profile.stage("reused", cache_hit=True):
        pass
    with pytest.raises(RuntimeError):
        with profile.stage("failed"):
            raise RuntimeError("failed compute")
    profile.finish(False)
    data = json.loads((tmp_path / "timings.json").read_text())
    assert not data["complete"]
    assert data["stages"][0]["cache_hit"]
    assert data["stages"][1]["status"] == "error"
    assert data["wall_seconds"] >= sum(stage["seconds"] for stage in data["stages"])
