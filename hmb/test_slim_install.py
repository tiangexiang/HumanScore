"""The lightweight installation (install/install_slim.sh): lock consistency and the code paths it relies on."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

from hmb import result_cache

REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "requirements"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(path: Path) -> dict[str, str]:
    """``name==version`` lines of a requirements file (``-r`` includes followed)."""
    pins = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("-r "):
            pins.update(_pins(path.parent / line[3:].strip()))
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", line)
        if match:
            pins[_norm(match.group(1))] = match.group(2)
    return pins


def _names(path: Path) -> set[str]:
    names = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            names.add(_norm(re.split(r"[\s=<>;\[]", line, maxsplit=1)[0]))
    return names


def test_lock_leaves_out_every_excluded_distribution():
    locked = _pins(REQUIREMENTS / "lock-slim.txt")
    excluded = _names(REQUIREMENTS / "slim-excludes.txt")
    assert excluded and not excluded & set(locked)
    # The SAM 3.1 extra adds triton back, and nothing else that is excluded.
    assert _names(REQUIREMENTS / "lock-slim-sam3.txt") & excluded == {"triton"}


def test_lock_keeps_every_pin_of_the_inputs():
    locked = _pins(REQUIREMENTS / "lock-slim.txt")
    excluded = _names(REQUIREMENTS / "slim-excludes.txt")
    overrides = _pins(REQUIREMENTS / "slim-overrides.txt")
    for name, version in _pins(REQUIREMENTS / "slim.in").items():
        if name in excluded:
            continue
        assert locked.get(name) == overrides.get(name, version), name
    # every locked distribution is hash-pinned (install_slim.sh uses --require-hashes)
    text = (REQUIREMENTS / "lock-slim.txt").read_text()
    assert text.count("==") <= text.count("--hash=sha256:")
    assert locked["torch"] == "2.8.0+cu128" and locked["numpy"] == "2.1.3"


def test_opencv_version_is_recorded_for_either_distribution(monkeypatch):
    installed = {"opencv-python-headless": "4.12.0.88"}

    def version(name):
        if name in installed:
            return installed[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(result_cache.importlib.metadata, "version", version)
    assert result_cache._runtime_versions(("opencv-python", "keras"))["opencv-python"] == "4.12.0.88"
    installed["opencv-python"] = "4.11.0.86"   # the regular distribution wins when both are installed
    versions = result_cache._runtime_versions(("opencv-python", "keras"))
    assert versions["opencv-python"] == "4.11.0.86" and versions["keras"] is None


@pytest.mark.parametrize("copy", ["baseline", "metrics/bone_length"])
@pytest.mark.parametrize("keras_installed", [False, True])
def test_tensorflow_hub_uses_tf_keras_without_keras_3(copy, keras_installed, monkeypatch):
    spec = importlib.util.spec_from_file_location(f"_utils_{copy.replace('/', '_')}",
                                                  REPO / copy / "monocular_demos/utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    fake_hub = types.ModuleType("tensorflow_hub")
    monkeypatch.setitem(sys.modules, "tensorflow_hub", fake_hub)
    monkeypatch.delenv("TF_USE_LEGACY_KERAS", raising=False)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: (object() if keras_installed else None) if name == "keras"
                        else real_find_spec(name, *a))
    assert utils._import_tensorflow_hub() is fake_hub
    import os
    assert os.environ.get("TF_USE_LEGACY_KERAS") == (None if keras_installed else "1")
