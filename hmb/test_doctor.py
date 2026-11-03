"""``hmb.doctor`` decisions on temporary installation trees (no models, no downloads)."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from hmb import doctor
from hmb.gvhmr_prepare import CHECKPOINTS
from hmb.hadm_prepare import DEFAULT_CHECKPOINT, DEFAULT_CONFIG

REPO = Path(__file__).resolve().parents[1]


class SAM3SetupError(RuntimeError):
    pass


class SAM3CheckpointError(RuntimeError):
    pass


def touch(path: Path, size: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.truncate(size)  # sparse: exact sizes without writing gigabytes
    return path


def fresh_tree(tmp_path):
    """A fresh clone after ``check_sources.sh --init`` plus the two manual downloads."""
    repo, hadm, gvhmr = tmp_path / "repo", tmp_path / "repo/hadm", tmp_path / "repo/GVHMR"
    touch(hadm / DEFAULT_CONFIG)
    touch(hadm / DEFAULT_CHECKPOINT)
    touch(gvhmr / "hmr4d/configs/demo.yaml")
    touch(repo / "outputs/assets/smplx/SMPLX_NEUTRAL.npz")
    touch(repo / "baseline/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml")
    return repo, hadm, gvhmr


def run(tmp_path, repo, hadm, gvhmr, environ=None, **kwargs):
    environ = {"TFHUB_CACHE_DIR": str(tmp_path / "tfhub")} if environ is None else environ
    kwargs.setdefault("multi_person", False)
    return doctor.diagnose(repo, hadm, gvhmr, repo / "outputs/assets/smplx", environ=environ, modules=(),
                           commands=(), which=lambda name: None, **kwargs)


def by_name(report):
    return {c["name"]: c for c in report["checks"]}


def test_fresh_install_is_ready_with_first_use_downloads(tmp_path):
    repo, hadm, gvhmr = fresh_tree(tmp_path)
    report = run(tmp_path, repo, hadm, gvhmr)
    checks = by_name(report)
    assert report["ready"] is True
    for name in ("GVHMR weights", "HMR2 weights", "ViTPose weights", "YOLO person detector (GVHMR)", "MeTRAbs model"):
        assert checks[name]["status"] == "first-use" and checks[name]["ok"], name
    assert sorted(report["first_use"]) == sorted(["GVHMR weights", "HMR2 weights", "ViTPose weights",
                                                  "YOLO person detector (GVHMR)", "MeTRAbs model"])
    assert str(repo / doctor.GVHMR_ASSET_DIR / "gvhmr/gvhmr_siga24_release.ckpt") in checks["GVHMR weights"]["detail"]
    assert str(tmp_path / "tfhub") in checks["MeTRAbs model"]["detail"]
    text = doctor.format_report(report)
    assert "AUTO" in text and "Ready." in text and "MISSING" not in text
    json.dumps(report)


def test_manual_assets_are_required(tmp_path):
    repo, hadm, gvhmr = fresh_tree(tmp_path)
    (hadm / DEFAULT_CHECKPOINT).unlink()
    (repo / "outputs/assets/smplx/SMPLX_NEUTRAL.npz").unlink()
    report = run(tmp_path, repo, hadm, gvhmr)
    checks = by_name(report)
    assert report["ready"] is False
    assert checks["HADM-G weights"]["status"] == "missing"
    assert checks["SMPL-X neutral model"]["status"] == "missing"
    assert "smpl-x.is.tue.mpg.de" in checks["SMPL-X neutral model"]["detail"]
    assert "Not ready: 2 required item(s) missing" in doctor.format_report(report)


def test_missing_sources_and_commands_are_required(tmp_path):
    repo, hadm, gvhmr = fresh_tree(tmp_path)
    (gvhmr / "hmr4d/configs/demo.yaml").unlink()
    report = doctor.diagnose(repo, hadm, gvhmr, repo / "outputs/assets/smplx", environ={}, modules=(),
                             commands=("definitely-not-a-command",), multi_person=False, which=lambda name: None)
    checks = by_name(report)
    assert not report["ready"]
    assert checks["GVHMR source"]["status"] == "missing" and "check_sources.sh --init" in checks["GVHMR source"]["detail"]
    assert checks["definitely-not-a-command"]["status"] == "missing"


@pytest.mark.parametrize("where", ["asset_dir", "gvhmr_inputs"])
def test_gvhmr_checkpoints_found_in_either_location(tmp_path, where):
    repo, hadm, gvhmr = fresh_tree(tmp_path)
    root = repo / doctor.GVHMR_ASSET_DIR if where == "asset_dir" else gvhmr / "inputs/checkpoints"
    for relative, (size, _) in CHECKPOINTS.items():
        touch(root / relative, size)
    checks = by_name(run(tmp_path, repo, hadm, gvhmr))
    for relative in CHECKPOINTS:
        check = checks[doctor.GVHMR_CHECKPOINT_NAMES[relative]]
        assert check["status"] == "ok" and check["detail"] == str(root / relative)


def test_incomplete_gvhmr_checkpoint_is_redownloaded(tmp_path):
    repo, hadm, gvhmr = fresh_tree(tmp_path)
    touch(repo / doctor.GVHMR_ASSET_DIR / "gvhmr/gvhmr_siga24_release.ckpt", 10)
    check = by_name(run(tmp_path, repo, hadm, gvhmr))["GVHMR weights"]
    assert check["status"] == "first-use" and "incomplete file" in check["detail"]


def test_metrabs_local_model_tfhub_cache_and_bad_path(tmp_path):
    model = tmp_path / "metrabs"
    assert doctor.metrabs_check({"HMB_METRABS_MODEL": str(model)})["status"] == "missing"
    touch(model / "saved_model.pb")
    assert doctor.metrabs_check({"HMB_METRABS_MODEL": str(model)})["status"] == "ok"
    cache = tmp_path / "tfhub"
    environ = {"TFHUB_CACHE_DIR": str(cache)}
    assert doctor.metrabs_check(environ)["status"] == "first-use"
    digest = hashlib.sha1(doctor.METRABS_DEFAULT_HANDLE.encode("utf8")).hexdigest()
    touch(cache / digest / "saved_model.pb")
    check = doctor.metrabs_check(environ)
    assert check["status"] == "ok" and str(cache / digest) in check["detail"]
    # a hub handle is downloaded on first use like the default one
    assert doctor.metrabs_check({"HMB_METRABS_MODEL": "https://example.org/m.tar.gz",
                                 "TFHUB_CACHE_DIR": str(cache)})["status"] == "first-use"


def test_metrabs_default_handle_matches_the_loader():
    source = (REPO / "baseline/monocular_demos/utils.py").read_text()
    assert f"'{doctor.METRABS_DEFAULT_HANDLE}'" in source


def test_yolo_seg_weights(tmp_path):
    repo = tmp_path / "repo"
    check = doctor.yolo_seg_check(repo, {})
    assert check["status"] == "first-use" and not check["required"]
    touch(repo / doctor.YOLO_SEG_DIR / doctor.YOLO_SEG_DEFAULT)
    assert doctor.yolo_seg_check(repo, {})["status"] == "ok"
    assert doctor.yolo_seg_check(repo, {doctor.YOLO_SEG_ENV: str(tmp_path / "nope/x-seg.pt")})["status"] == "missing"
    assert doctor.yolo_seg_check(repo, {doctor.YOLO_SEG_ENV: "yolov8l-seg.pt"})["status"] == "first-use"


def test_yolo_constants_match_backend():
    from hmb.tracking import yolo_backend

    assert doctor.YOLO_SEG_ENV == yolo_backend.ENV_WEIGHTS
    assert doctor.YOLO_SEG_DEFAULT == yolo_backend.DEFAULT_WEIGHTS
    assert yolo_backend.ASSET_DIR == REPO / doctor.YOLO_SEG_DIR


def test_optional_sam3_checks_never_change_ready(tmp_path):
    repo, hadm, gvhmr = fresh_tree(tmp_path)

    def failing(exc):
        def check():
            raise exc
        return check

    cases = {
        "setup": (failing(SAM3SetupError("The sam3.1 tracker needs SAM 3.1\n\nlong help")), "missing", "not set up"),
        "checkpoint": (failing(SAM3CheckpointError("SAM 3.1 checkpoint not available: gated")), "missing",
                       "HMB_SAM3_CHECKPOINT"),
        "no_cuda": (lambda: {"python": "/x/python", "torch": "2.10", "cuda": False, "checkpoint": "/c.pt"}, "missing",
                    "CUDA"),
        "ok": (lambda: {"python": "/x/python", "torch": "2.10", "cuda": True, "checkpoint": "/c.pt"}, "ok", "/c.pt"),
        "broken": (failing(ImportError("numpy")), "missing", "check failed"),
    }
    for key, (check, status, text) in cases.items():
        report = run(tmp_path, repo, hadm, gvhmr, multi_person=True, sam3=check)
        sam3 = next(c for c in report["checks"] if c["name"].startswith("SAM 3.1"))
        assert report["ready"] is True, key
        assert sam3["status"] == status and not sam3["required"] and text in sam3["detail"], key
        if status == "missing":
            assert sam3["name"] in report["optional_missing"]
            assert "OPTIONAL" in doctor.format_report(report)


def test_offline_hub_restores_environment(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with doctor._offline_hub():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ


def test_cli_doctor_exit_status(tmp_path, monkeypatch, capsys):
    from hmb import cli

    repo, hadm, gvhmr = fresh_tree(tmp_path)
    (hadm / DEFAULT_CHECKPOINT).unlink()
    monkeypatch.setattr("sys.argv", ["hmb.cli", "doctor", "--hadm-root", str(hadm), "--gvhmr-root", str(gvhmr),
                                     "--smplx-model-dir", str(repo / "outputs/assets/smplx"), "--skip-optional",
                                     "--json"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert by_name(report)["HADM-G weights"]["status"] == "missing"
    assert not any(not c["required"] for c in report["checks"])


# ----------------------------------------------------------------------------- install/check_sources.sh --assets
def test_check_sources_lists_the_gvhmr_checkpoint_sizes():
    script = (REPO / "install/check_sources.sh").read_text()
    for relative, (size, _) in CHECKPOINTS.items():
        assert f'"{relative} {size}"' in script, relative


def run_check_sources(tmp_path, environ=None):
    import shutil
    import subprocess

    repo = tmp_path / "repo"
    (repo / "install").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "install/check_sources.sh", repo / "install/check_sources.sh")
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HMB_", "TFHUB_"))}
    env.update(environ or {})
    proc = subprocess.run(["bash", str(repo / "install/check_sources.sh"), "--assets"], capture_output=True,
                          text=True, env=env)
    return repo, proc.returncode, proc.stdout + proc.stderr


def test_check_sources_assets_use_the_pipeline_locations(tmp_path):
    repo, _, out = run_check_sources(tmp_path)
    assert "MISSING source: GVHMR" in out  # no submodules in the temporary clone
    for relative in CHECKPOINTS:
        assert f"AUTO asset: {relative}" in out
    assert "MISSING asset:" in out and "HADM-G_0249999.pth" in out
    assert "MISSING model: SMPLX_NEUTRAL.npz" in out
    assert "AUTO model: MeTRAbs" in out
    assert "AUTO optional:" in out and "yolov8x-seg.pt" in out

    # GVHMR checkpoints in outputs/assets (or linked into GVHMR/inputs) and the manual downloads
    for relative, (size, _) in CHECKPOINTS.items():
        asset = touch(repo / doctor.GVHMR_ASSET_DIR / relative, size)
        link = repo / "GVHMR/inputs/checkpoints" / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(asset)
    touch(repo / "metrics/multi-limbs/HADM/pretrained_models/HADM-G_0249999.pth")
    touch(repo / "outputs/assets/smplx/SMPLX_NEUTRAL.npz")
    _, _, out = run_check_sources(tmp_path)
    for relative in CHECKPOINTS:
        assert f"OK asset: {repo / 'GVHMR/inputs/checkpoints' / relative}" in out
    assert "MISSING asset" not in out and "MISSING model" not in out

    # an explicit but broken MeTRAbs path is an error; a complete SavedModel is fine
    _, _, out = run_check_sources(tmp_path, {"HMB_METRABS_MODEL": str(tmp_path / "metrabs")})
    assert "MISSING model: HMB_METRABS_MODEL=" in out
    touch(tmp_path / "metrabs/saved_model.pb")
    _, _, out = run_check_sources(tmp_path, {"HMB_METRABS_MODEL": str(tmp_path / "metrabs")})
    assert "OK model: MeTRAbs" in out
