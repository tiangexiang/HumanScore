"""Tests of the fingerprints in hmb.result_cache: what invalidates reconstruction products and scores."""
import json

import pytest

from hmb import cli
from hmb.layout import SessionLayout, write_manifest
from hmb.result_cache import (
    load_cached_scores,
    reconstruction_fingerprints,
    save_cached_scores,
    saved_score_request,
    score_code_files,
    score_fingerprint,
)
from hmb.test_batch import FakeEngines
from hmb.test_workflow import cli_args


def test_bone_geometry_changes_only_variable_fit_fingerprint(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    layout = SessionLayout.create(tmp_path / "session")
    repo = tmp_path / "repo"
    for folder in (repo / "baseline", repo / "metrics/bone_length"):
        geometry = folder / "monocular_demos/biomechanics_mjx/data/humanoid"
        geometry.mkdir(parents=True)
        (geometry / "humanoid_torque.xml").write_text('<mujoco model="original"/>')

    before = reconstruction_fingerprints({}, video, layout, repo)
    bone_xml = repo / "metrics/bone_length/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
    bone_xml.write_text('<mujoco model="modified"/>')
    after = reconstruction_fingerprints({}, video, layout, repo)

    assert after["variable_fit"] != before["variable_fit"]
    for stage in ("detection", "rigid_fit", "mesh"):
        assert after[stage] == before[stage]


def test_same_path_savedmodel_changes_regenerate_detection_and_both_fits(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    model = tmp_path / "model"
    (model / "variables").mkdir(parents=True)
    (model / "saved_model.pb").write_bytes(b"graph")
    weights = model / "variables/variables.data-00000-of-00001"
    weights.write_bytes(b"weights1")
    layout = SessionLayout.create(tmp_path / "session")
    config = {"requested_fit_options": {"metrabs_model": str(model), "max_iters": 2}}
    before = reconstruction_fingerprints(config, video, layout, tmp_path / "repo")
    weights.write_bytes(b"weights2")
    after = reconstruction_fingerprints(config, video, layout, tmp_path / "repo")
    assert {stage for stage in before if before[stage] != after[stage]} == {"detection", "rigid_fit", "variable_fit"}


def test_hadm_configuration_changes_invalidate_prediction_cache(tmp_path, monkeypatch):
    engines = FakeEngines(tmp_path, monkeypatch)
    hadm = tmp_path / "hadm"
    config = hadm / "projects/ViTDet/configs/model.py"
    config.parent.mkdir(parents=True)
    config.write_text("image_size = 1024")
    session = cli.prepare_session(cli_args(tmp_path, "--hadm-root", str(hadm)))
    prediction = session / "cache/multi_limbs/walk/hadm_results/output.json"
    prediction.write_text('{"kept": true}')
    engines.clear()
    cli.prepare_session(cli_args(tmp_path, "--hadm-root", str(hadm)))
    assert prediction.read_text() == '{"kept": true}' and engines.runs() == []
    config.write_text("image_size = 1536")
    cli.prepare_session(cli_args(tmp_path, "--hadm-root", str(hadm)))
    assert engines.engines() == ["hadm"] and prediction.read_text() == "{}"  # detection and fit are kept


def scored_session(tmp_path, monkeypatch, *extra):
    FakeEngines(tmp_path, monkeypatch)
    session = cli.prepare_session(cli_args(tmp_path, *extra))
    layout = SessionLayout.create(session)
    return layout, json.loads(layout.manifest_path.read_text())


def test_score_cache_tracks_inputs_code_and_report(tmp_path, monkeypatch):
    layout, manifest = scored_session(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    code = repo / "hmb/scoring/joint_range.py"
    code.parent.mkdir(parents=True)
    code.write_text("version = 1")
    fingerprint = score_fingerprint(manifest, layout, repo)
    report = layout.session_dir / "scores.json"
    report.write_text('{"scores": {"bone_length": 90}}')
    save_cached_scores(fingerprint, layout)
    assert load_cached_scores(manifest, layout, repo) == {"scores": {"bone_length": 90}}
    assert saved_score_request(layout) == {"paper_scores": False}
    assert load_cached_scores(manifest, layout, repo, paper_scores=True) is None   # paper scores were not computed
    assert score_fingerprint(manifest, layout, repo, paper_scores=True) != fingerprint
    code.write_text("version = 2")
    assert score_fingerprint(manifest, layout, repo) != fingerprint and load_cached_scores(manifest, layout, repo) is None
    code.write_text("version = 1")
    (layout.baseline_dir / "walk_keypoints.npz").write_bytes(b"new detection")
    assert score_fingerprint(manifest, layout, repo) != fingerprint
    (layout.baseline_dir / "walk_keypoints.npz").write_text("keypoints")
    assert load_cached_scores(manifest, layout, repo) is not None
    report.write_text('{"scores": {"bone_length": 1}}')  # a tampered report is never reused
    assert load_cached_scores(manifest, layout, repo) is None


@pytest.mark.parametrize("relative", ["metrics/kinematic_extreme/main.py", "metrics/kinematic_extreme/definition.py",
                                      "metrics/kinematic_extreme/utils.py", "metrics/motion_smoothness/main.py",
                                      "metrics/motion_smoothness/definition.py", "metrics/motion_smoothness/utils.py",
                                      "hmb/scoring/score_session.py", "hmb/video.py"])
def test_score_fingerprint_covers_the_paper_kinetic_code_of_the_default_scores(tmp_path, monkeypatch, relative):
    """Kinematic extremes and motion smoothness run the paper code (metrics/) in the default scorer."""
    layout, manifest = scored_session(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("A = 1\n")
    before = score_fingerprint(manifest, layout, repo)
    path.write_text("A = 2\n")
    assert score_fingerprint(manifest, layout, repo) != before
    path.write_text("A = 1\n")
    test_file = path.parent / "test_extra.py"
    test_file.write_text("tests are not dependencies")
    assert score_fingerprint(manifest, layout, repo) == before


@pytest.mark.parametrize("relative", ["metrics/bone_length/bone_length.py", "metrics/range_of_motion/joints_violaton.py",
                                      "metrics/self_collision_bvh/main.py", "metrics/multi-limbs/eval_extra_limbs.py"])
def test_paper_only_code_invalidates_only_the_paper_scores(tmp_path, monkeypatch, relative):
    layout, manifest = scored_session(tmp_path, monkeypatch, "--paper-scores")
    repo = tmp_path / "repo"
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("A = 1\n")
    default, paper = score_fingerprint(manifest, layout, repo), score_fingerprint(manifest, layout, repo, True)
    assert path in score_code_files(repo, True) and path not in score_code_files(repo)
    path.write_text("A = 2\n")
    assert score_fingerprint(manifest, layout, repo) == default
    assert score_fingerprint(manifest, layout, repo, True) != paper


def test_variable_fit_is_an_input_of_the_paper_scores_only(tmp_path, monkeypatch):
    layout, manifest = scored_session(tmp_path, monkeypatch, "--paper-scores")
    default, paper = (score_fingerprint(manifest, layout, cli.REPO_ROOT, flag) for flag in (False, True))
    (layout.bone_fit_dir / "walk_bone_length.npz").write_bytes(b"refit")
    assert score_fingerprint(manifest, layout, cli.REPO_ROOT) == default
    assert score_fingerprint(manifest, layout, cli.REPO_ROOT, True) != paper


def test_kinetic_grid_export_is_a_score_input(tmp_path, monkeypatch):
    layout, manifest = scored_session(tmp_path, monkeypatch)
    before = score_fingerprint(manifest, layout, cli.REPO_ROOT)
    (layout.baseline_dir / "walk_opensim_skeleton_grid.mot").write_text("uniform grid export")
    assert score_fingerprint(manifest, layout, cli.REPO_ROOT) != before


def test_saved_request_without_a_cache_record(tmp_path):
    layout = SessionLayout.create(tmp_path / "session")
    assert saved_score_request(layout) is None
    write_manifest(layout.cache_dir / "score_cache.json", {"unexpected": True})
    assert saved_score_request(layout) is None
