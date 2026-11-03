"""Reconstruction fingerprints (hmb.result_cache): numerical code and options invalidate their dependent stages."""

import pytest

from hmb.layout import SessionLayout
from hmb.result_cache import hadm_inference_fingerprint, reconstruction_fingerprints


def reconstruction_fixture(tmp_path):
    repository = tmp_path / "repository"
    baseline = repository / "baseline"
    variable = repository / "metrics/bone_length"
    functions = (
        "configure_runtime",
        "load_camera_calibration",
        "save_metrabs_data",
        "process_videos_with_metrabs",
        "load_metrabs_data",
        "get_framerate",
        "process_videos_with_biomechanics",
        "render_mjx",
    )
    for folder in (baseline, variable):
        (folder / "monocular_demos").mkdir(parents=True)
        (folder / "main.py").write_text("\n".join(f"def {name}():\n    return 1\n" for name in functions))
        (folder / "monocular_demos/utils.py").write_text("joint_names = ['before']\n")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    layout = SessionLayout.create(tmp_path / "session")
    (layout.baseline_dir / "clip_keypoints.npz").write_bytes(b"keypoints")
    config = {"requested_fit_options": {"max_iters": 2, "static_cam": True}}
    return repository, video, layout, config


@pytest.mark.parametrize(
    "stage,relative",
    [
        ("rigid_fit", "baseline"),
        ("variable_fit", "metrics/bone_length"),
    ],
)
@pytest.mark.parametrize("helper", ["configure_runtime", "load_metrabs_data", "get_framerate"])
def test_fit_helper_mutation_invalidates_dependent_stage(tmp_path, stage, relative, helper):
    repository, video, layout, config = reconstruction_fixture(tmp_path)
    before = reconstruction_fingerprints(config, video, layout, repository)
    source = repository / relative / "main.py"
    source.write_text(source.read_text().replace(f"def {helper}():\n    return 1", f"def {helper}():\n    return 2"))
    after = reconstruction_fingerprints(config, video, layout, repository)
    assert after[stage] != before[stage]
    assert after["mesh"] == before["mesh"]
    if stage == "variable_fit":
        assert after["detection"] == before["detection"]
        assert after["rigid_fit"] == before["rigid_fit"]


def test_variable_joint_mapping_mutation_only_invalidates_variable_fit(tmp_path):
    repository, video, layout, config = reconstruction_fixture(tmp_path)
    before = reconstruction_fingerprints(config, video, layout, repository)
    utility = repository / "metrics/bone_length/monocular_demos/utils.py"
    utility.write_text("joint_names = ['after']\n")
    after = reconstruction_fingerprints(config, video, layout, repository)
    assert after["variable_fit"] != before["variable_fit"]
    assert all(after[stage] == before[stage] for stage in ("detection", "rigid_fit", "mesh"))


def test_rendering_helper_changes_do_not_invalidate_numerical_fits(tmp_path):
    repository, video, layout, config = reconstruction_fixture(tmp_path)
    before = reconstruction_fingerprints(config, video, layout, repository)
    for relative in ("baseline", "metrics/bone_length"):
        source = repository / relative / "main.py"
        source.write_text(
            source.read_text().replace("def render_mjx():\n    return 1", "def render_mjx():\n    return 2")
        )
    assert reconstruction_fingerprints(config, video, layout, repository) == before


# ----------------------------------------------------------------------------- engine sessions
ENGINE_MODULES = ("detect_engine", "fit_engine", "fit_export", "mesh_engine", "hadm_engine", "video")


def engine_fixture(tmp_path):
    repository, video, layout, config = reconstruction_fixture(tmp_path)
    for name in ENGINE_MODULES:
        path = repository / "hmb" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VERSION = '{name}-1'\n")
    plan = layout.input_dir / "frames.json"
    plan.write_text('{"indices": [0, 1, 2]}')
    session = {
        "frame_plan": str(plan),
        "requested_fit_options": {"max_iters": 2, "static_cam": True, "sample_fps": None,
                                  "max_side": None, "person": "first"},
        "engine_options": {
            "detect": {"person": "first", "batch_size": 8},
            "mesh": {"camera": "static", "precision": "tf32"},
            "hadm": {"precision": "fp16"},
            "fit": {"max_iters": 2, "batch": 1, "modes": ["rigid", "variable"], "jax_cache_dir": "/cache/a"},
        },
    }
    return repository, video, layout, session


def changed(before, after):
    return {stage for stage in before if before[stage] != after[stage]}


@pytest.mark.parametrize(
    "module,stages",
    [
        ("detect_engine", {"detection", "rigid_fit", "variable_fit"}),
        ("fit_engine", {"rigid_fit", "variable_fit"}),
        ("fit_export", {"rigid_fit", "variable_fit"}),
        ("mesh_engine", {"mesh"}),
        ("video", {"detection", "rigid_fit", "variable_fit", "mesh"}),
        ("hadm_engine", set()),
    ],
)
def test_engine_module_changes_invalidate_their_stages(tmp_path, module, stages):
    repository, video, layout, config = engine_fixture(tmp_path)
    before = reconstruction_fingerprints(config, video, layout, repository)
    hadm_before = hadm_inference_fingerprint(config, video, repository)
    (repository / "hmb" / f"{module}.py").write_text("VERSION = 'edited'\n")
    assert changed(before, reconstruction_fingerprints(config, video, layout, repository)) == stages
    hadm_changed = hadm_inference_fingerprint(config, video, repository) != hadm_before
    assert hadm_changed == (module in ("hadm_engine", "video"))


def test_frame_plan_and_sampling_invalidate_every_stage(tmp_path):
    repository, video, layout, session = engine_fixture(tmp_path)
    before = reconstruction_fingerprints(session, video, layout, repository)
    hadm = hadm_inference_fingerprint(session, video, repository)
    (layout.input_dir / "frames.json").write_text('{"indices": [0, 2]}')
    assert changed(before, reconstruction_fingerprints(session, video, layout, repository)) == set(before)
    assert hadm_inference_fingerprint(session, video, repository) != hadm
    (layout.input_dir / "frames.json").write_text('{"indices": [0, 1, 2]}')
    assert reconstruction_fingerprints(session, video, layout, repository) == before
    session["requested_fit_options"]["sample_fps"] = 24.0
    assert changed(before, reconstruction_fingerprints(session, video, layout, repository)) == set(before)


@pytest.mark.parametrize(
    "engine,key,value,stages",
    [
        ("fit", "max_iters", 3, {"rigid_fit", "variable_fit"}),
        ("fit", "jax_cache_dir", "/cache/b", set()),  # execution-only
        ("fit", "modes", ["rigid"], set()),  # which fits run, not how
        ("detect", "person", "track", {"detection", "rigid_fit", "variable_fit"}),
        ("mesh", "precision", "fp32", {"mesh"}),
        ("hadm", "precision", "fp32", set()),
    ],
)
def test_engine_options_invalidate_their_stages(tmp_path, engine, key, value, stages):
    repository, video, layout, session = engine_fixture(tmp_path)
    before = reconstruction_fingerprints(session, video, layout, repository)
    hadm = hadm_inference_fingerprint(session, video, repository)
    session["engine_options"][engine][key] = value
    assert changed(before, reconstruction_fingerprints(session, video, layout, repository)) == stages
    assert (hadm_inference_fingerprint(session, video, repository) != hadm) == (engine == "hadm")
