"""The six scores of the original HumanScore paper implementation (``metrics/``) for a session.

HumanScore uses this module in two ways:

* kinematic extremes and motion smoothness are always scored with the paper code
  (``metrics/kinematic_extreme``, ``metrics/motion_smoothness``);
* with ``--paper-scores`` all six paper metrics are computed on the same
  reconstruction and reported next to the HumanScore scores (``paper_scores``).

It calls the paper metric functions directly with the session's artifact paths and
the paper's settings, and writes their usual JSON outputs under
``<S>/paper_metrics/<metric>/``:

* extra limbs: ``metrics/multi-limbs/eval_extra_limbs.py`` on the HADM detections;
* bone length: ``metrics/bone_length`` on the variable-scale skeleton fit
  (``<S>/bone_fit/<stem>_bone_length.npz``, made only when paper scores are requested);
* joint range: ``metrics/range_of_motion`` on ``<S>/baseline/<stem>_opensim_skeleton.mot``;
* self-collision: ``metrics/self_collision_bvh`` on the GVHMR mesh (needs torch-mesh-isect);
* kinematic extremes and motion smoothness: ``metrics/kinematic_extreme`` and
  ``metrics/motion_smoothness`` on ``<S>/baseline/<stem>_opensim_skeleton_grid.mot``, the
  rigid fit exported on a uniform 24 Hz grid (finite differences need a uniform time
  step; the analysed frames of a decimated video are not uniformly spaced).

Run it on its own (writes ``<S>/paper_scores.json``)::

    python -m hmb.scoring.paper_scores --session-dir S [--metrics joint_range ...] [--no-cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
#: MuJoCo humanoid of the skeleton fit (the kinematic-extremes code needs its body tree).
DEFAULT_MODEL = REPO_ROOT / "baseline/monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
METRICS = ("extra_limbs", "bone_length", "joint_range", "self_collision",
           "kinematic_extremes", "motion_smoothness")
OUTPUT_DIR = "paper_metrics"
OUTPUT_NAME = "paper_scores.json"
#: Paper aggregation weights alpha, beta, gamma (App. F.1).
WEIGHTS = {"alpha": .5, "beta": .3, "gamma": .2}


def _paths(session: Path, stem: str) -> dict:
    """Input files of the paper metrics inside a session."""
    hadm = session / "cache" / "multi_limbs" / stem / "hadm_results" / "output.json"
    if not hadm.is_file():  # layout of the original HADM runner: hadm_results/<timestamp>/output.json
        nested = sorted((session / "cache" / "multi_limbs" / stem / "hadm_results").glob("*/output.json"))
        hadm = nested[-1] if nested else hadm
    bone = session / "bone_fit" / f"{stem}_bone_length.npz"
    if not bone.is_file():
        bone = session / "baseline" / f"{stem}_bone_length.npz"
    grid = session / "baseline" / f"{stem}_opensim_skeleton_grid.mot"
    return {"mot": session / "baseline" / f"{stem}_opensim_skeleton.mot", "bone": bone, "hadm": hadm,
            # the kinetic metrics read the uniform-grid export when present (see the module docstring)
            "mot_kinetic": grid if grid.is_file() else session / "baseline" / f"{stem}_opensim_skeleton.mot",
            "gvhmr": session / "gvhmr" / stem / "hmr4d_results.pt"}


def _field(res, *names):
    """Read a score from a paper metric result (dataclass, dict or bare number)."""
    if isinstance(res, (int, float)):
        return float(res)
    for name in names:
        if isinstance(res, dict) and name in res:
            return res[name]
        if hasattr(res, name):
            return getattr(res, name)
    raise KeyError(f"none of {names} in paper result {type(res).__name__}")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _require(path: Path) -> Path:
    """``path`` if it is a file, else ``FileNotFoundError`` (reported as ``missing_input``)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class _Inputs:
    """Everything a paper metric needs from one session; each motion file is read once."""

    def __init__(self, session: Path, stem: str, use_cuda: bool):
        self.session, self.stem, self.use_cuda = session, stem, use_cuda
        self.paths = _paths(session, stem)
        self._motions = {}

    def motion(self, key: str = "mot"):
        """The parsed ``.mot`` file ``paths[key]`` (shared by the metrics that read it)."""
        if key not in self._motions:
            from metrics.kinematic_extreme import utils as kinematics

            self._motions[key] = kinematics.load_mot(str(_require(self.paths[key])))
        return self._motions[key]


# ----------------------------------------------------------------------------- paper metrics
# Each function runs one paper metric, writes its outputs into ``out_dir`` and returns the score.
def _extra_limbs(inp: _Inputs, out_dir: Path):
    import importlib.util

    # the paper script lives in a folder whose name is not a valid package name: load it by path
    spec = importlib.util.spec_from_file_location(
        "hmb_paper_eval_extra_limbs", REPO_ROOT / "metrics/multi-limbs/eval_extra_limbs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    res = module.compute_extra_limbs_score(str(_require(inp.paths["hadm"])))
    return _field(res, "extra_limbs_score", "score", "Score")


def _bone_length(inp: _Inputs, out_dir: Path):
    from metrics.bone_length import bone_length as bone

    res = bone.process_single_fitted_file(str(_require(inp.paths["bone"])), str(out_dir))
    return _field(res, "bone_length_score")


def _joint_range(inp: _Inputs, out_dir: Path):
    from metrics.range_of_motion import joints_violaton as rom

    m = inp.motion()
    res = rom.compute_rom_for_mot_file(str(inp.paths["mot"]), str(out_dir),
                                       preloaded=(m.joint_angles, m.joint_names, m.time))
    return _field(res, "rom_score")


def _kinematic_extremes(inp: _Inputs, out_dir: Path):
    from metrics.kinematic_extreme import main as extremes
    from metrics.kinematic_extreme import utils as kinematics

    mot_path = str(inp.paths["mot_kinetic"])
    # the paper code caches body kinematics in the folder named by this variable: point it into
    # the session for this call and restore it, so other sessions in the process are unaffected
    previous = os.environ.get("KEXTREME_BODY_KIN_CACHE_DIR")
    os.environ["KEXTREME_BODY_KIN_CACHE_DIR"] = str(inp.session / "cache" / "kinematic_extreme")
    try:
        prepared = kinematics.preload_motion_data(mot_path, str(DEFAULT_MODEL), mot=inp.motion("mot_kinetic"))
    finally:
        if previous is None:
            os.environ.pop("KEXTREME_BODY_KIN_CACHE_DIR", None)
        else:
            os.environ["KEXTREME_BODY_KIN_CACHE_DIR"] = previous
    res = extremes.compute_and_save_for_mot(
        mot_path, str(out_dir), str(DEFAULT_MODEL), **WEIGHTS,
        frame_violation_threshold=extremes.FRAME_VIOLATION_THRESHOLD, preloaded=prepared)
    return _field(res, "kextreme_score")


def _motion_smoothness(inp: _Inputs, out_dir: Path):
    from metrics.motion_smoothness import main as smoothness
    from metrics.motion_smoothness import utils as acceleration

    mot_path = str(inp.paths["mot_kinetic"])
    prepared = acceleration.preload_motion_data(mot_path, mot=inp.motion("mot_kinetic"))
    res = smoothness.compute_and_save_for_mot(
        mot_path, str(out_dir), **WEIGHTS,
        frame_violation_threshold=smoothness.FRAME_VIOLATION_THRESHOLD, preloaded=prepared)
    return _field(res, "msmooth_score")


def _self_collision(inp: _Inputs, out_dir: Path):
    from metrics.self_collision_bvh.main import evaluate_sequence

    pt = _require(inp.paths["gvhmr"])
    # the SMPL-X model is located by smpl_utils.resolve_smplx_model_path (HMB_SMPLX_MODEL,
    # HMB_SMPLX_MODEL_DIR, outputs/assets/smplx, GVHMR body models); setting a default
    # folder here would hide the other install locations from this and later metrics.
    # The remaining arguments are the paper CLI's defaults.
    res = evaluate_sequence(str(pt), use_global=True, use_cuda=inp.use_cuda, to_smpl=False,
                            max_collisions=150, min_face_ratio=0.0, min_num_pairs=1,
                            **WEIGHTS, delta=0.0)
    (out_dir / f"{inp.stem}_self_collision_bvh.json").write_text(json.dumps(res, indent=1, default=str))
    return res["self_collision_bvh_score"]


PAPER_METRICS = {
    "extra_limbs": _extra_limbs,
    "bone_length": _bone_length,
    "joint_range": _joint_range,
    "kinematic_extremes": _kinematic_extremes,
    "motion_smoothness": _motion_smoothness,
    "self_collision": _self_collision,
}


def score_session(session_dir, stem: str | None = None, *, metrics=METRICS, write: bool = False,
                  use_cuda: bool = True) -> dict:
    """Paper scores of ``metrics`` for one session.

    Returns ``{"session", "stem", "scores", "overall", "metrics": {metric: {"score",
    "status", "error"?, "seconds"}}}``; ``status`` is ``ok``, ``unsupported`` (no finite
    score), ``missing_input`` or ``error``.  ``overall`` is the mean of the six (null
    unless all six are finite).  With ``write`` the report is also saved as
    ``<S>/paper_scores.json``.
    """
    from hmb.scoring.score_session import session_stem

    session = Path(session_dir)
    stem = stem or session_stem(session)
    inp = _Inputs(session, stem, use_cuda)
    results = {}
    for metric in metrics:
        started = time.perf_counter()
        out_dir = session / OUTPUT_DIR / metric
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            score = PAPER_METRICS[metric](inp, out_dir)  # an unknown metric raises KeyError -> error
            entry = {"score": float(score) if _finite(score) else None,
                     "status": "ok" if _finite(score) else "unsupported"}
        except FileNotFoundError as exc:
            entry = {"score": None, "status": "missing_input", "error": str(exc)}
        except Exception as exc:  # one broken metric must not hide the others
            entry = {"score": None, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc(limit=6)}
        entry["seconds"] = time.perf_counter() - started
        results[metric] = entry
    scores = {m: results[m]["score"] for m in results}
    vals = [scores.get(m) for m in METRICS]
    report = {"session": str(session), "stem": stem, "scores": scores,
              "overall": sum(vals) / 6 if all(_finite(v) for v in vals) else None, "metrics": results}
    if write:
        (session / OUTPUT_NAME).write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv=None) -> int:
    """Command line: paper scores of one session; prints scores, overall and status."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session-dir", required=True, help="a prepared session directory")
    ap.add_argument("--stem", help="video stem inside the session (default: found from the session)")
    ap.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    ap.add_argument("--no-cuda", action="store_true", help="self-collision on the CPU")
    args = ap.parse_args(argv)
    rep = score_session(args.session_dir, args.stem, metrics=tuple(args.metrics), use_cuda=not args.no_cuda,
                        write=True)
    print(json.dumps({"scores": rep["scores"], "overall": rep["overall"],
                      "status": {m: rep["metrics"][m]["status"] for m in rep["metrics"]}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
