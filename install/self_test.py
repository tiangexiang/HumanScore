#!/usr/bin/env python3
"""Installation self-test: score a short public video with scripts/humanscore_single.py.

``bash install.sh --self-test`` downloads the clip of ``install/assets.json`` (group selftest) and
runs this script with the environment's interpreter. It checks the exit status of
``humanscore_single.py --paper-scores`` and the schema of the ``humanscore.json`` it writes: six
metric scores, three tier scores and the overall score in [0, 100], the overall equal to the mean
of the six, every metric ``ok``, and the six ``paper_scores`` of the original implementation (so
the installation is checked for both). The result is written to ``OUT/<clip>/self_test.json``.

    python install/self_test.py [--video CLIP] [--out DIR] [--no-paper-scores] [--force]

The scores themselves are not compared with a reference (other GPUs can differ slightly); the
manifest's reference overall score is only printed next to the result.

Exit status: 0 passed, 1 failed.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("extra_limbs", "bone_length", "joint_range", "self_collision", "kinematic_extremes",
           "motion_smoothness")
TIERS = ("anatomy", "kinematic", "kinetic")


def _score(value) -> bool:
    """A finite number in [0, 100] (bools excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) \
        and 0.0 <= value <= 100.0


def check_result(result: Mapping, paper_scores: bool = True) -> List[str]:
    """Problems with a ``humanscore.json`` (empty when it matches the documented schema)."""
    problems = []
    for key in ("video", "scores", "tiers", "overall", "status", "reasons", "person", "files", "timings"):
        if key not in result:
            problems.append(f"missing key {key!r}")
    if problems:
        return problems
    scores, status = result["scores"], result["status"]
    if set(scores) != set(METRICS):
        problems.append(f"scores has {sorted(scores)}, expected {sorted(METRICS)}")
    problems += [f"score {m} = {scores.get(m)!r} is not in [0, 100]" for m in METRICS if not _score(scores.get(m))]
    problems += [f"tier {t} = {result['tiers'].get(t)!r} is not in [0, 100]" for t in TIERS
                 if not _score(result["tiers"].get(t))]
    if not _score(result["overall"]):
        problems.append(f"overall = {result['overall']!r} is not in [0, 100]")
    elif not problems:
        mean = sum(scores[m] for m in METRICS) / len(METRICS)
        if abs(mean - result["overall"]) > 1e-6:
            problems.append(f"overall {result['overall']} is not the mean of the six scores ({mean})")
    not_ok = {m: status.get(m) for m in METRICS if status.get(m) != "ok"}
    if not_ok:
        reasons = result.get("reasons") or {}
        problems += [f"metric {m} is {s!r}: {reasons.get(m, '')}".rstrip(": ") for m, s in not_ok.items()]
    if paper_scores:
        paper = result.get("paper_scores")
        if not isinstance(paper, Mapping):
            problems.append("paper_scores missing")
        else:
            values = {**(paper.get("scores") or {}), "overall": paper.get("overall")}
            problems += [f"paper score {m} = {values.get(m)!r} is not in [0, 100]" for m in (*METRICS, "overall")
                         if not _score(values.get(m))]
    return problems


def main(argv=None) -> int:
    """Run humanscore_single.py on the clip, check its result and write ``self_test.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", default=sys.executable, help="interpreter of the HumanScore environment")
    parser.add_argument("--video", default=None, help="clip to score (default: the manifest's self-test clip)")
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs/install/self-test"), help="output directory")
    parser.add_argument("--no-paper-scores", dest="paper_scores", action="store_false",
                        help="skip the paper scores (and their second skeleton fit)")
    parser.add_argument("--force", action="store_true", help="recompute instead of reusing an earlier run")
    args = parser.parse_args(argv)

    manifest = json.loads((REPO_ROOT / "install/assets.json").read_text())
    clip = next(a for a in manifest["assets"] if a["name"] == "selftest-clip")
    video = Path(args.video) if args.video else REPO_ROOT / clip["dest"]
    if not video.is_file():
        print(f"[self-test] FAILED: {video} not found (python3 install/fetch_assets.py fetch --groups selftest)")
        return 1
    out = Path(args.out).resolve() / video.stem
    command = [args.python, str(REPO_ROOT / "scripts/humanscore_single.py"), str(video), "--out", str(out)]
    command += (["--paper-scores"] if args.paper_scores else []) + (["--force"] if args.force else [])
    print(f"[self-test] scoring {video.name} ({clip['license']}; {clip['source']})", flush=True)
    print(f"[self-test] $ {' '.join(command)}", flush=True)
    started = time.monotonic()
    code = subprocess.call(command, cwd=str(REPO_ROOT))
    seconds = time.monotonic() - started
    report = {"video": str(video), "out": str(out), "exit_status": code, "seconds": round(seconds, 1),
              "problems": []}
    if code != 0:
        report["problems"].append(f"humanscore_single.py exited with status {code} (0 = every score computed)")
    result_path = out / "humanscore.json"
    try:
        result = json.loads(result_path.read_text())
    except (OSError, ValueError) as exc:
        report["problems"].append(f"cannot read {result_path}: {exc}")
        result = None
    if result is not None:
        report["problems"] += check_result(result, args.paper_scores)
        report["overall"] = result.get("overall")
        report["paper_overall"] = (result.get("paper_scores") or {}).get("overall")
    reference = clip.get("reference", {}).get("overall")
    report["reference_overall"] = reference
    report["passed"] = not report["problems"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "self_test.json").write_text(json.dumps(report, indent=2))
    if report["passed"]:
        note = ""
        if reference is not None and isinstance(report.get("overall"), (int, float)):
            note = f"; reference {reference:.2f}"
        print(f"[self-test] PASSED in {seconds:.0f} s: overall {report['overall']:.2f}{note} ({result_path})",
              flush=True)
        return 0
    print(f"[self-test] FAILED after {seconds:.0f} s:", flush=True)
    for problem in report["problems"]:
        print(f"[self-test]   {problem}", flush=True)
    print(f"[self-test] details: {out / 'self_test.json'} and the logs in {out}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
