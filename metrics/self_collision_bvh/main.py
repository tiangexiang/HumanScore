# -*- coding: utf-8 -*-
"""Metric (IV) self-collision: the original HumanScore paper implementation.

Measures how much of the reconstructed body mesh intersects itself, frame by frame: a body
part passing through another is impossible for a real person.  The mesh is GVHMR's world-frame
SMPL-X body (``hmr4d_results.pt``); a PromptHMR-style ``results.pkl`` also works.

Where it sits in HumanScore
    With ``--paper-scores``, ``hmb/scoring/paper_scores.py`` calls :func:`evaluate_sequence`
    on ``<S>/gvhmr/<stem>/hmr4d_results.pt`` and writes the result to
    ``<S>/paper_metrics/self_collision/``.  The default self-collision score is
    ``hmb/scoring/self_collision.py``.

Pipeline: ``smpl_utils.load_smpl_sequence`` (SMPL-X vertices; needs ``SMPLX_NEUTRAL.npz``)
-> ``bvh_metric.compute_bvh_self_collision_metrics`` (per-frame collided-face ratio ``M_t``
with torch-mesh-isect on CUDA) -> ``scoring.aggregate_collision_score`` (video score).

Run on its own (from the repository root, CUDA required)::

    python -m metrics.self_collision_bvh.main --input_path S/gvhmr/clip/hmr4d_results.pt --output_dir out

Without ``--input_path`` every ``.pt``/``.pkl`` under ``--input_dir`` is scored.  Outputs in
``output_dir``: ``<sequence>_self_collision_bvh.json`` per file,
``self_collision_bvh_scores_summary.csv`` and ``all_self_collision_bvh_results.json``.
"""
import os
import json
import argparse
from typing import Dict, Any, List


def evaluate_sequence(
    result_path: str,
    use_global: bool,
    use_cuda: bool,
    to_smpl: bool,
    max_collisions: int,
    min_face_ratio: float,
    min_num_pairs: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    person_id=None,
) -> Dict[str, Any]:
    """Self-collision result of one mesh file (``.pt`` from GVHMR or PromptHMR-style ``.pkl``).

    ``use_global`` selects GVHMR's world-frame parameters (``smpl_params_global``) instead of
    the camera-frame ones; ``to_smpl`` maps the mesh to SMPL topology first.  The remaining
    arguments are those of ``bvh_metric.compute_bvh_self_collision_metrics``.

    Returns a dict with keys:
        - sequence
        - self_collision_bvh_score        # = metrics["score"] (0-100, higher means fewer collisions)
        - penetration_frame_rate
        - face_collision_ratio
        - avg_pairs_per_frame
        - avg_face_ratio_per_frame
        - max_collisions
        - min_face_ratio
        - min_num_pairs
        - use_global_params
        - use_smpl_topology
    """
    import torch
    from .smpl_utils import load_smpl_sequence
    from .bvh_metric import compute_bvh_self_collision_metrics

    # Recover (T, V, 3) vertices and (F, 3) faces from the result file
    verts, faces = load_smpl_sequence(
        result_path=result_path,
        use_global=use_global,
        use_cuda=use_cuda,
        to_smpl=to_smpl,
        person_id=person_id,
        return_on_cpu=False,
    )

    # Decide which device string to use for compute_bvh_self_collision_metrics
    if use_cuda and torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"

    metrics = compute_bvh_self_collision_metrics(
        verts=verts,
        faces=faces,
        max_collisions=max_collisions,
        min_face_ratio=min_face_ratio,
        min_num_pairs=min_num_pairs,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        device=device_str,
    )

    seq_name = os.path.splitext(os.path.basename(result_path))[0]

    result = {
        "sequence": seq_name,
        "scoring": "humanscore-paper",
        "mesh_source": "prompthmr" if result_path.lower().endswith(".pkl") else "gvhmr",
        "flat_hand_mean": result_path.lower().endswith(".pkl"),
        "self_collision_bvh_score": float(metrics["score"]),  # higher = fewer collisions
        "penetration_frame_rate": float(metrics["penetration_frame_rate"]),
        "r_mild": float(metrics["r_mild"]),
        "r_severe": float(metrics["r_severe"]),
        "severity_mean": float(metrics["severity_mean"]),
        "continuity": float(metrics["continuity"]),
        "avg_M_t": float(metrics["avg_M_t"]),
        "face_collision_ratio": float(metrics["face_collision_ratio"]),
        "avg_pairs_per_frame": float(metrics["avg_pairs_per_frame"]),
        "avg_face_ratio_per_frame": float(metrics["avg_face_ratio_per_frame"]),
        "max_collisions": int(max_collisions),
        "min_face_ratio": float(min_face_ratio),
        "min_num_pairs": int(min_num_pairs),
        "use_global_params": bool(use_global),
        "use_smpl_topology": bool(to_smpl),
    }

    return result


def main():
    """Command-line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(
        description=(
            "Metric (IV) self-collision, paper implementation: "
            "GVHMR outputs -> SMPL-X meshes -> BVH self-intersection -> score"
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Single result file to evaluate (.pt or .pkl). If omitted, iterate over --input_dir.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/baseline",
        help="Directory containing result files: .pt (GVHMR) or .pkl (PromptHMR-style)",
    )
    parser.add_argument(
        "--person_id",
        type=int,
        default=None,
        help="For .pkl only: person id in people dict (default: first available)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/metrics/self_collision_bvh",
        help="Directory to save BVH self-collision evaluation results",
    )
    parser.add_argument(
        "--use_incam",
        action="store_true",
        help="Use smpl_params_incam instead of smpl_params_global",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA (note: mesh_intersection/BVH typically requires GPU)",
    )
    parser.add_argument(
        "--to_smpl",
        action="store_true",
        help="Attempt to map SMPL-X to SMPL topology (requires smplx2smpl_sparse.pt)",
    )
    parser.add_argument(
        "--max_collisions",
        type=int,
        default=150,
        help="Native pair-capacity multiplier per triangle (default: 150); padded slots per frame = faces * multiplier",
    )
    parser.add_argument(
        "--min_face_ratio",
        type=float,
        default=0,
        help=(
            "Minimum collided-face ratio threshold to count a frame as penetration; "
            "frame_face_ratio >= min_face_ratio is considered penetration (default: 0)"
        ),
    )
    parser.add_argument(
        "--min_num_pairs",
        type=int,
        default=1,
        help=(
            "Minimum number of non-local collision pairs to count a frame as penetration; "
            "num_pairs >= min_num_pairs is considered penetration (default: 1)"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Aggregation weight: frequency of severe self-collision frames r_severe (default 0.5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Aggregation weight: mean severity of the colliding frames, severity_mean (default 0.3)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="Aggregation weight: longest run of colliding frames, continuity (default 0.2)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="Aggregation weight: frequency of colliding frames r_mild (default 0.0; the paper does not use it)",
    )

    args = parser.parse_args()

    import torch
    import pandas as pd

    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    use_global = not args.use_incam
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()

    if not use_cuda:
        print(
            "[SelfCollision-BVH] Warning: CUDA is disabled or unavailable. "
            "torch-mesh-isect BVH may not work or will be very slow on CPU."
        )

    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    input_paths: List[str] = []
    if args.input_path:
        if not os.path.isfile(args.input_path):
            raise FileNotFoundError(args.input_path)
        input_paths = [args.input_path]
    else:
        input_dir = args.input_dir
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"input_dir '{input_dir}' not found")
        for root, _, files in os.walk(input_dir):
            for fname in sorted(files):
                if fname.endswith(".pt") or fname.endswith(".pkl"):
                    input_paths.append(os.path.join(root, fname))

    if not input_paths:
        raise FileNotFoundError("No .pt or .pkl result files found to score.")

    for result_path in sorted(set(input_paths)):
        fname = os.path.basename(result_path)
        if not (fname.endswith(".pt") or fname.endswith(".pkl")):
            continue
        seq_name = os.path.splitext(fname)[0]

        print(f"[SelfCollision-BVH] Processing {seq_name} ...")

        try:
            result = evaluate_sequence(
                result_path=result_path,
                use_global=use_global,
                use_cuda=use_cuda,
                to_smpl=args.to_smpl,
                max_collisions=args.max_collisions,
                min_face_ratio=args.min_face_ratio,
                min_num_pairs=args.min_num_pairs,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                delta=args.delta,
                person_id=args.person_id,
            )
        except Exception as e:
            import traceback
            print(f"[SelfCollision-BVH] Failed on {seq_name}: {e}")
            print("[SelfCollision-BVH] Full traceback:")
            traceback.print_exc()
            all_results[seq_name] = {"error": str(e)}
            continue

        all_results[seq_name] = result

        summary_rows.append(
            {
                "sequence": result["sequence"],
                "self_collision_bvh_score": result["self_collision_bvh_score"],
                "penetration_frame_rate": result["penetration_frame_rate"],
                "r_mild": result["r_mild"],
                "r_severe": result["r_severe"],
                "severity_mean": result["severity_mean"],
                "continuity": result["continuity"],
                "avg_M_t": result["avg_M_t"],
                "face_collision_ratio": result["face_collision_ratio"],
                "avg_pairs_per_frame": result["avg_pairs_per_frame"],
                "avg_face_ratio_per_frame": result["avg_face_ratio_per_frame"],
            }
        )

        # per-sequence JSON
        out_json = os.path.join(
            output_dir, f"{seq_name}_self_collision_bvh.json"
        )
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"[SelfCollision-BVH] {seq_name}: "
            f"score={result['self_collision_bvh_score']:.2f}, "
            f"penetration_frame_rate={result['penetration_frame_rate']:.4f}, "
            f"r_mild={result['r_mild']:.4f}, "
            f"r_severe={result['r_severe']:.4f}, "
            f"severity_mean={result['severity_mean']:.4f}, "
            f"continuity={result['continuity']:.4f}, "
            f"avg_M_t={result['avg_M_t']:.4f}, "
            f"avg_face_ratio_per_frame={result['avg_face_ratio_per_frame']:.5f}"
        )

    # all results (including failures) in one JSON
    all_json_path = os.path.join(
        output_dir, "all_self_collision_bvh_results.json"
    )
    with open(all_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # summary CSV
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(
            output_dir, "self_collision_bvh_scores_summary.csv"
        )
        df.to_csv(summary_csv_path, index=False)
        print(f"[SelfCollision-BVH] Summary saved to {summary_csv_path}")
    else:
        print("[SelfCollision-BVH] No valid BVH self-collision results computed.")

    failed = [name for name, result in all_results.items() if "error" in result]
    if failed:
        raise RuntimeError(f"Self-collision evaluation failed for: {', '.join(failed)}. See {all_json_path}.")


if __name__ == "__main__":
    main()
