# -*- coding: utf-8 -*-
"""Per-frame self-intersection of a body-mesh sequence for metric (IV) self-collision (paper implementation).

For every frame, the torch-mesh-isect BVH (``mesh_intersection``, CUDA only) finds the
pairs of intersecting triangles.  Pairs of triangles that share a vertex are neighbours on
the surface, not a collision, and are dropped.  ``M_t`` is the fraction of the mesh's faces
that take part in a remaining (non-local) pair; frames with fewer than ``min_num_pairs``
pairs or a ratio below ``min_face_ratio`` count as ``M_t = 0`` (numerical noise).
:func:`scoring.aggregate_collision_score` turns ``M_t`` into the video score.

Frames are processed in chunks whose padded BVH output stays below 256 MiB; chunking
never skips a frame or changes the score.  ``main.py`` builds the SMPL-X meshes and is the
command line.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Any

from .scoring import aggregate_collision_score

try:
    from mesh_intersection.bvh_search_tree import BVH
except ImportError:
    BVH = None


#: Upper bound of the padded pair buffer the BVH allocates per call (frames x faces x max_collisions).
_MAX_BVH_OUTPUT_BYTES = 256 * 1024 * 1024


def _nonlocal_collision_counts(
    collisions: torch.Tensor, faces: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-frame counts of non-local collision pairs and of the faces in them, on the GPU.

    A pair is local when its two triangles share a vertex.  Padding slots (negative
    indices), self-pairs and reversed or duplicate pairs are removed within each frame.
    Returns ``(pair_counts, face_counts, face_keys)``; ``face_keys`` encodes each unique
    (frame, face) as ``frame * num_faces + face``.
    """
    if collisions.ndim == 2:
        collisions = collisions.unsqueeze(0)
    if collisions.ndim != 3 or collisions.shape[-1] < 2:
        raise ValueError("BVH collisions must have shape (frames, slots, >=2)")
    num_frames, num_slots, _ = collisions.shape
    num_faces = faces.shape[0]
    first, second = collisions[..., 0].long(), collisions[..., 1].long()
    valid = ((first >= 0) & (first < num_faces) & (second >= 0)
             & (second < num_faces) & (first != second))
    frame = torch.arange(num_frames, device=faces.device)[:, None].expand(num_frames, num_slots)[valid]
    first, second = first[valid], second[valid]
    shared_vertex = (faces[first, :, None] == faces[second, None, :]).any(dim=(1, 2))
    first, second, frame = first[~shared_vertex], second[~shared_vertex], frame[~shared_vertex]
    lower, upper = torch.minimum(first, second), torch.maximum(first, second)
    pair_keys = torch.unique((frame * num_faces + lower) * num_faces + upper)
    frame = pair_keys // (num_faces * num_faces)
    lower, upper = pair_keys // num_faces % num_faces, pair_keys % num_faces
    pair_counts = torch.bincount(frame, minlength=num_frames)
    face_keys = torch.unique(torch.cat((frame * num_faces + lower, frame * num_faces + upper)))
    face_counts = torch.bincount(face_keys // num_faces, minlength=num_frames)
    return pair_counts, face_counts, face_keys


def _bounded_chunk_size(num_faces: int, max_collisions: int, requested: int) -> int:
    """Bound the native padded int64 pair output independently of video length."""
    if requested < 1 or max_collisions < 1:
        raise ValueError("frame_chunk_size and max_collisions must be positive")
    bytes_per_frame = num_faces * max_collisions * 2 * torch.tensor([], dtype=torch.long).element_size()
    return max(1, min(requested, _MAX_BVH_OUTPUT_BYTES // max(1, bytes_per_frame)))


def compute_bvh_self_collision_metrics(
    verts: Any,
    faces: Any,
    max_collisions: int = 150,
    # 1) Threshold to pre-filter tiny numerical noise:
    #    only raw_ratio >= min_face_ratio and num_pairs >= min_num_pairs proceed to severity calculation.
    min_face_ratio: float = 0.001,
    min_num_pairs: int = 5,
    # 2) Dual thresholds: distinguish mild vs severe penetration (applied on filtered M_t)
    tau_mild: float = 0.01,
    tau_severe: float = 0.03,
    # 3) Sequence-level weights (automatically normalized to w1..w4)
    alpha: float = 0.5,   # frequency of severe violations r_severe
    beta: float = 0.3,    # average severity
    gamma: float = 0.2,   # continuity of all positive collision severities
    delta: float = 0.0,   # r_mild (kept for statistics; default weight 0)
    device: str = "cuda",
    frame_chunk_size: int = 8,
) -> Dict[str, float]:
    """Self-collision score of a mesh sequence (0-100, higher is better).

    ``verts`` is (T, V, 3) and ``faces`` (F, 3); CUDA is required.  The paper scores (and
    ``main.py``'s defaults) use ``min_face_ratio=0`` and ``min_num_pairs=1``.

    Per frame:
        raw_ratio_t = |collided_faces_t| / F
        If raw_ratio_t < min_face_ratio or num_pairs_t < min_num_pairs:
            M_t = 0   # treated as noise
        Else:
            M_t = raw_ratio_t

        Piecewise definition:
            if M_t <= tau_mild:
                b_mild_t = 0, b_severe_t = 0, m_t = 0
            elif M_t >= tau_severe:
                b_mild_t = 1, b_severe_t = 1, m_t = 1
            else:
                b_mild_t = 1, b_severe_t = 0,
                m_t = (M_t - tau_mild) / (tau_severe - tau_mild)

    Sequence level (:func:`scoring.aggregate_collision_score`):
        r_mild   = (1/T) Σ b_mild_t
        r_severe = (1/T) Σ b_severe_t
        severity_mean = (Σ m_t) / max(1, Σ b_mild_t)
        continuity = (longest consecutive segment with b_mild_t = 1) / T

        Normalize (alpha, beta, gamma, delta) -> w1..w4
        D = w1*r_severe + w2*severity_mean + w3*continuity + w4*r_mild
        Score = 100 * (1 - clip(D, 0, 1))

    Also returns ``face_collision_ratio`` (faces collided in any accepted frame),
    ``avg_pairs_per_frame`` and ``avg_face_ratio_per_frame``.
    """
    if BVH is None:
        raise ImportError(
            "mesh_intersection is not installed correctly; BVH-based self-collision cannot run."
        )

    # Input validation
    if not torch.is_tensor(verts):
        verts = torch.as_tensor(verts, dtype=torch.float32)
    if not torch.is_tensor(faces):
        faces = torch.as_tensor(faces, dtype=torch.long)

    if verts.ndim != 3 or verts.shape[-1] != 3:
        raise ValueError(f"verts must be (T, V, 3); got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3); got {faces.shape}")

    T, V, _ = verts.shape
    F = faces.shape[0]
    if T == 0 or V == 0 or F == 0:
        raise ValueError("Cannot score a mesh sequence with no frames, vertices, or faces.")
    if not torch.isfinite(verts).all().item():
        raise ValueError("Mesh vertices contain nonfinite coordinates; cannot score a failed reconstruction.")
    if torch.any(faces < 0).item() or torch.any(faces >= V).item():
        raise ValueError("Mesh faces reference invalid vertex indices.")

    # the BVH extension runs on CUDA only
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("torch-mesh-isect BVH requires CUDA; CPU collision scoring is unavailable.")

    verts = verts.to(dev)
    faces_dev = faces.to(dev, dtype=torch.long)

    # BVH
    bvh = BVH(max_collisions=max_collisions)
    if hasattr(bvh, "to"):
        bvh = bvh.to(dev)

    chunk_size = _bounded_chunk_size(F, max_collisions, frame_chunk_size)
    all_collided_faces = torch.zeros(F, dtype=torch.bool, device=dev)
    pair_count_chunks, face_count_chunks = [], []
    with torch.no_grad():
        for start in range(0, T, chunk_size):
            triangles = verts[start:start + chunk_size, faces_dev].contiguous()
            collisions = bvh(triangles)
            if collisions is None:
                counts = torch.zeros(len(triangles), dtype=torch.long, device=dev)
                pairs, counts, face_keys = counts, counts.clone(), counts[:0]
            else:
                pairs, counts, face_keys = _nonlocal_collision_counts(collisions, faces_dev)
            pair_count_chunks.append(pairs)
            face_count_chunks.append(counts)
            # Compare in double precision, as the per-frame acceptance below does, so the
            # two agree exactly at the threshold.
            accepted = (pairs >= min_num_pairs) & (counts.double() / F >= min_face_ratio)
            accepted_faces = face_keys[accepted[face_keys // F]] % F
            all_collided_faces[accepted_faces] = True

    # Copy only the per-frame counts to the host, once; the score is computed in float64.
    per_frame_pairs = torch.cat(pair_count_chunks).cpu().numpy()
    per_frame_face_ratio = torch.cat(face_count_chunks).cpu().numpy() / float(F)
    accepted = (per_frame_pairs >= min_num_pairs) & (per_frame_face_ratio >= min_face_ratio)
    M_ts = np.where(accepted, per_frame_face_ratio, 0.0)

    score = aggregate_collision_score(
        M_ts, tau_mild=tau_mild, tau_severe=tau_severe,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
    )

    # summary statistics
    F = faces.shape[0]
    face_collision_ratio = float(all_collided_faces.sum().item()) / float(F)
    avg_pairs_per_frame = float(np.mean(per_frame_pairs))
    avg_face_ratio_per_frame = float(np.mean(per_frame_face_ratio))

    return {
        **score,
        "face_collision_ratio": face_collision_ratio,
        "avg_pairs_per_frame": avg_pairs_per_frame,
        "avg_face_ratio_per_frame": avg_face_ratio_per_frame,
    }
