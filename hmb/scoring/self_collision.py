"""Metric (IV) self-collision: penetration depth between NON-ADJACENT body parts.

Paper meaning (App. F.1 / E.2): per frame, a body mesh (SMPL-X from GVHMR) is tested
for triangle-triangle self-intersections; adjacent faces are excluded; the per-frame
collision magnitude ``M_t`` is ramped between a mild and a severe threshold and
aggregated with the paper weights::

    D = 0.5 r_severe + 0.3 s + 0.2 p,     score = 100 (1 - clip(D, 0, 1)).

The paper's implementation (``metrics/self_collision_bvh``) uses ``M_t`` = fraction of
SMPL-X faces in an intersecting pair.  On the SMPL-X template that fraction is never
zero, so HumanScore measures the video rather than the body model:

1. **Eyeballs removed.**  The two eyeballs are separate closed components that
   intersect the eye sockets in every pose (most of the rest pose's intersections).
2. **Fingers removed.**  GVHMR estimates the 21 body joints only, so the fingers are
   frozen in a half-curled mean pose and carry no evidence from the video
   (``EXCLUDE_FINGERS``); the hand segment is the palm/wrist surface.
3. **Only non-adjacent parts.**  Skinning interpenetrates the skin at every flexed
   joint (armpit, groin, elbow/knee crease, chin on chest); whether that joint angle is
   plausible is the job of metric (III).  Only pairs between NON-ADJACENT segments of a
   14-segment partition of the SMPL-X skinning weights count (``SEGMENT_OF_JOINT``;
   adjacency = same segment or parent/child in ``SEGMENT_PARENT``).  The rest pose then
   has no counted pair.
4. **Depth instead of face fraction.**  Contact (hands on hips, crossed legs) produces
   shallow interpenetration of the reconstructed surfaces; a body part passing through
   another produces deep interpenetration.  ``M_t`` is the penetration DEPTH, and only
   depth beyond a contact tolerance is penalised.

Penetration depth (per frame, per colliding segment pair A, B).  A vertex of A is
inside B when its generalised winding number w.r.t. B's (open) surface exceeds 1/2
(Jacobson et al. 2013); its depth is the exact distance to B's triangles.  The
penetrating patch of A is the inside set connected, along A's mesh edges, to a vertex
of an A face that intersects B, so the depth is always tied to a detected
intersection.  depth(A, B) = max over both patches; the frame depth is the max over
colliding pairs.

Scoring (:class:`CollisionConfig`): ``m_t = ramp(depth_t, contact, severe)``,
``b_t = depth_t > contact``, ``r_severe = mean(depth_t >= severe)``; ``s`` and ``p``
come from :func:`hmb.scoring.aggregate.aggregate` (``s`` averages flagged frames only)
and ``r`` is replaced by ``r_severe`` as in the paper.  ``contact_depth_m`` = 2 cm covers
skin compression at a contact plus the error of a monocular body mesh; ``severe_depth_m``
= 5 cm is the contact tolerance plus roughly a forearm radius.

Speed.  The default backend (:func:`triangle_pairs_patches`) never tests adjacent
segments: faces are grouped into 32-face patches per segment, patch AABBs of all
allowed patch pairs are tested for all frames at once, and only face pairs of
overlapping patches go to the exact 17-axis separating-axis test (the predicate of
torch-mesh-isect).  It is plain torch on CPU or CUDA and needs no compiled extension.
The torch-mesh-isect BVH (``backend="bvh"``) and a numpy reference
(``backend="numpy"``) give the same pair sets.

Entry points
------------
* :func:`score_session` / CLI ``python -m hmb.scoring.self_collision --session-dir S``
  read ``<S>/gvhmr/<stem>/hmr4d_results.pt`` (GVHMR body parameters) and write
  ``<S>/metrics/self_collision/<stem>_self_collision.json`` (+ per-frame npz).  The
  licensed SMPL-X neutral model is located by :func:`resolve_smplx_model`
  (``HMB_SMPLX_MODEL``, ``HMB_SMPLX_MODEL_DIR``, then the install locations).
  A CUDA GPU is used when available; ``--backend cpu`` runs everything on the CPU.
* :func:`measure_params` (GVHMR parameter dict -> per-frame measurements, torch +
  smplx) and :func:`score_measurements` (pure numpy).
* :func:`measure_mesh_sequence` for arbitrary meshes with a custom
  :class:`BodyTopology` (used by the synthetic unit tests).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hmb.scoring.aggregate import aggregate, ramp

REPO_ROOT = Path(__file__).resolve().parents[2]
SMPLX_MODEL_NAME = "SMPLX_NEUTRAL.npz"
#: Install location of the model (the installer links it here); the last place searched.
DEFAULT_SMPLX_MODEL = REPO_ROOT / "outputs/assets/smplx" / SMPLX_MODEL_NAME
#: GVHMR's body-model folder (``hmb.gvhmr_prepare`` links the model there).
GVHMR_SMPLX_MODEL = REPO_ROOT / "GVHMR/inputs/checkpoints/body_models/smplx" / SMPLX_MODEL_NAME


def smplx_model_candidates() -> List[Tuple[str, Path]]:
    """``(source, path)`` of every place searched for the SMPL-X neutral model, in order.

    As in the rest of the code base (``hmb.cli.default_smplx_dir``,
    ``metrics/self_collision_bvh/smpl_utils.resolve_smplx_model_path``): an
    explicit ``HMB_SMPLX_MODEL`` (the ``.npz`` file) is the only candidate when
    set; otherwise an explicit ``HMB_SMPLX_MODEL_DIR`` (the folder holding
    ``SMPLX_NEUTRAL.npz``, or its parent with a ``smplx/`` subfolder) is the only
    place searched; otherwise ``hmb.cli.default_smplx_dir()``, GVHMR's body-model
    folder and ``outputs/assets/smplx/``.  An explicit setting is never silently
    replaced by another model.
    """
    override = os.environ.get("HMB_SMPLX_MODEL", "").strip()
    if override:
        return [("HMB_SMPLX_MODEL", Path(override).expanduser())]
    model_dir = os.environ.get("HMB_SMPLX_MODEL_DIR", "").strip()
    if model_dir:
        folder = Path(model_dir).expanduser()
        return [("HMB_SMPLX_MODEL_DIR", folder / SMPLX_MODEL_NAME),
                ("HMB_SMPLX_MODEL_DIR", folder / "smplx" / SMPLX_MODEL_NAME)]
    found: List[Tuple[str, Path]] = []
    try:
        from hmb.cli import default_smplx_dir

        found.append(("hmb.cli default_smplx_dir", Path(default_smplx_dir()) / SMPLX_MODEL_NAME))
    except ImportError:  # pragma: no cover - hmb.cli is part of this package
        pass
    found += [("GVHMR body models", GVHMR_SMPLX_MODEL), ("outputs/assets", DEFAULT_SMPLX_MODEL)]
    unique, seen = [], set()
    for source, path in found:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            unique.append((source, path))
    return unique


def resolve_smplx_model(model_path: Optional[Path | str] = None) -> Path:
    """The SMPL-X neutral model file: ``model_path`` when given, else the first existing
    :func:`smplx_model_candidates` entry.  Raises ``FileNotFoundError`` naming every place searched
    (``hmb.scoring.score_session`` reports it as ``missing_input`` with that message)."""
    if model_path is not None and str(model_path) != "":
        path = Path(model_path).expanduser()
        if path.is_file():
            return path.resolve()
        searched = [("model_path", path)]
    else:
        searched = smplx_model_candidates()
        for _, path in searched:
            if path.is_file():
                return path.resolve()
    places = "; ".join(f"{path} ({source})" for source, path in searched)
    raise FileNotFoundError(
        f"SMPL-X model {SMPLX_MODEL_NAME} not found: place the licensed model in outputs/assets/smplx/, or set "
        f"HMB_SMPLX_MODEL to the file or HMB_SMPLX_MODEL_DIR to its folder; searched: {places}")


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CollisionConfig:
    """Scoring constants of the self-collision metric.

    contact_depth_m
        Contact tolerance (the paper's ``tau_mild`` analogue): penetration up to this
        depth between non-adjacent parts is treated as contact (skin compression and
        monocular mesh error).  2 cm.
    severe_depth_m
        Depth at which a frame is fully severe (``tau_severe`` analogue): the contact
        tolerance plus 3 cm, roughly the radius of a forearm, i.e. a part that has sunk
        a limb radius into another part is passing through it.
    weights
        Paper App. F.1 weights (alpha, beta, gamma) for r_severe, s, p.
    """

    contact_depth_m: float = 0.02
    severe_depth_m: float = 0.05
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)

    def validate(self) -> "CollisionConfig":
        """Raise ``ValueError`` unless ``0 <= contact_depth_m < severe_depth_m``; return self."""
        if not (0 <= self.contact_depth_m < self.severe_depth_m):
            raise ValueError("require 0 <= contact_depth_m < severe_depth_m")
        return self


DEFAULT_CONFIG = CollisionConfig()

# Native BVH capacity: torch-mesh-isect writes every AABB-overlap candidate
# into a buffer of ``num_faces * MAX_COLLISIONS`` slots WITHOUT bounds checks,
# so the capacity must exceed the per-face average number of AABB overlaps
# (adjacent faces included).  150 (the paper code's value) is a wide margin over
# the few candidates per face of SMPL-X; an overflow would corrupt GPU memory.
# The default path (patch broad phase) has no such buffer.
MAX_COLLISIONS = 150
# Bound the padded int64 BVH output per call.
MAX_BVH_OUTPUT_BYTES = 256 * 1024 * 1024
# Bound the (frames x points x triangles) winding/distance work per chunk
# (~30 float temporaries per element -> ~0.25 GB at 2M elements).
WINDING_CHUNK_ELEMENTS = 2_000_000
# Torch broad phase: (frame, face, grid cell) entries per chunk (~100 B each).
TORCH_BROADPHASE_ENTRIES = 4_000_000
# Patch broad phase (default path): faces per patch, patch-pair AABB tests
# per chunk (frames x allowed patch pairs) and face pairs per exact-test chunk.
PATCH_SIZE = 32
PATCH_PAIR_TESTS = 8_000_000
PATCH_FACE_PAIRS = 2_000_000
# Region growing stops after this many edge rings (>= 2 m of skin at the
# ~1 cm SMPL-X edge length, i.e. never binding for a body-sized mesh).
MAX_GROWTH_RINGS = 400
# Grid cell coordinates of the torch broad phase are clamped to +-GRID_OFFSET cells and
# packed with the frame index into one int64 key (GRID_SIZE values per axis).
GRID_SIZE, GRID_OFFSET = 4096, 2048


# ---------------------------------------------------------------------------
# Body partition (anatomy, not data)
# ---------------------------------------------------------------------------
#: SMPL-X joints in skinning-weight column order (body, jaw, eyes, then 2 x 15 finger joints).
SMPLX_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw",
    "left_eye_smplhf", "right_eye_smplhf",
) + tuple(f"left_hand_{i}" for i in range(15)) + tuple(f"right_hand_{i}" for i in range(15))

#: The 14 body segments whose mutual penetration is scored.
SEGMENTS = ("torso", "head", "l_upperarm", "r_upperarm", "l_forearm", "r_forearm",
            "l_hand", "r_hand", "l_thigh", "r_thigh", "l_shank", "r_shank", "l_foot", "r_foot")
# Kinematic tree over segments.  Collars are merged into the torso and the
# neck into the head, so the armpit (upper arm vs chest) and the chin/chest
# folds are parent/child folds and are excluded like every other joint crease.
SEGMENT_PARENT = {
    "head": "torso", "l_upperarm": "torso", "r_upperarm": "torso",
    "l_forearm": "l_upperarm", "r_forearm": "r_upperarm",
    "l_hand": "l_forearm", "r_hand": "r_forearm",
    "l_thigh": "torso", "r_thigh": "torso",
    "l_shank": "l_thigh", "r_shank": "r_thigh", "l_foot": "l_shank", "r_foot": "r_shank",
}
# Segment of every SMPL-X joint (index into SMPLX_JOINT_NAMES); finger joints belong to the hand.
SEGMENT_OF_JOINT = {
    0: "torso", 3: "torso", 6: "torso", 9: "torso", 13: "torso", 14: "torso",
    12: "head", 15: "head", 22: "head", 23: "eye", 24: "eye",
    16: "l_upperarm", 17: "r_upperarm", 18: "l_forearm", 19: "r_forearm",
    20: "l_hand", 21: "r_hand", 1: "l_thigh", 2: "r_thigh", 4: "l_shank", 5: "r_shank",
    7: "l_foot", 10: "l_foot", 8: "r_foot", 11: "r_foot",
}
SEGMENT_OF_JOINT.update({j: "l_hand" for j in range(25, 40)})
SEGMENT_OF_JOINT.update({j: "r_hand" for j in range(40, 55)})
# Eyeballs: excluded from the mesh (disconnected closed components that
# intersect the eye sockets in every pose).
EXCLUDED_SEGMENTS = ("eye",)
# Fingers: GVHMR estimates the 21 body joints only; every finger is frozen in
# the learned mean (half-curled) hand pose, so finger geometry carries no
# evidence from the video.  With a hand resting on the hip, thigh or head the
# frozen fingertips sink several cm into that part.  The choice follows from this
# observability argument; ``exclude_fingers=False`` scores the finger vertices as
# part of the hand.
EXCLUDE_FINGERS = True


def segment_adjacency(segments: Sequence[str] = SEGMENTS,
                      parent: Optional[Dict[str, str]] = None) -> np.ndarray:
    """(S, S) bool: same segment or parent/child in the segment tree."""
    parent = SEGMENT_PARENT if parent is None else parent
    index = {name: i for i, name in enumerate(segments)}
    adjacent = np.eye(len(segments), dtype=bool)
    for child, par in parent.items():
        if child in index and par in index:
            adjacent[index[child], index[par]] = adjacent[index[par], index[child]] = True
    return adjacent


@dataclass
class BodyTopology:
    """Mesh connectivity + part labels of the kept (scored) surface.

    ``keep_vertices`` indexes the full vertex array of the body model; the
    kept vertices are renumbered 0..V-1 and ``faces`` refers to that
    numbering.  ``face_full_index`` maps kept faces back to the full face
    list (the paper's face-fraction definition uses every face).
    """

    faces_full: np.ndarray            # (F_full, 3) int64, full vertex numbering
    keep_vertices: np.ndarray         # (V,) int64 indices into the full vertex array
    faces: np.ndarray                 # (F, 3) int64, kept numbering
    face_full_index: np.ndarray       # (F,) int64 indices into faces_full
    vertex_segment: np.ndarray        # (V,) int64
    face_segment: np.ndarray          # (F,) int64
    adjacent: np.ndarray              # (S, S) bool
    segment_names: Tuple[str, ...]
    num_full_vertices: int
    edges: np.ndarray = field(init=False)  # (E, 2) unique undirected edges, kept numbering

    def __post_init__(self):
        f = self.faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        self.edges = np.unique(np.sort(e, axis=1), axis=0)

    @property
    def num_segments(self) -> int:
        """Number of segments ``S``."""
        return len(self.segment_names)

    @classmethod
    def from_labels(cls, faces_full, vertex_segment_full, segment_names, adjacent,
                    keep_vertex_mask=None) -> "BodyTopology":
        """Generic constructor (used for synthetic meshes in tests).

        ``vertex_segment_full`` holds a segment index per full vertex (any
        negative value marks an excluded vertex).  A face is kept only when
        all three vertices are kept; its segment is the majority label of its
        vertices (ties: first vertex).
        """
        faces_full = np.asarray(faces_full, dtype=np.int64)
        labels = np.asarray(vertex_segment_full, dtype=np.int64)
        keep = labels >= 0 if keep_vertex_mask is None else (np.asarray(keep_vertex_mask, bool) & (labels >= 0))
        keep_idx = np.flatnonzero(keep)
        remap = np.full(labels.size, -1, dtype=np.int64)
        remap[keep_idx] = np.arange(keep_idx.size)
        face_keep = keep[faces_full].all(axis=1)
        face_full_index = np.flatnonzero(face_keep)
        faces = remap[faces_full[face_keep]]
        vs = labels[keep_idx]
        fl = vs[faces]
        maj = np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0])
        return cls(faces_full=faces_full, keep_vertices=keep_idx, faces=faces,
                   face_full_index=face_full_index, vertex_segment=vs, face_segment=maj,
                   adjacent=np.asarray(adjacent, bool), segment_names=tuple(segment_names),
                   num_full_vertices=int(labels.size))


def smplx_topology(model_path: Optional[Path | str] = None,
                   exclude_fingers: Optional[bool] = None) -> BodyTopology:
    """14-segment partition of SMPL-X from its skinning weights.

    Vertex label = segment with the largest summed skinning weight; face
    label = segment with the largest summed weight over its three vertices.
    Eyeball vertices (label ``eye``) are always removed with their faces.
    With ``exclude_fingers`` the vertices dominated by the 30 finger joints
    are removed too (``EXCLUDE_FINGERS`` documents why); the hand segment is
    then the palm/wrist surface whose pose GVHMR does estimate.
    """
    exclude_fingers = EXCLUDE_FINGERS if exclude_fingers is None else exclude_fingers
    with np.load(str(resolve_smplx_model(model_path)), allow_pickle=True) as data:
        faces_full = data["f"].astype(np.int64)
        weights = np.asarray(data["weights"], dtype=np.float64)
    names = SEGMENTS + EXCLUDED_SEGMENTS + ("l_fingers", "r_fingers")
    seg_of = dict(SEGMENT_OF_JOINT)
    seg_of.update({j: "l_fingers" for j in range(25, 40)})
    seg_of.update({j: "r_fingers" for j in range(40, 55)})
    joint_to_seg = np.array([names.index(seg_of[j]) for j in range(weights.shape[1])])
    seg_w = np.zeros((weights.shape[0], len(names)))
    np.add.at(seg_w.T, joint_to_seg, weights.T)
    if not exclude_fingers:  # fingers belong to their hand
        for fing, hand in (("l_fingers", "l_hand"), ("r_fingers", "r_hand")):
            seg_w[:, names.index(hand)] += seg_w[:, names.index(fing)]
            seg_w[:, names.index(fing)] = 0.0
    vlab = seg_w.argmax(axis=1)
    excluded = vlab >= len(SEGMENTS)
    labels = np.where(excluded, -1, vlab)
    topo = BodyTopology.from_labels(faces_full, labels, SEGMENTS, segment_adjacency())
    # Face labels from summed weights (more stable at segment borders than majority).
    fw = seg_w[faces_full[topo.face_full_index]].sum(axis=1)[:, : len(SEGMENTS)]
    topo.face_segment = fw.argmax(axis=1).astype(np.int64)
    return topo


# ---------------------------------------------------------------------------
# Narrow phase: triangle-triangle tests (CUDA BVH or exact CPU fallback)
# ---------------------------------------------------------------------------
def triangle_pairs_cpu(vertices: np.ndarray, faces: np.ndarray, cell: Optional[float] = None) -> np.ndarray:
    """Exact intersecting triangle pairs of ONE mesh, CPU (numpy).

    Same predicate as torch-mesh-isect: closed separating-axis test over the
    17 axes (2 normals, 9 edge-edge cross products, 6 in-plane axes), pairs
    whose triangles share a vertex are excluded.  Broad phase: uniform grid
    of AABBs.  Returns (N, 2) int64 sorted pairs.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]                                   # (F, 3, 3)
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    ext = (hi - lo).max(axis=1)
    if cell is None:
        cell = float(max(np.median(ext) * 2.0, 1e-9))
    ilo = np.floor(lo / cell).astype(np.int64)
    ihi = np.floor(hi / cell).astype(np.int64)
    span = ihi - ilo + 1
    counts = span.prod(axis=1)
    owner = np.repeat(np.arange(len(f)), counts)
    start = np.repeat(np.cumsum(counts) - counts, counts)
    local = np.arange(owner.size) - start
    sp = span[owner]
    cx = ilo[owner, 0] + local % sp[:, 0]
    cy = ilo[owner, 1] + (local // sp[:, 0]) % sp[:, 1]
    cz = ilo[owner, 2] + local // (sp[:, 0] * sp[:, 1])
    # spatial hash of the grid cell (Teschner et al. 2003); a hash collision only adds a
    # candidate that the AABB test below removes
    key = (cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)
    order = np.lexsort((owner, key))
    key, owner = key[order], owner[order]
    # after sorting, faces sharing a cell are consecutive: pair every entry with the next d-th one
    cand = []
    for d in range(1, owner.size):
        same = key[d:] == key[:-d]
        if not same.any():
            break
        cand.append(np.stack([owner[:-d][same], owner[d:][same]], axis=1))
    if not cand:
        return np.zeros((0, 2), np.int64)
    cand = np.concatenate(cand)
    cand = np.unique(np.sort(cand, axis=1), axis=0)
    cand = cand[cand[:, 0] != cand[:, 1]]
    overlap = ((lo[cand[:, 0]] <= hi[cand[:, 1]]) & (hi[cand[:, 0]] >= lo[cand[:, 1]])).all(axis=1)
    cand = cand[overlap]
    shared = (f[cand[:, 0]][:, :, None] == f[cand[:, 1]][:, None, :]).any(axis=(1, 2))
    cand = cand[~shared]
    if not len(cand):
        return cand
    hit = _sat_intersect(tri[cand[:, 0]], tri[cand[:, 1]])
    return cand[hit]


def _sat_intersect(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Vectorised closed separating-axis triangle test, (N,3,3) x (N,3,3) -> (N,)."""
    e1 = np.stack([t1[:, 1] - t1[:, 0], t1[:, 2] - t1[:, 0], t1[:, 2] - t1[:, 1]], axis=1)
    e2 = np.stack([t2[:, 1] - t2[:, 0], t2[:, 2] - t2[:, 0], t2[:, 2] - t2[:, 1]], axis=1)
    n1 = np.cross(e1[:, 1], e1[:, 2])
    n2 = np.cross(e2[:, 1], e2[:, 2])
    axes = [n1, n2]
    axes += [np.cross(e1[:, i], e2[:, j]) for i in range(3) for j in range(3)]
    axes += [np.cross(n1, e1[:, i]) for i in range(3)] + [np.cross(n1, e2[:, j]) for j in range(3)]
    ok = np.ones(len(t1), dtype=bool)
    for ax in axes:
        p1 = np.einsum("nkd,nd->nk", t1, ax)
        p2 = np.einsum("nkd,nd->nk", t2, ax)
        ok &= (p1.min(axis=1) <= p2.max(axis=1)) & (p1.max(axis=1) >= p2.min(axis=1))
    return ok


def triangle_pairs_bvh(vertices, faces, max_collisions: int = MAX_COLLISIONS):
    """Intersecting non-local face pairs for a (T, V, 3) CUDA tensor sequence.

    Returns an (N, 3) int64 numpy array of unique (frame, lo, hi) rows.
    """
    import torch
    from mesh_intersection.bvh_search_tree import BVH

    verts = vertices if torch.is_tensor(vertices) else torch.as_tensor(vertices)
    verts = verts.to("cuda", torch.float32)
    f = torch.as_tensor(np.asarray(faces), dtype=torch.long, device="cuda")
    num_faces = int(f.shape[0])
    per_frame = num_faces * max_collisions * 2 * 8          # bytes of the BVH output per frame
    chunk = max(1, MAX_BVH_OUTPUT_BYTES // max(1, per_frame))
    bvh = BVH(max_collisions=max_collisions)
    rows = []
    with torch.no_grad():
        for start in range(0, verts.shape[0], chunk):
            tris = verts[start:start + chunk][:, f].contiguous()
            col = bvh(tris)                                   # (B, F*mc, 2)
            valid = (col[..., 0] >= 0) & (col[..., 1] >= 0)
            fr, slot = torch.nonzero(valid, as_tuple=True)
            a = col[fr, slot, 0]
            b = col[fr, slot, 1]
            shared = (f[a][:, :, None] == f[b][:, None, :]).any(dim=2).any(dim=1)
            keep = ~shared & (a != b)
            fr, a, b = fr[keep] + start, a[keep], b[keep]
            lo, hi = torch.minimum(a, b), torch.maximum(a, b)
            key = torch.unique((fr * num_faces + lo) * num_faces + hi)
            rows.append(torch.stack([key // (num_faces * num_faces), key // num_faces % num_faces,
                                     key % num_faces], dim=1).cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, 3), np.int64)


def _sat_intersect_torch(t1, t2):
    """Torch twin of :func:`_sat_intersect` (closed separating-axis test, 17 axes)."""
    import torch
    e1 = torch.stack([t1[:, 1] - t1[:, 0], t1[:, 2] - t1[:, 0], t1[:, 2] - t1[:, 1]], dim=1)
    e2 = torch.stack([t2[:, 1] - t2[:, 0], t2[:, 2] - t2[:, 0], t2[:, 2] - t2[:, 1]], dim=1)
    n1 = torch.cross(e1[:, 1], e1[:, 2], dim=-1)
    n2 = torch.cross(e2[:, 1], e2[:, 2], dim=-1)
    axes = [n1, n2] + [torch.cross(e1[:, i], e2[:, j], dim=-1) for i in range(3) for j in range(3)]
    axes += [torch.cross(n1, e1[:, i], dim=-1) for i in range(3)]
    axes += [torch.cross(n1, e2[:, j], dim=-1) for j in range(3)]
    ax = torch.stack(axes, dim=1)                                   # (N,17,3)
    p1 = torch.einsum("nkd,nad->nak", t1, ax)                       # (N,17,3)
    p2 = torch.einsum("nkd,nad->nak", t2, ax)
    ok = (p1.amin(-1) <= p2.amax(-1)) & (p1.amax(-1) >= p2.amin(-1))
    return ok.all(dim=1)


def triangle_pairs_torch(vertices, faces, *, face_segment=None, allowed=None,
                         cell: Optional[float] = None, max_entries: int = TORCH_BROADPHASE_ENTRIES) -> np.ndarray:
    """Intersecting face pairs for a (T, V, 3) sequence with plain torch ops (CPU or CUDA).

    Same predicate as torch-mesh-isect (closed 17-axis separating-axis test;
    triangles sharing a vertex are excluded).  Broad phase: all frames of a
    chunk are hashed into one uniform grid of face AABBs and sorted once, so
    the work is a few large kernels per chunk instead of a BVH build per
    frame.  With ``face_segment`` (F,) and ``allowed`` (S, S) bool, candidate
    pairs whose segments are not allowed are dropped BEFORE the exact test
    (only non-adjacent segment pairs are scored).  Returns unique
    (frame, lo, hi) int64 rows.
    """
    import torch
    verts = vertices if torch.is_tensor(vertices) else torch.as_tensor(np.asarray(vertices))
    verts = verts.double() if verts.dtype == torch.float64 else verts.float()
    dev = verts.device
    f = torch.as_tensor(np.asarray(faces), dtype=torch.long, device=dev)
    T, F = int(verts.shape[0]), int(f.shape[0])
    seg = allow = None
    if face_segment is not None and allowed is not None:
        seg = torch.as_tensor(np.asarray(face_segment), dtype=torch.long, device=dev)
        allow = torch.as_tensor(np.asarray(allowed), dtype=torch.bool, device=dev)
    if T == 0 or F == 0:
        return np.zeros((0, 3), np.int64)
    if cell is None:
        tri0 = verts[0][f]
        cell = float(max(2.0 * (tri0.amax(1) - tri0.amin(1)).amax(-1).median().item(), 1e-6))
    K, OFF = GRID_SIZE, GRID_OFFSET
    rows = []
    tstep = max(1, int(max_entries // max(1, F * 4)))
    for t0 in range(0, T, tstep):
        tri = verts[t0:t0 + tstep][:, f]                                    # (n,F,3,3)
        n = tri.shape[0]
        lo, hi = tri.amin(2), tri.amax(2)                                   # (n,F,3)
        ilo = torch.floor(lo / cell).long().clamp(-OFF, OFF - 1)
        ihi = torch.floor(hi / cell).long().clamp(-OFF, OFF - 1)
        span = (ihi - ilo + 1).reshape(-1, 3)
        cnt = span.prod(-1)
        owner = torch.repeat_interleave(torch.arange(n * F, device=dev), cnt)
        start = torch.cumsum(cnt, 0) - cnt
        local = torch.arange(owner.numel(), device=dev) - start[owner]
        sp = span[owner]
        base = ilo.reshape(-1, 3)[owner]
        cx = base[:, 0] + local % sp[:, 0]
        cy = base[:, 1] + (local // sp[:, 0]) % sp[:, 1]
        cz = base[:, 2] + local // (sp[:, 0] * sp[:, 1])
        frame = owner // F
        key = ((frame * K + cx + OFF) * K + cy + OFF) * K + cz + OFF
        key, order = torch.sort(key)
        owner = owner[order]
        # entries sharing a (frame, cell) key are consecutive: pair each with the next d-th one
        cand = []
        d = 1
        while d < key.numel():
            same = key[d:] == key[:-d]
            if not bool(same.any()):
                break
            cand.append(torch.stack([owner[:-d][same], owner[d:][same]], dim=1))
            d += 1
        if not cand:
            continue
        cand = torch.cat(cand)
        fr = cand[:, 0] // F
        a, b = cand[:, 0] % F, cand[:, 1] % F
        a, b = torch.minimum(a, b), torch.maximum(a, b)
        keep = a != b
        if allow is not None:
            keep &= allow[seg[a], seg[b]]
        fr, a, b = fr[keep], a[keep], b[keep]
        uk = torch.unique((fr * F + a) * F + b)
        fr, a, b = uk // (F * F), (uk // F) % F, uk % F
        shared = (f[a][:, :, None] == f[b][:, None, :]).any(dim=2).any(dim=1)
        fr, a, b = fr[~shared], a[~shared], b[~shared]
        la, ha, lb, hb = lo[fr, a], hi[fr, a], lo[fr, b], hi[fr, b]
        ov = ((la <= hb) & (ha >= lb)).all(dim=1)
        fr, a, b = fr[ov], a[ov], b[ov]
        hit = torch.zeros(fr.numel(), dtype=torch.bool, device=dev)
        step = 1 << 20                                                       # exact tests per batch
        for s in range(0, fr.numel(), step):
            sl = slice(s, s + step)
            hit[sl] = _sat_intersect_torch(tri[fr[sl], a[sl]], tri[fr[sl], b[sl]])
        rows.append(torch.stack([fr[hit] + t0, a[hit], b[hit]], dim=1).cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, 3), np.int64)


def _morton(xyz: np.ndarray, bits: int = 10) -> np.ndarray:
    """Morton (Z-order) code of 3-D points: nearby points get nearby codes."""
    lo, hi = xyz.min(0), xyz.max(0)
    q = ((xyz - lo) / np.maximum(hi - lo, 1e-12) * (2 ** bits - 1)).astype(np.int64)
    code = np.zeros(len(xyz), np.int64)
    for b in range(bits):
        for ax in range(3):
            code |= ((q[:, ax] >> b) & 1) << (3 * b + ax)
    return code


def build_patches(rest_vertices: np.ndarray, faces: np.ndarray, face_segment: np.ndarray,
                  allowed: np.ndarray, size: int = PATCH_SIZE):
    """Group each segment's faces into compact patches of ``size`` faces.

    Faces are ordered by the Morton code of their centroid (within the
    segment) and cut into consecutive runs, so a patch is a small connected-ish
    surface piece.  Returns (patch_faces (P, size) int64, padded by repeating
    the last face; patch pairs (Q, 2) int64 allowed by ``allowed``).
    """
    cent = np.asarray(rest_vertices, float)[faces].mean(axis=1)
    patches, pseg = [], []
    for s in np.unique(face_segment):
        idx = np.flatnonzero(face_segment == s)
        idx = idx[np.argsort(_morton(cent[idx]), kind="stable")]
        for start in range(0, idx.size, size):
            chunk = idx[start:start + size]
            patches.append(np.concatenate([chunk, np.full(size - chunk.size, chunk[-1])]))
            pseg.append(s)
    patch_faces = np.asarray(patches, np.int64)
    pseg = np.asarray(pseg)
    i, j = np.triu_indices(len(patch_faces), k=1)
    ok = np.asarray(allowed, bool)[pseg[i], pseg[j]]
    return patch_faces, np.stack([i[ok], j[ok]], axis=1)


def triangle_pairs_patches(vertices, faces, face_segment, allowed, *, size: int = PATCH_SIZE,
                           max_face_pairs: int = PATCH_FACE_PAIRS) -> np.ndarray:
    """Intersecting face pairs between ALLOWED segments (default path).

    Two-level broad phase in plain torch (CPU or CUDA): (1) per frame, AABBs
    of compact face patches (:func:`build_patches`) are tested for every
    allowed patch pair at once; (2) all face pairs of overlapping patch pairs
    are AABB-tested and then decided by the exact separating-axis test (same
    predicate as torch-mesh-isect; vertex-sharing pairs excluded).  Because
    adjacent segments are never paired, the dense self-overlap of a
    segment's own surface (face, hands, joint folds) costs nothing.
    """
    import torch
    verts = vertices if torch.is_tensor(vertices) else torch.as_tensor(np.asarray(vertices))
    verts = verts.double() if verts.dtype == torch.float64 else verts.float()
    dev = verts.device
    T = int(verts.shape[0])
    faces = np.asarray(faces, np.int64)
    if T == 0 or len(faces) == 0:
        return np.zeros((0, 3), np.int64)
    pf_np, qp_np = build_patches(verts[0].detach().cpu().numpy(), faces, np.asarray(face_segment), allowed, size)
    if len(qp_np) == 0:
        return np.zeros((0, 3), np.int64)
    f = torch.as_tensor(faces, device=dev)
    pf = torch.as_tensor(pf_np, device=dev)
    qi, qj = torch.as_tensor(qp_np[:, 0], device=dev), torch.as_tensor(qp_np[:, 1], device=dev)
    Q = len(qp_np)
    tstep = max(1, int(PATCH_PAIR_TESTS // Q))
    rows = []
    for t0 in range(0, T, tstep):
        tri = verts[t0:t0 + tstep][:, f]                                   # (n,F,3,3)
        flo, fhi = tri.amin(2), tri.amax(2)                                # (n,F,3)
        plo, phi = flo[:, pf].amin(2), fhi[:, pf].amax(2)                  # (n,P,3)
        ov = ((plo[:, qi] <= phi[:, qj]) & (phi[:, qi] >= plo[:, qj])).all(-1)   # (n,Q)
        fr, q = torch.nonzero(ov, as_tuple=True)
        if fr.numel() == 0:
            continue
        cstep = max(1, max_face_pairs // (size * size))
        for c in range(0, fr.numel(), cstep):
            frc, qc = fr[c:c + cstep], q[c:c + cstep]
            a = pf[qi[qc]][:, :, None].expand(-1, size, size).reshape(-1)
            b = pf[qj[qc]][:, None, :].expand(-1, size, size).reshape(-1)
            frr = frc[:, None].expand(-1, size * size).reshape(-1)
            keep = ((flo[frr, a] <= fhi[frr, b]) & (fhi[frr, a] >= flo[frr, b])).all(-1)
            frr, a, b = frr[keep], a[keep], b[keep]
            if frr.numel() == 0:
                continue
            lo_, hi_ = torch.minimum(a, b), torch.maximum(a, b)
            F = len(faces)
            uk = torch.unique((frr * F + lo_) * F + hi_)
            frr, a, b = uk // (F * F), (uk // F) % F, uk % F
            shared = (f[a][:, :, None] == f[b][:, None, :]).any(dim=2).any(dim=1)
            frr, a, b = frr[~shared], a[~shared], b[~shared]
            hit = _sat_intersect_torch(tri[frr, a], tri[frr, b])
            rows.append(torch.stack([frr[hit] + t0, a[hit], b[hit]], dim=1).cpu().numpy())
    if not rows:
        return np.zeros((0, 3), np.int64)
    out = np.concatenate(rows)
    return np.unique(out, axis=0) if len(out) else out


def triangle_pairs(vertices, faces, backend: str = "auto", max_collisions: int = MAX_COLLISIONS,
                   face_segment=None, allowed=None) -> np.ndarray:
    """Intersecting non-local face pairs, (N, 3) rows (frame, lo, hi).

    backends: ``torch`` (default: plain torch on CUDA when available, else
    CPU; with ``face_segment``/``allowed`` the two-level patch broad phase
    :func:`triangle_pairs_patches`, otherwise a uniform-grid broad phase),
    ``bvh`` (torch-mesh-isect CUDA extension, as the paper code), ``numpy`` (per-frame
    reference implementation).  ``face_segment`` / ``allowed`` restrict the
    output to allowed segment pairs (all backends).
    """
    if backend in ("auto", "cpu"):
        backend = "torch"
    if backend == "torch":
        if face_segment is not None and allowed is not None:
            return triangle_pairs_patches(vertices, faces, face_segment, allowed)
        return triangle_pairs_torch(vertices, faces)
    if backend == "bvh":
        out = triangle_pairs_bvh(vertices, faces, max_collisions=max_collisions)
    elif backend == "numpy":
        v = vertices.detach().cpu().numpy() if hasattr(vertices, "detach") else np.asarray(vertices)
        rows = []
        for t in range(v.shape[0]):
            p = triangle_pairs_cpu(v[t], faces)
            if len(p):
                rows.append(np.column_stack([np.full(len(p), t, np.int64), p]))
        out = np.concatenate(rows) if rows else np.zeros((0, 3), np.int64)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    if face_segment is not None and allowed is not None and len(out):
        fs = np.asarray(face_segment)
        out = out[np.asarray(allowed)[fs[out[:, 1]], fs[out[:, 2]]]]
    return out


def _resolve_backend(backend: str, device: Optional[str] = None):
    """(backend, torch.device): ``auto``/``cpu`` -> torch; torch on CUDA when available, else CPU."""
    import torch
    if backend == "cpu":                      # torch kernels on the CPU
        backend, device = "torch", "cpu"
    elif backend == "auto":
        backend = "torch"
    if backend == "bvh":
        device = "cuda"
    elif backend == "numpy":
        device = "cpu"
    elif backend == "torch" and device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif backend != "torch":
        raise ValueError(f"unknown backend {backend!r}")
    return backend, torch.device(device)


# ---------------------------------------------------------------------------
# Penetration depth
# ---------------------------------------------------------------------------
def vertex_normals_and_areas(verts, faces):
    """Area-weighted vertex normals (T,V,3) and vertex areas (T,V) (1/3 of incident faces)."""
    import torch
    f = torch.as_tensor(np.asarray(faces), dtype=torch.long, device=verts.device)
    tri = verts[:, f]                                                     # (T,F,3,3)
    fn = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)  # 2*area*n
    T, V = verts.shape[0], verts.shape[1]
    vn = torch.zeros((T, V, 3), dtype=verts.dtype, device=verts.device)
    area = torch.zeros((T, V), dtype=verts.dtype, device=verts.device)
    fa = fn.norm(dim=-1) / 2.0
    for k in range(3):
        vn.index_add_(1, f[:, k], fn)
        area.index_add_(1, f[:, k], fa / 3.0)
    vn = vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vn, area, fa


def winding_and_distance(points, triangles, max_elements: int = WINDING_CHUNK_ELEMENTS):
    """Generalised winding number and unsigned distance of points w.r.t. a triangle soup.

    points: (n, Q, 3); triangles: (n, F, 3, 3) (outward-oriented faces).
    Winding number (Jacobson et al. 2013) = sum of signed solid angles / 4 pi
    (Van Oosterom & Strackee formula); it is ~1 inside and ~0 outside even for
    an open surface such as a body segment cut at its joints.  Distance is
    the exact point-triangle distance.  Returns (n, Q) winding, (n, Q) distance.
    """
    import torch
    n, Q = points.shape[0], points.shape[1]
    F = triangles.shape[1]
    w_out = torch.empty((n, Q), dtype=points.dtype, device=points.device)
    d_out = torch.empty((n, Q), dtype=points.dtype, device=points.device)
    qstep = max(1, int(max_elements // max(1, n * F)))
    a0, b0, c0 = triangles[:, None, :, 0], triangles[:, None, :, 1], triangles[:, None, :, 2]  # (n,1,F,3)
    ab, ac = b0 - a0, c0 - a0
    normal = torch.cross(ab, ac, dim=-1)
    nn2 = (normal * normal).sum(-1).clamp_min(1e-30)
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    for s in range(0, Q, qstep):
        q = points[:, s:s + qstep, None, :]                                  # (n,q,1,3)
        a, b, c = a0 - q, b0 - q, c0 - q
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        det = (a * torch.cross(b, c, dim=-1)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        w_out[:, s:s + qstep] = (2.0 * torch.atan2(det, den)).sum(-1) / (4.0 * math.pi)
        ap = -a                                                              # q - A
        d20, d21 = (ap * ab).sum(-1), (ap * ac).sum(-1)
        u = (d11 * d20 - d01 * d21) / nn2
        v = (d00 * d21 - d01 * d20) / nn2
        plane = (ap * normal).sum(-1) ** 2 / nn2
        inside_tri = (u >= 0) & (v >= 0) & (u + v <= 1)
        dist2 = torch.where(inside_tri, plane, torch.full_like(plane, float("inf")))
        for p0, p1 in ((a0, b0), (b0, c0), (c0, a0)):
            e = p1 - p0
            off = q - p0
            t = ((off * e).sum(-1) / (e * e).sum(-1).clamp_min(1e-30)).clamp(0.0, 1.0)
            r = off - t[..., None] * e
            dist2 = torch.minimum(dist2, (r * r).sum(-1))
        d_out[:, s:s + qstep] = dist2.amin(dim=-1).clamp_min(0).sqrt()
    return w_out, d_out


def _penetrating_patch(verts, vareas, idx_a, faces_b, seeds_a, adj_a):
    """Depth / area of the part of segment A lying inside segment B.

    verts: (n, V, 3); vareas: (n, V); idx_a: vertex index tensor of A;
    faces_b: (F_B, 3) face tensor of B (kept numbering);
    seeds_a: (n, |A|) bool, vertices of A's triangles that intersect B;
    adj_a: sparse (|A|, |A|) adjacency of A's vertices.
    A vertex of A is inside B when its winding number w.r.t. B's faces
    exceeds 1/2; the penetrating patch is the inside set connected to the
    seeds; depth = largest distance from the patch to B's surface.
    Returns depth (n,), area (n,), patch size (n,).
    """
    import torch
    n = verts.shape[0]
    xa = verts[:, idx_a]
    # Only vertices inside B's bounding box can be inside B: evaluate the
    # winding number for the union (over frames) of those vertices only.
    vb = verts[:, torch.unique(faces_b.reshape(-1))]
    in_box = ((xa >= vb.amin(1, keepdim=True)) & (xa <= vb.amax(1, keepdim=True))).all(-1)   # (n,|A|)
    cols = torch.nonzero(in_box.any(0), as_tuple=True)[0]
    wind = torch.zeros((n, int(idx_a.numel())), dtype=verts.dtype, device=verts.device)
    dist = torch.zeros_like(wind)
    if cols.numel():
        fstep = max(1, WINDING_CHUNK_ELEMENTS // max(1, int(faces_b.shape[0]) * 16))
        for s in range(0, n, fstep):
            vs = verts[s:s + fstep]
            w, d = winding_and_distance(xa[s:s + fstep][:, cols], vs[:, faces_b])
            wind[s:s + fstep, cols], dist[s:s + fstep, cols] = w, d
    inside = (wind > 0.5) & in_box
    sd = torch.where(inside, -dist, dist)
    # grow the seeds one edge ring at a time through inside vertices; the convergence test
    # needs a device sync, so it runs only every 8th ring
    region = seeds_a & inside
    for ring in range(MAX_GROWTH_RINGS):
        grown = torch.sparse.mm(adj_a, region.T.float()).T > 0
        new = region | (grown & inside)
        if ring % 8 == 7 and bool((new == region).all()):
            region = new
            break
        region = new
    depth = torch.where(region, -sd, torch.zeros_like(sd)).amax(dim=1)
    area = (vareas[:, idx_a] * region).sum(dim=1)
    return depth, area, region.sum(dim=1)


def _sparse_adjacency(edges_local: np.ndarray, size: int, device):
    """Symmetric sparse (size, size) vertex adjacency from undirected edges."""
    import torch
    if len(edges_local) == 0:
        return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0),
                                       (size, size), device=device).coalesce()
    e = torch.as_tensor(edges_local.T, dtype=torch.long)
    ind = torch.cat([e, e.flip(0)], dim=1)
    return torch.sparse_coo_tensor(ind, torch.ones(ind.shape[1]), (size, size)).coalesce().to(device)


# ---------------------------------------------------------------------------
# Measurement of a mesh sequence
# ---------------------------------------------------------------------------
def measure_mesh_sequence(verts_full, topo: BodyTopology, *, backend: str = "auto",
                          with_full_mesh: bool = True, max_collisions: int = MAX_COLLISIONS,
                          timings: Optional[dict] = None,
                          nonadjacent_only: bool = False,
                          device: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Per-frame collision measurements of a (T, V_full, 3) vertex sequence (metres).

    Returns a dict of numpy arrays (all length T unless noted):

    * ``full_mesh_pairs``/``full_mesh_faces`` (only with ``with_full_mesh``): pairs and
      collided faces on every face of the full mesh (incl. eyeballs), as the paper counts.
    * ``body_pairs``/``body_faces``: eyeballs removed, all other pairs.
    * ``na_pairs``/``na_faces``/``na_face_area``: non-adjacent segment pairs
      only (collided faces and their total area, m^2).
    * ``depth``: max penetration depth over colliding non-adjacent segment
      pairs (m); ``inside_area``: area of the penetrating patches (m^2);
      ``deepest_pair``: index into ``segment_pairs`` (-1 if none);
      ``segment_pair_depth`` (P, T): depth per colliding segment pair and frame.
    * ``body_area``: total kept surface area per frame (m^2).
    * scalars ``num_faces_full``, ``num_faces_body``; ``segment_pairs`` (P,2).

    With ``nonadjacent_only`` (and without ``with_full_mesh``) only non-adjacent
    segment pairs are tested (the fast path used for scoring); ``body_pairs`` /
    ``body_faces`` then equal the non-adjacent counts.  ``timings`` (optional dict)
    accumulates ``pairs_s`` and ``depth_s``.
    """
    import torch
    timings = {} if timings is None else timings
    verts_full = verts_full if torch.is_tensor(verts_full) else torch.as_tensor(np.asarray(verts_full))
    if verts_full.ndim != 3 or verts_full.shape[-1] != 3:
        raise ValueError("vertices must be (T, V, 3)")
    if verts_full.shape[1] != topo.num_full_vertices:
        raise ValueError("vertex count does not match topology")
    if not torch.isfinite(verts_full).all():
        raise ValueError("non-finite vertices")
    backend, device = _resolve_backend(backend, device)
    verts_full = verts_full.to(device, torch.float32)
    T = int(verts_full.shape[0])
    F_full, F = len(topo.faces_full), len(topo.faces)
    out: Dict[str, np.ndarray] = {"num_faces_full": np.int64(F_full), "num_faces_body": np.int64(F)}

    # 1. intersecting triangle pairs, (frame, face, face) rows in the kept numbering
    t0 = time.perf_counter()
    if with_full_mesh:
        # test the full mesh (as the paper counts), then map the pairs to the kept faces
        pairs_full = triangle_pairs(verts_full, topo.faces_full, backend, max_collisions)
        out["full_mesh_pairs"] = np.bincount(pairs_full[:, 0], minlength=T)
        out["full_mesh_faces"] = _face_counts(pairs_full, T, F_full)
        to_body = np.full(F_full, -1, np.int64)
        to_body[topo.face_full_index] = np.arange(F)
        a, b = to_body[pairs_full[:, 1]], to_body[pairs_full[:, 2]]
        ok = (a >= 0) & (b >= 0)
        pairs = np.column_stack([pairs_full[ok, 0], np.minimum(a[ok], b[ok]), np.maximum(a[ok], b[ok])])
    elif nonadjacent_only:
        # scoring path: only non-adjacent segment pairs are tested exactly
        keep_t = torch.as_tensor(topo.keep_vertices, device=device)
        pairs = triangle_pairs(verts_full[:, keep_t], topo.faces, backend, max_collisions,
                               face_segment=topo.face_segment, allowed=~topo.adjacent)
    else:
        keep_t = torch.as_tensor(topo.keep_vertices, device=device)
        pairs = triangle_pairs(verts_full[:, keep_t], topo.faces, backend, max_collisions)
    timings["pairs_s"] = timings.get("pairs_s", 0.0) + time.perf_counter() - t0

    # 2. per-frame pair / face counts; keep the pairs between non-adjacent segments
    t0 = time.perf_counter()
    out["body_pairs"] = np.bincount(pairs[:, 0], minlength=T) if len(pairs) else np.zeros(T, np.int64)
    out["body_faces"] = _face_counts(pairs, T, F)
    sa, sb = topo.face_segment[pairs[:, 1]], topo.face_segment[pairs[:, 2]]
    na = ~topo.adjacent[sa, sb]
    pna = pairs[na]
    sa, sb = sa[na], sb[na]
    out["na_pairs"] = np.bincount(pna[:, 0], minlength=T) if len(pna) else np.zeros(T, np.int64)
    out["na_faces"] = _face_counts(pna, T, F)

    keep_t = torch.as_tensor(topo.keep_vertices, device=device)
    verts = verts_full[:, keep_t]
    normals, vareas, fareas = vertex_normals_and_areas(verts, topo.faces)
    out["body_area"] = fareas.sum(dim=1).double().cpu().numpy()
    fa_np = fareas.double().cpu().numpy()
    out["na_face_area"] = _face_area(pna, T, fa_np)

    # 3. penetration depth per colliding segment pair (both directions), max over pairs per frame
    depth = np.zeros(T)
    inside_area = np.zeros(T)
    deepest = np.full(T, -1, np.int64)
    seg_pairs = (np.unique(np.stack([np.minimum(sa, sb), np.maximum(sa, sb)], 1), axis=0) if len(pna)
                 else np.zeros((0, 2), np.int64))
    pair_depth_max = np.zeros(len(seg_pairs))
    pair_depth = np.zeros((len(seg_pairs), T))
    pair_frames = np.zeros(len(seg_pairs), np.int64)
    for k, (s1, s2) in enumerate(seg_pairs):
        sel = ((sa == s1) & (sb == s2)) | ((sa == s2) & (sb == s1))
        rows = pna[sel]
        frames = np.unique(rows[:, 0])
        fidx = np.searchsorted(frames, rows[:, 0])
        d_tot = np.zeros(len(frames))
        a_tot = np.zeros(len(frames))
        for sA, sB in ((s1, s2), (s2, s1)):
            idx_a = np.flatnonzero(topo.vertex_segment == sA)
            idx_b = np.flatnonzero(topo.vertex_segment == sB)
            if not idx_a.size or not idx_b.size:
                continue
            local = np.full(len(topo.vertex_segment), -1, np.int64)
            local[idx_a] = np.arange(idx_a.size)
            # seed vertices: vertices of A-side faces of intersecting pairs
            fa_side = np.where(topo.face_segment[rows[:, 1]] == sA, rows[:, 1], rows[:, 2])
            sv = topo.faces[fa_side]                                    # (n_rows, 3)
            seeds = np.zeros((len(frames), idx_a.size), bool)
            lv = local[sv]
            ok = lv >= 0
            seeds[np.repeat(fidx, 3)[ok.ravel()], lv.ravel()[ok.ravel()]] = True
            e = topo.edges
            ein = (local[e[:, 0]] >= 0) & (local[e[:, 1]] >= 0)
            adj = _sparse_adjacency(local[e[ein]], idx_a.size, device)
            ft = torch.as_tensor(frames, device=device)
            faces_b = torch.as_tensor(topo.faces[topo.face_segment == sB], device=device)
            d, ar, _ = _penetrating_patch(verts[ft], vareas[ft],
                                          torch.as_tensor(idx_a, device=device), faces_b,
                                          torch.as_tensor(seeds, device=device), adj)
            d_tot = np.maximum(d_tot, d.double().cpu().numpy())
            a_tot = a_tot + ar.double().cpu().numpy()
        better = d_tot > depth[frames]
        deepest[frames[better]] = k
        depth[frames] = np.maximum(depth[frames], d_tot)
        inside_area[frames] += a_tot
        pair_depth_max[k] = d_tot.max() if d_tot.size else 0.0
        pair_depth[k, frames] = d_tot
        pair_frames[k] = len(frames)
    timings["depth_s"] = timings.get("depth_s", 0.0) + time.perf_counter() - t0
    out.update(depth=depth, inside_area=inside_area, deepest_pair=deepest,
               segment_pairs=seg_pairs.astype(np.int64), segment_pair_max_depth=pair_depth_max,
               segment_pair_frames=pair_frames, segment_pair_depth=pair_depth, num_frames=np.int64(T),
               segment_names=np.asarray(topo.segment_names))
    return out


def _face_counts(pairs: np.ndarray, T: int, F: int) -> np.ndarray:
    """Per frame, the number of distinct faces in any (frame, a, b) pair."""
    if not len(pairs):
        return np.zeros(T, np.int64)
    keys = np.unique(np.concatenate([pairs[:, 0] * F + pairs[:, 1], pairs[:, 0] * F + pairs[:, 2]]))
    return np.bincount(keys // F, minlength=T)


def _face_area(pairs: np.ndarray, T: int, face_area: np.ndarray) -> np.ndarray:
    """Per frame, the total area of the distinct faces in any pair (``face_area`` (T, F))."""
    if not len(pairs):
        return np.zeros(T)
    F = face_area.shape[1]
    keys = np.unique(np.concatenate([pairs[:, 0] * F + pairs[:, 1], pairs[:, 0] * F + pairs[:, 2]]))
    fr, fi = keys // F, keys % F
    return np.bincount(fr, weights=face_area[fr, fi], minlength=T)


# ---------------------------------------------------------------------------
# Scoring (pure numpy)
# ---------------------------------------------------------------------------
def frame_severity(depth, config: CollisionConfig = DEFAULT_CONFIG):
    """Per-frame severity m_t, flag b_t and severe flag from penetration depth (m)."""
    config.validate()
    depth = np.nan_to_num(np.asarray(depth, dtype=float), nan=0.0)
    m = ramp(depth, config.contact_depth_m, config.severe_depth_m)
    flags = depth > config.contact_depth_m
    severe = depth >= config.severe_depth_m
    return m, flags, severe


def score_depth(depth, config: CollisionConfig = DEFAULT_CONFIG, valid=None) -> dict:
    """0-100 score from per-frame penetration depth (paper aggregation)."""
    m, flags, severe = frame_severity(depth, config)
    agg = aggregate(m, flags=flags, valid=valid, weights=config.weights)
    v = np.ones_like(flags) if valid is None else np.asarray(valid, bool)
    r_severe = float(severe[v].mean()) if v.any() else float("nan")
    w = np.asarray(config.weights, float) / float(np.sum(config.weights))
    D = float(np.clip(w[0] * r_severe + w[1] * agg.s + w[2] * agg.p, 0.0, 1.0)) if v.any() else float("nan")
    return {"score": 100.0 * (1.0 - D), "D": D, "r_severe": r_severe, "r_flagged": agg.r,
            "s": agg.s, "p": agg.p, "num_frames": agg.num_frames, "num_flagged": agg.num_flagged,
            "longest_run": agg.longest_run, "coverage": agg.coverage}


def score_measurements(meas: Dict[str, np.ndarray], config: CollisionConfig = DEFAULT_CONFIG,
                       frame_indices: Optional[np.ndarray] = None) -> dict:
    """Score + diagnostics from :func:`measure_mesh_sequence` output (CPU only).

    ``frame_indices`` optionally restricts the score to a subset of the frames.
    """
    depth = np.asarray(meas["depth"], float)
    deepest_all = np.asarray(meas.get("deepest_pair", np.full(depth.size, -1)))
    sel = slice(None) if frame_indices is None else np.asarray(frame_indices)
    out = score_depth(depth[sel], config)
    seg_pairs = np.asarray(meas.get("segment_pairs", np.zeros((0, 2))), np.int64)
    seg_names = [str(x) for x in meas.get("segment_names", SEGMENTS)]
    names = [f"{seg_names[a]}|{seg_names[b]}" for a, b in seg_pairs] if seg_pairs.size else []
    deepest = deepest_all[sel]
    flagged = depth[sel] > config.contact_depth_m
    culprit = {}
    for k in np.unique(deepest[flagged]):
        if k >= 0:
            culprit[names[k]] = int(np.sum(deepest[flagged] == k))
    out.update({
        "config": asdict(config),
        "max_depth_m": float(depth[sel].max()) if depth.size else 0.0,
        "p95_depth_m": float(np.percentile(depth[sel], 95)) if depth.size else 0.0,
        "frames_with_nonadjacent_intersection": int(np.sum(np.asarray(meas["na_pairs"])[sel] > 0)),
        "flagged_frames_by_segment_pair": culprit,
        "segment_pair_max_depth_m": {n: float(d) for n, d in zip(names, meas.get("segment_pair_max_depth", []))},
    })
    return out


# ---------------------------------------------------------------------------
# GVHMR / SMPL-X front end
# ---------------------------------------------------------------------------
def load_gvhmr_params(pt_path: Path | str, use_global: bool = True) -> Dict[str, np.ndarray]:
    """body_pose (T,63), betas (T,10), global_orient (T,3), transl (T,3) from hmr4d_results.pt."""
    import torch
    # written by the pipeline's own GVHMR stage; it holds non-tensor objects, so weights_only=False
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    key = "smpl_params_global" if use_global else "smpl_params_incam"
    params = data[key]
    out = {}
    for name in ("body_pose", "betas", "global_orient", "transl"):
        t = torch.as_tensor(params[name]).float()
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        out[name] = t.numpy()
    T = out["body_pose"].shape[0]
    for name in ("betas", "global_orient", "transl"):  # one shape / pose for all frames -> repeat it
        if out[name].shape[0] != T:
            out[name] = np.broadcast_to(out[name][:1], (T, out[name].shape[1])).copy()
    return out


def smplx_vertices(params: Dict[str, np.ndarray], model_path: Optional[Path | str] = None,
                   device: str = "cuda", chunk: int = 512):
    """SMPL-X vertices (T, 10475, 3) for GVHMR body pose and shape.

    Global orientation and translation are set to zero: self-collision is invariant to
    a rigid motion of the whole body.  Jaw, eyes and expression are neutral and the
    hands take SMPL-X's learned mean pose (GVHMR does not estimate them).
    """
    import torch
    T = int(params["body_pose"].shape[0])
    dev = torch.device(device)
    model = _smplx_model(model_path, int(params["betas"].shape[1]), str(dev))
    out = []
    with torch.no_grad():
        for s in range(0, T, chunk):
            sl = slice(s, min(T, s + chunk))
            n = sl.stop - sl.start
            z3 = torch.zeros((n, 3), device=dev)
            kw = dict(
                body_pose=torch.as_tensor(params["body_pose"][sl], device=dev).float(),
                betas=torch.as_tensor(params["betas"][sl], device=dev).float(),
                global_orient=z3,
                transl=z3,
                jaw_pose=z3, leye_pose=z3, reye_pose=z3,
                left_hand_pose=torch.zeros((n, 45), device=dev), right_hand_pose=torch.zeros((n, 45), device=dev),
                expression=torch.zeros((n, model.num_expression_coeffs), device=dev),
            )
            out.append(model(**kw).vertices)
    return torch.cat(out, dim=0)


_TOPOLOGY_CACHE: Dict[tuple, BodyTopology] = {}
_MODEL_CACHE: Dict[tuple, object] = {}


def _smplx_model(model_path, num_betas: int, device: str):
    """One SMPL-X layer per (file, betas, device) per process (loading the npz costs ~1 s)."""
    model_path = resolve_smplx_model(model_path)
    key = (str(model_path), num_betas, device)
    if key not in _MODEL_CACHE:
        import smplx
        _MODEL_CACHE[key] = smplx.create(model_path=str(model_path), model_type="smplx", use_pca=False,
                                         flat_hand_mean=False, batch_size=1, num_betas=num_betas,
                                         ext="npz").to(device).eval()
    return _MODEL_CACHE[key]


def cached_smplx_topology(model_path: Optional[Path | str] = None,
                          exclude_fingers: Optional[bool] = None) -> BodyTopology:
    """:func:`smplx_topology`, computed once per (model file, ``exclude_fingers``) per process."""
    exclude_fingers = EXCLUDE_FINGERS if exclude_fingers is None else exclude_fingers
    model_path = resolve_smplx_model(model_path)
    key = (str(model_path), bool(exclude_fingers))
    if key not in _TOPOLOGY_CACHE:
        _TOPOLOGY_CACHE[key] = smplx_topology(model_path, exclude_fingers)
    return _TOPOLOGY_CACHE[key]


def measure_params(params: Dict[str, np.ndarray], *, model_path: Optional[Path | str] = None,
                   backend: str = "auto", with_full_mesh: bool = False,
                   exclude_fingers: Optional[bool] = None,
                   timings: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """GVHMR parameters -> per-frame measurements (see :func:`measure_mesh_sequence`)."""
    timings = {} if timings is None else timings
    backend, device = _resolve_backend(backend)
    t0 = time.perf_counter()
    model_path = resolve_smplx_model(model_path)
    topo = cached_smplx_topology(model_path, exclude_fingers)
    verts = smplx_vertices(params, model_path, device=str(device))
    timings["mesh_s"] = timings.get("mesh_s", 0.0) + time.perf_counter() - t0
    return measure_mesh_sequence(verts, topo, backend=backend, with_full_mesh=with_full_mesh, timings=timings,
                                 nonadjacent_only=not with_full_mesh, device=str(device))


def find_gvhmr_result(session_dir: Path | str) -> Path:
    """``<S>/gvhmr/<stem>/hmr4d_results.pt`` of a session (the first one if several)."""
    session = Path(session_dir)
    hits = sorted(session.glob("gvhmr/*/hmr4d_results.pt"))
    if not hits:
        raise FileNotFoundError(f"no gvhmr/*/hmr4d_results.pt under {session}")
    return hits[0]


def score_session(session_dir: Path | str, *, config: CollisionConfig = DEFAULT_CONFIG,
                  model_path: Optional[Path | str] = None, backend: str = "auto",
                  write: bool = True) -> dict:
    """Score ``<S>/gvhmr/<stem>/hmr4d_results.pt``; with ``write`` also writes
    ``<S>/metrics/self_collision/<stem>_self_collision.json`` (+ per-frame npz)."""
    session = Path(session_dir)
    pt = find_gvhmr_result(session)
    stem = pt.parent.name
    timings: dict = {}
    t0 = time.perf_counter()
    meas = measure_params(load_gvhmr_params(pt), model_path=model_path, backend=backend, timings=timings)
    result = score_measurements(meas, config)
    timings["total_s"] = time.perf_counter() - t0
    result.update({"self_collision_score": result["score"], "input": str(pt), "timings": timings,
                   "num_frames": int(meas["num_frames"])})
    if write:
        out_dir = session / "metrics" / "self_collision"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}_self_collision.json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(out_dir / f"{stem}_self_collision_frames.npz", **meas)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: score a session or one GVHMR result and print the scalar results."""
    ap = argparse.ArgumentParser(description="HumanScore self-collision (non-adjacent penetration depth)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session-dir", help="session directory with gvhmr/<stem>/hmr4d_results.pt")
    g.add_argument("--input-pt", help="a GVHMR hmr4d_results.pt")
    ap.add_argument("--backend", default="auto", choices=("auto", "torch", "bvh", "cpu", "numpy"),
                    help="triangle test: auto/torch (default; CUDA when available), cpu, bvh (torch-mesh-isect), "
                         "numpy (reference)")
    ap.add_argument("--model", default=None,
                    help=f"{SMPLX_MODEL_NAME} (default: $HMB_SMPLX_MODEL, $HMB_SMPLX_MODEL_DIR, then the install "
                         "locations: GVHMR body models, outputs/assets/smplx)")
    ap.add_argument("--no-write", action="store_true", help="do not write metrics/self_collision/ into the session")
    args = ap.parse_args(argv)
    if args.session_dir:
        result = score_session(args.session_dir, model_path=args.model, backend=args.backend,
                               write=not args.no_write)
    else:
        meas = measure_params(load_gvhmr_params(args.input_pt), model_path=args.model, backend=args.backend)
        result = score_measurements(meas)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
