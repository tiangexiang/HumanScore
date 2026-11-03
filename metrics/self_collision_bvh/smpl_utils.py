# -*- coding: utf-8 -*-
"""SMPL-X body meshes from reconstruction results, for metric (IV) self-collision (paper implementation).

:func:`load_smpl_sequence` runs the SMPL-X body model on the per-frame parameters of a
GVHMR ``hmr4d_results.pt`` (or a PromptHMR-style ``results.pkl``) and returns the vertices
(T, V, 3) and faces (F, 3).  The licensed ``SMPLX_NEUTRAL.npz`` is found by
:func:`resolve_smplx_model_path` (environment variables first, then the locations the
HumanScore installer uses).
"""
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

try:
    import joblib
except ImportError:
    joblib = None


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_smplx_model_path() -> Path:
    """Path of ``SMPLX_NEUTRAL.npz``, independent of the working directory.

    Order: ``$HMB_SMPLX_MODEL`` (the file), ``$HMB_SMPLX_MODEL_DIR`` (a folder holding it,
    directly or in ``smplx/``), then ``outputs/assets/smplx/``, the GVHMR body-model folder
    and two ``inputs/`` folders of this repository.  Raises ``FileNotFoundError`` listing
    the places checked.
    """
    override = os.environ.get("HMB_SMPLX_MODEL", "")
    model_dir = os.environ.get("HMB_SMPLX_MODEL_DIR", "")
    if not override and model_dir:
        model_dir_path = Path(model_dir).expanduser()
        direct = model_dir_path / "SMPLX_NEUTRAL.npz"
        override = str(direct if direct.is_file() else model_dir_path / "smplx/SMPLX_NEUTRAL.npz")
    candidates = [Path(override).expanduser()] if override else [
        REPO_ROOT / "outputs/assets/smplx/SMPLX_NEUTRAL.npz",
        REPO_ROOT / "GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
        REPO_ROOT / "inputs/models/smplx/SMPLX_NEUTRAL.npz",
        REPO_ROOT / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
        Path(__file__).resolve().parent / "inputs/models/smplx/SMPLX_NEUTRAL.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "SMPLX_NEUTRAL.npz is required. Set HMB_SMPLX_MODEL to your licensed local model. "
        "Checked: " + ", ".join(str(candidate) for candidate in candidates)
    )


def _to_tensor(x, device):
    """float32 tensor on ``device``."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32, device=device)


def _ensure_LD(name: str, t: torch.Tensor, L: int) -> torch.Tensor:
    """
    Ensure that the tensor has shape (L, D):
    - (L, D): returned as-is
    - (1, D): repeated to (L, D)
    - (D,):   expanded to (L, D)
    """
    if t.ndim == 2 and t.shape[0] == L:
        return t
    if t.ndim == 2 and t.shape[0] == 1:
        return t.expand(L, -1)
    if t.ndim == 1:
        return t.unsqueeze(0).expand(L, -1)
    raise ValueError(f"{name} must be (L,D) or (D,), got {tuple(t.shape)}")


def _run_smplx_forward(
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    device: torch.device,
    to_smpl: bool,
    additional_pose: Dict[str, torch.Tensor] | None = None,
    return_on_cpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run SMPL-X forward and optionally map to SMPL topology.
    All inputs on device, shape (L, ...). Returns verts (L,V,3), faces (F,3).
    Outputs stay on ``device`` when return_on_cpu=False for direct BVH use.
    """
    import smplx

    L = body_pose.shape[0]
    body_model = smplx.create(
        model_path=str(resolve_smplx_model_path()), model_type="smplx",
        # GVHMR's supermotion model uses the learned neutral hand mean. A
        # PromptHMR full pose already contains all 165 absolute pose entries.
        use_pca=False, flat_hand_mean=additional_pose is not None, batch_size=int(L),
        num_betas=int(betas.shape[-1]), ext="npz",
    ).to(device)
    body_model.eval()

    with torch.no_grad():
        out = body_model(
            body_pose=body_pose,
            betas=betas,
            global_orient=global_orient,
            transl=transl,
            **(additional_pose or {}),
        )
        verts = out.vertices  # (L, Vx, 3)
        faces = getattr(body_model, "faces", None)

    if faces is None:
        raise RuntimeError("SMPL-X model has no faces attribute; cannot run BVH.")

    if to_smpl:
        try:
            map_path = REPO_ROOT / "GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt"
            smplx2smpl = torch.load(map_path, map_location=device)
            if hasattr(smplx2smpl, "to_dense"):
                smplx2smpl = smplx2smpl.to_dense()
            if not isinstance(smplx2smpl, torch.Tensor) or smplx2smpl.ndim != 2:
                raise ValueError("smplx2smpl_sparse.pt has wrong format.")
            verts = torch.einsum("mv, tvj -> tmj", smplx2smpl.to(device), verts)
            smpl_model = smplx.create(
                model_path=str(REPO_ROOT / "GVHMR/inputs/checkpoints/body_models"),
                model_type="smpl", gender="neutral",
            )
            faces = torch.as_tensor(
                getattr(smpl_model, "faces", faces),
                dtype=torch.long,
                device=device,
            )
        except Exception as e:
            raise RuntimeError("Requested SMPL-X to SMPL mapping failed.") from e

    output_device = torch.device("cpu") if return_on_cpu else device
    verts = verts.detach().to(output_device, dtype=torch.float32)
    faces = torch.as_tensor(faces, dtype=torch.long, device=output_device)
    return verts, faces


def _load_pkl_smpl_params(
    result_path: str,
    person_id: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Load SMPL-X parameters from a PromptHMR-style results.pkl.

    Expected pkl structure::
        {
            "people": {
                <person_id>: {
                    "smplx_world": {
                        "pose": (T, 165),   # axis-angle, 55 joints * 3 (global_orient 3 + body 63 + ...)
                        "shape": (T, 10),
                        "trans": (T, 3)
                    }
                }
            }
        }

    Returns:
        body_pose (L, 63), betas (L, 10), global_orient (L, 3), transl (L, 3) as numpy/torch on CPU.
    """
    if joblib is None:
        raise RuntimeError("joblib is required to load .pkl files. Install with: pip install joblib")

    with open(result_path, "rb") as f:
        data = joblib.load(f)

    if "people" not in data or not data["people"]:
        raise KeyError(f"pkl must have non-empty 'people', got keys: {list(data.keys())}")

    if person_id is None:
        person_id = next(iter(data["people"].keys()))
    if person_id not in data["people"]:
        raise KeyError(f"person_id {person_id} not in people {list(data['people'].keys())}")

    person = data["people"][person_id]
    if "smplx_world" not in person:
        raise KeyError(f"smplx_world not in people[{person_id}]. Keys: {list(person.keys())}")

    world = person["smplx_world"]
    pose = world["pose"]   # (T, 165)
    shape = world["shape"] # (T, 10)
    trans = world["trans"] # (T, 3)

    # SMPL-X full pose 165 = 55*3: global_orient (3) + body_pose (21*3=63) + ...
    if pose.shape[1] != 165:
        raise ValueError(
            f"smplx_world.pose must be (T, 165), got {pose.shape}. "
            "Other pkl layouts are not supported."
        )
    global_orient = pose[:, 0:3]    # (T, 3)
    body_pose = pose[:, 3:66]      # (T, 63)
    betas = shape
    transl = trans

    # Convert to torch float32
    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32)

    additional_pose = {
        "jaw_pose": to_t(pose[:, 66:69]),
        "leye_pose": to_t(pose[:, 69:72]),
        "reye_pose": to_t(pose[:, 72:75]),
        "left_hand_pose": to_t(pose[:, 75:120]),
        "right_hand_pose": to_t(pose[:, 120:165]),
    }
    return to_t(body_pose), to_t(betas), to_t(global_orient), to_t(transl), additional_pose


def load_smpl_sequence_from_pkl(
    result_path: str,
    person_id: Any = None,
    use_cuda: bool = True,
    to_smpl: bool = False,
    return_on_cpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover per-frame SMPL-X vertices and faces from a PromptHMR-style results.pkl.

    Uses ``smplx_world`` (world-space axis-angle pose) for the first person in
    ``people`` (or the given person_id). pose (T, 165) is split into
    global_orient (T, 3) and body_pose (T, 63).

    Returns:
        verts: (T, V, 3) float32 (on the CPU unless ``return_on_cpu=False``)
        faces: (F, 3) long
    """
    assert os.path.isfile(result_path), f"File not found: {result_path}"
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")

    body_pose, betas, global_orient, transl, additional_pose = _load_pkl_smpl_params(result_path, person_id=person_id)
    body_pose = body_pose.to(device)
    betas = betas.to(device)
    global_orient = global_orient.to(device)
    transl = transl.to(device)

    L = body_pose.shape[0]
    betas = _ensure_LD("betas", betas, L)
    # global_orient, transl already (L, 3)

    return _run_smplx_forward(
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        transl=transl,
        device=device,
        to_smpl=to_smpl,
        additional_pose={name: value.to(device) for name, value in additional_pose.items()},
        return_on_cpu=return_on_cpu,
    )


def load_smpl_sequence(
    result_path: str,
    use_global: bool = True,
    use_cuda: bool = True,
    to_smpl: bool = False,
    person_id: Any = None,
    return_on_cpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover per-frame SMPL-X/SMPL vertices and faces from a result file.

    Supports:
    - ``.pt``: GVHMR hmr4d_results.pt with ``smpl_params_global`` / ``smpl_params_incam``.
    - ``.pkl``: PromptHMR-style results.pkl with ``people[].smplx_world`` (pose/shape/trans).

    Args:
        result_path: path to .pt or .pkl file
        use_global:  for .pt only: use smpl_params_global (True) or smpl_params_incam (False)
        use_cuda:    whether to run SMPL-X on CUDA
        to_smpl:     if True, try to map to SMPL topology
        person_id:   for .pkl only: which person to use (default: first key in people)
        return_on_cpu: return CPU tensors (default); False keeps them on the model's device,
            which avoids a GPU-to-host round trip when the BVH runs next.

    Returns:
        verts: (T, V, 3) float32
        faces: (F, 3) long
    """
    assert os.path.isfile(result_path), f"File not found: {result_path}"

    if result_path.lower().endswith(".pkl"):
        return load_smpl_sequence_from_pkl(
            result_path=result_path,
            person_id=person_id,
            use_cuda=use_cuda,
            to_smpl=to_smpl,
            return_on_cpu=return_on_cpu,
        )

    # .pt path
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")
    data: Dict = torch.load(result_path, map_location=device)

    key = "smpl_params_global" if use_global else "smpl_params_incam"
    if key not in data:
        raise KeyError(f"{key} not in {result_path}. Keys: {list(data.keys())}")
    smpl_params = data[key]
    if not isinstance(smpl_params, dict):
        raise TypeError(f"{key} must be a dict, got {type(smpl_params)}")

    required = ["body_pose", "betas", "global_orient", "transl"]
    for r in required:
        if r not in smpl_params:
            raise KeyError(f"{r} not found in {key} of {result_path}")

    body_pose = _to_tensor(smpl_params["body_pose"], device)
    betas = _to_tensor(smpl_params["betas"], device)
    global_orient = _to_tensor(smpl_params["global_orient"], device)
    transl = _to_tensor(smpl_params["transl"], device)

    # Remove a possible batch dimension (1, L, D)
    def drop_batch(t):
        return t.squeeze(0) if t.ndim == 3 and t.shape[0] == 1 else t

    body_pose, betas, global_orient, transl = (drop_batch(t) for t in (body_pose, betas, global_orient, transl))

    if body_pose.ndim != 2:
        raise ValueError(f"body_pose must be (L,63), got {tuple(body_pose.shape)}")

    L = body_pose.shape[0]
    betas = _ensure_LD("betas", betas, L)
    global_orient = _ensure_LD("global_orient", global_orient, L)
    transl = _ensure_LD("transl", transl, L)

    return _run_smplx_forward(
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        transl=transl,
        device=device,
        to_smpl=to_smpl,
        return_on_cpu=return_on_cpu,
    )
