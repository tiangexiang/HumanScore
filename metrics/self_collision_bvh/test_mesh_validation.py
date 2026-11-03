"""Optional native tests, runnable in an environment with the CUDA extension."""

import pytest


@pytest.fixture
def native_metric():
    torch = pytest.importorskip("torch")
    from metrics.self_collision_bvh import bvh_metric
    if bvh_metric.BVH is None:
        pytest.skip("torch-mesh-isect is not installed")
    return torch, bvh_metric.compute_bvh_self_collision_metrics


def test_invalid_meshes_cannot_receive_perfect_scores(native_metric):
    torch, compute = native_metric
    vertices = torch.zeros((1, 3, 3))
    faces = torch.tensor([[0, 1, 2]])
    with pytest.raises(ValueError, match="no frames"):
        compute(vertices[:0], faces)
    with pytest.raises(ValueError, match="nonfinite"):
        compute(vertices + float("nan"), faces)
    with pytest.raises(ValueError, match="invalid vertex indices"):
        compute(vertices, torch.tensor([[0, 1, 3]]))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        compute(vertices, faces, device="cpu")


def test_native_intersection_and_separation(native_metric):
    torch, compute = native_metric
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    vertices = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                              [.25, .25, -1.], [.25, .25, 1.], [.75, .25, 0.]]])
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert compute(vertices, faces, min_num_pairs=1)["score"] == 0
    vertices[:, 3:, 2] += 3
    assert compute(vertices, faces, min_num_pairs=1)["score"] == 100


def test_body_model_override_is_independent_of_cwd(native_metric, monkeypatch, tmp_path):
    from metrics.self_collision_bvh.smpl_utils import resolve_smplx_model_path
    model = tmp_path / "SMPLX_NEUTRAL.npz"
    model.touch()
    monkeypatch.setenv("HMB_SMPLX_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("HMB_SMPLX_MODEL", raising=False)
    monkeypatch.chdir("/tmp")
    assert resolve_smplx_model_path() == model


def test_vectorized_nonlocal_counts_preserve_frame_and_topology_semantics():
    torch = pytest.importorskip("torch")
    from metrics.self_collision_bvh.bvh_metric import _nonlocal_collision_counts
    faces = torch.tensor([[0, 1, 2], [2, 3, 4], [5, 6, 7], [8, 9, 10]])
    collisions = torch.tensor([
        [[0, 2], [2, 0], [0, 1], [1, 2], [2, 2], [-1, -1], [5, 0]],
        [[2, 3], [3, 2], [0, 1], [0, 0], [-1, 2], [1, 4], [-1, -1]],
        [[-1, -1]] * 7,
    ])
    pairs, counts, keys = _nonlocal_collision_counts(collisions, faces)
    assert pairs.tolist() == [2, 1, 0]
    assert counts.tolist() == [3, 2, 0]
    assert keys.tolist() == [0, 1, 2, 6, 7]


def test_collision_chunks_preserve_exact_aggregation(native_metric):
    torch, compute = native_metric
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    intersecting = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                 [.25, .25, -1.], [.25, .25, 1.], [.75, .25, 0.]])
    separated = intersecting.clone()
    separated[3:, 2] += 3
    vertices = torch.stack([intersecting, intersecting, separated, intersecting, separated])
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    single = compute(vertices, faces, min_num_pairs=1, frame_chunk_size=1)
    assert compute(vertices, faces, min_num_pairs=1, frame_chunk_size=3) == single
    assert compute(vertices, faces, min_num_pairs=1, frame_chunk_size=8) == single
    assert single["r_severe"] == pytest.approx(3 / 5)
    assert single["continuity"] == pytest.approx(2 / 5)
    assert single["face_collision_ratio"] == 1
    rejected = compute(vertices, faces, min_num_pairs=2, frame_chunk_size=3)
    assert rejected["score"] == 100
    assert rejected["face_collision_ratio"] == 0
    assert rejected["avg_pairs_per_frame"] == pytest.approx(3 / 5)


def test_native_padded_output_is_memory_bounded():
    pytest.importorskip("torch")
    from metrics.self_collision_bvh.bvh_metric import _bounded_chunk_size, _MAX_BVH_OUTPUT_BYTES
    size = _bounded_chunk_size(20908, 150, 8)
    assert 1 <= size <= 8
    assert size * 20908 * 150 * 2 * 8 <= _MAX_BVH_OUTPUT_BYTES
    with pytest.raises(ValueError, match="positive"):
        _bounded_chunk_size(20908, 150, 0)
