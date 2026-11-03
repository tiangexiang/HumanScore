"""Tests of the self-collision metric (hmb.scoring.self_collision; CPU, synthetic meshes + optional SMPL-X)."""
import numpy as np
import pytest

from hmb.scoring import self_collision as sc

torch = pytest.importorskip("torch")

SIDE = 0.10   # 10 cm cubes
N_SUB = 8     # 1.25 cm grid, the centre vertex of every face exists


def cube(n=N_SUB, side=SIDE, offset=(0.0, 0.0, 0.0)):
    """Closed, outward-oriented, subdivided cube [0, side]^3 + offset."""
    verts, faces, index = [], [], {}

    def vid(p):
        key = tuple(np.round(np.asarray(p) * 1e6).astype(np.int64))
        if key not in index:
            index[key] = len(verts)
            verts.append(np.asarray(p, float))
        return index[key]

    g = np.linspace(0.0, side, n + 1)
    for axis in range(3):
        for hi in (0, 1):
            u_ax, v_ax = [a for a in range(3) if a != axis]
            for i in range(n):
                for j in range(n):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[axis] = side * hi
                        p[u_ax], p[v_ax] = g[i + di], g[j + dj]
                        quad.append(vid(p))
                    a, b, c, d = quad
                    tris = [(a, b, c), (a, c, d)]
                    for t in tris:
                        tri = np.array([verts[k] for k in t])
                        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                        outward = np.zeros(3)
                        outward[axis] = 1.0 if hi else -1.0
                        faces.append(t if normal @ outward > 0 else (t[0], t[2], t[1]))
    return np.asarray(verts) + np.asarray(offset), np.asarray(faces, np.int64)


def two_cubes(overlaps, adjacent=False, exclude_b=False):
    """Frames with cube B shifted along x so that it overlaps cube A by ``d`` (negative = gap)."""
    va, fa = cube()
    vb, fb = cube()
    faces = np.concatenate([fa, fb + len(va)])
    labels = np.concatenate([np.zeros(len(va), np.int64), np.full(len(vb), -1 if exclude_b else 1, np.int64)])
    adj = np.eye(2, dtype=bool)
    if adjacent:
        adj[:] = True
    topo = sc.BodyTopology.from_labels(faces, labels, ("A", "B"), adj)
    frames = []
    for d in overlaps:
        frames.append(np.concatenate([va, vb + np.array([SIDE - d, 0.013, 0.007])]))
    return torch.as_tensor(np.asarray(frames), dtype=torch.float32), topo


def measure(overlaps, **kw):
    verts, topo = two_cubes(overlaps, **kw)
    return sc.measure_mesh_sequence(verts, topo, backend="cpu", with_full_mesh=True)


# ---------------------------------------------------------------------------
def test_segment_adjacency_is_the_kinematic_tree():
    adj = sc.segment_adjacency()
    idx = {n: i for i, n in enumerate(sc.SEGMENTS)}
    assert (adj == adj.T).all() and adj.diagonal().all()
    for a, b in (("torso", "l_upperarm"), ("l_forearm", "l_hand"), ("torso", "head"), ("r_shank", "r_foot")):
        assert adj[idx[a], idx[b]]
    for a, b in (("torso", "l_hand"), ("l_thigh", "r_thigh"), ("head", "r_hand"), ("l_hand", "r_hand"),
                 ("torso", "l_forearm"), ("l_upperarm", "l_hand")):
        assert not adj[idx[a], idx[b]]


def test_cpu_triangle_test_matches_definition():
    t = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.25, 0.25, -1], [0.25, 0.25, 1], [0.75, 0.25, 0]], float)
    f = np.array([[0, 1, 2], [3, 4, 5]])
    assert sc.triangle_pairs_cpu(t, f).tolist() == [[0, 1]]
    t2 = t.copy()
    t2[3:, 2] += 3.0
    assert len(sc.triangle_pairs_cpu(t2, f)) == 0
    shared = np.array([[0, 1, 2], [0, 4, 5]])  # shares vertex 0 -> local, excluded
    assert len(sc.triangle_pairs_cpu(t, shared)) == 0


def test_torch_backend_matches_numpy_reference():
    rng = np.random.default_rng(3)
    # random triangle soup in a small box: many intersections, some shared vertices
    v = rng.uniform(0, 0.1, (2, 300, 3))
    f = rng.integers(0, 300, (400, 3))
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    ref = sc.triangle_pairs(v, f, "numpy")
    got = sc.triangle_pairs(torch.as_tensor(v), f, "cpu")
    assert len(ref) > 50
    assert {tuple(r) for r in ref} == {tuple(r) for r in got}
    verts, topo = two_cubes([0.02, -0.01, 0.03])
    verts = verts.double()  # same precision as the numpy reference (float32 differs only on touching edges)
    ref = sc.triangle_pairs(verts, topo.faces_full, "numpy")
    got = sc.triangle_pairs(verts, topo.faces_full, "cpu")
    assert {tuple(r) for r in ref} == {tuple(r) for r in got}
    # two-level patch broad phase (production path) == reference restricted to allowed segment pairs
    allowed = ~topo.adjacent
    got = sc.triangle_pairs(verts[:, torch.as_tensor(topo.keep_vertices)], topo.faces, "cpu",
                            face_segment=topo.face_segment, allowed=allowed)
    ref = sc.triangle_pairs(verts[:, torch.as_tensor(topo.keep_vertices)], topo.faces, "numpy",
                            face_segment=topo.face_segment, allowed=allowed)
    assert len(ref) > 0 and {tuple(r) for r in ref} == {tuple(r) for r in got}


def test_nonadjacent_only_path_matches_full_path():
    verts, topo = two_cubes([0.005, 0.02, -0.01, 0.03])
    full = sc.measure_mesh_sequence(verts, topo, backend="cpu", with_full_mesh=True)
    fast = sc.measure_mesh_sequence(verts, topo, backend="cpu", with_full_mesh=False, nonadjacent_only=True)
    for key in ("na_pairs", "na_faces", "depth"):
        np.testing.assert_allclose(full[key], fast[key])


def test_winding_number_and_distance():
    v, f = cube()
    tri = torch.as_tensor(v[f], dtype=torch.float64)[None]
    pts = torch.tensor([[[0.05, 0.05, 0.05], [0.05, 0.05, 0.09], [0.2, 0.05, 0.05]]], dtype=torch.float64)
    w, d = sc.winding_and_distance(pts, tri)
    assert w[0].tolist() == pytest.approx([1.0, 1.0, 0.0], abs=1e-6)
    assert d[0].tolist() == pytest.approx([0.05, 0.01, 0.10], abs=1e-9)


def test_measured_depth_equals_injected_overlap():
    overlaps = [-0.01, 0.005, 0.01, 0.02, 0.03]
    m = measure(overlaps)
    assert m["na_pairs"][0] == 0 and m["depth"][0] == 0.0
    assert (m["na_pairs"][1:] > 0).all()
    np.testing.assert_allclose(m["depth"][1:], overlaps[1:], atol=1e-4)
    # depth is monotone in the injected overlap
    assert np.all(np.diff(m["depth"]) > 0)


def test_adjacent_or_excluded_parts_are_ignored():
    m = measure([0.03, 0.03], adjacent=True)
    assert (m["body_pairs"] > 0).all() and (m["na_pairs"] == 0).all() and (m["depth"] == 0).all()
    m = measure([0.03, 0.03], exclude_b=True)
    assert (m["full_mesh_pairs"] > 0).all() and (m["body_pairs"] == 0).all() and (m["depth"] == 0).all()
    assert sc.score_measurements(m)["score"] == 100.0


def test_rigid_motion_invariance():
    verts, topo = two_cubes([0.02, 0.03])
    m0 = sc.measure_mesh_sequence(verts, topo, backend="cpu")
    ang = 0.7
    R = torch.tensor([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]], dtype=torch.float32)
    moved = verts @ R.T + torch.tensor([1.5, -2.0, 0.3])
    m1 = sc.measure_mesh_sequence(moved, topo, backend="cpu")
    np.testing.assert_allclose(m0["depth"], m1["depth"], atol=2e-5)


def test_clean_sequence_scores_100_and_contact_is_tolerated():
    cfg = sc.CollisionConfig(contact_depth_m=0.02, severe_depth_m=0.05)
    assert sc.score_depth(np.zeros(100), cfg)["score"] == 100.0
    assert sc.score_depth(np.full(100, 0.019), cfg)["score"] == 100.0  # shallow contact everywhere
    # end to end: shallow overlap below the contact tolerance -> 100
    m = measure([0.01] * 4)
    assert sc.score_measurements(m, cfg)["score"] == 100.0


def test_score_decreases_monotonically_with_defect_extent_and_depth():
    cfg = sc.CollisionConfig(contact_depth_m=0.02, severe_depth_m=0.05)
    base = np.zeros(60)
    by_frames = []
    for k in (0, 1, 5, 10, 30, 60):
        d = base.copy()
        d[:k] = 0.08
        by_frames.append(sc.score_depth(d, cfg)["score"])
    assert by_frames[0] == 100.0 and all(a > b for a, b in zip(by_frames, by_frames[1:]))
    by_depth = []
    for depth in (0.0, 0.025, 0.035, 0.045, 0.06, 0.10):
        d = base.copy()
        d[10:20] = depth
        by_depth.append(sc.score_depth(d, cfg)["score"])
    assert all(a >= b for a, b in zip(by_depth, by_depth[1:])) and by_depth[-1] < by_depth[1] < 100.0


def test_end_to_end_injected_penetration_lowers_score():
    cfg = sc.CollisionConfig(contact_depth_m=0.01, severe_depth_m=0.03)
    scores = []
    for d in (0.005, 0.015, 0.025, 0.04):
        m = measure([-0.01] * 3 + [d] * 3)
        scores.append(sc.score_measurements(m, cfg)["score"])
    assert scores[0] == 100.0 and all(a > b for a, b in zip(scores, scores[1:]))


def test_frame_order_and_duplication_invariance():
    cfg = sc.CollisionConfig(contact_depth_m=0.02, severe_depth_m=0.05)
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 0.08, 40) * (rng.uniform(size=40) < 0.2)
    a = sc.score_depth(d, cfg)
    b = sc.score_depth(d[::-1], cfg)
    c = sc.score_depth(np.repeat(d, 2), cfg)  # same motion sampled at twice the frame rate
    assert a["score"] == pytest.approx(b["score"]) == pytest.approx(c["score"])


def test_config_validation():
    with pytest.raises(ValueError):
        sc.CollisionConfig(contact_depth_m=0.05, severe_depth_m=0.02).validate()


# ---------------------------------------------------------------------------
# Locating the SMPL-X model
# ---------------------------------------------------------------------------
@pytest.fixture
def model_places(tmp_path, monkeypatch):
    """Empty install locations (hmb.cli default, GVHMR body models, outputs/assets) under tmp_path."""
    from hmb import cli

    places = {"default": tmp_path / "default_dir", "gvhmr": tmp_path / "gvhmr/smplx",
              "assets": tmp_path / "assets/smplx"}
    monkeypatch.delenv("HMB_SMPLX_MODEL", raising=False)
    monkeypatch.delenv("HMB_SMPLX_MODEL_DIR", raising=False)
    monkeypatch.setattr(cli, "default_smplx_dir", lambda: places["default"])
    monkeypatch.setattr(sc, "GVHMR_SMPLX_MODEL", places["gvhmr"] / sc.SMPLX_MODEL_NAME)
    monkeypatch.setattr(sc, "DEFAULT_SMPLX_MODEL", places["assets"] / sc.SMPLX_MODEL_NAME)
    return places


def _model(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / sc.SMPLX_MODEL_NAME
    path.write_bytes(b"npz")
    return path.resolve()


def test_smplx_model_search_order(model_places, monkeypatch):
    with pytest.raises(FileNotFoundError) as missing:
        sc.resolve_smplx_model()
    message = str(missing.value)
    for folder in model_places.values():                       # every place searched is named
        assert str(folder / sc.SMPLX_MODEL_NAME) in message
    assert "HMB_SMPLX_MODEL" in message and "HMB_SMPLX_MODEL_DIR" in message
    assert _model(model_places["assets"]) == sc.resolve_smplx_model()        # outputs/assets last
    assert _model(model_places["gvhmr"]) == sc.resolve_smplx_model()         # GVHMR body models before it
    assert _model(model_places["default"]) == sc.resolve_smplx_model()       # hmb.cli.default_smplx_dir first
    assert [source for source, _ in sc.smplx_model_candidates()] == ["hmb.cli default_smplx_dir",
                                                                     "GVHMR body models", "outputs/assets"]


def test_smplx_model_environment_overrides(model_places, tmp_path, monkeypatch):
    _model(model_places["default"])
    folder = tmp_path / "licensed"
    monkeypatch.setenv("HMB_SMPLX_MODEL_DIR", str(folder))
    with pytest.raises(FileNotFoundError, match="HMB_SMPLX_MODEL_DIR") as missing:
        sc.resolve_smplx_model()                                 # an explicit folder is never replaced
    assert str(folder / sc.SMPLX_MODEL_NAME) in str(missing.value)
    assert str(folder / "smplx" / sc.SMPLX_MODEL_NAME) in str(missing.value)
    nested = _model(folder / "smplx")                            # <dir>/smplx/SMPLX_NEUTRAL.npz
    assert sc.resolve_smplx_model() == nested
    direct = _model(folder)                                      # <dir>/SMPLX_NEUTRAL.npz first
    assert sc.resolve_smplx_model() == direct

    explicit = tmp_path / "elsewhere" / "my_smplx.npz"
    monkeypatch.setenv("HMB_SMPLX_MODEL", str(explicit))         # the file wins over the folder
    with pytest.raises(FileNotFoundError, match="HMB_SMPLX_MODEL") as missing:
        sc.resolve_smplx_model()
    assert str(explicit) in str(missing.value) and str(direct) not in str(missing.value)
    explicit.parent.mkdir()
    explicit.write_bytes(b"npz")
    assert sc.resolve_smplx_model() == explicit.resolve()

    given = tmp_path / "given.npz"                              # an explicit argument wins over everything
    with pytest.raises(FileNotFoundError, match="given.npz"):
        sc.resolve_smplx_model(given)
    given.write_bytes(b"npz")
    assert sc.resolve_smplx_model(given) == given.resolve()


def test_missing_smplx_model_is_reported_as_missing_input(model_places, tmp_path):
    from hmb.scoring import score_session as ss

    session = tmp_path / "S"
    (session / "gvhmr" / "clip").mkdir(parents=True)
    z = torch.zeros((2, 3))
    torch.save({"smpl_params_global": {"body_pose": torch.zeros((2, 63)), "betas": torch.zeros((2, 10)),
                                       "global_orient": z, "transl": z}},
               session / "gvhmr" / "clip" / "hmr4d_results.pt")
    with pytest.raises(FileNotFoundError, match="SMPLX_NEUTRAL.npz not found"):
        sc.score_session(session, backend="cpu", write=False)
    rep = ss.score_session(session, "clip", metrics=("self_collision",), write=False, collision_backend="cpu")
    entry = rep["metrics"]["self_collision"]
    assert entry["status"] == "missing_input" and entry["score"] is None
    assert entry["reason"].startswith("SMPL-X model SMPLX_NEUTRAL.npz not found: place the licensed model")
    assert "\n" not in entry["reason"] and "HMB_SMPLX_MODEL_DIR" in entry["reason"]
    assert str(model_places["assets"] / sc.SMPLX_MODEL_NAME) in entry["error"]


# ---------------------------------------------------------------------------
# SMPL-X template (skipped without the licensed model / smplx package)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smplx_ready():
    pytest.importorskip("smplx")
    try:
        sc.resolve_smplx_model()
    except FileNotFoundError:
        pytest.skip("SMPLX_NEUTRAL.npz not available")
    return True


def test_smplx_partition(smplx_ready):
    topo = sc.smplx_topology(exclude_fingers=False)
    # eyeballs are the last 1092 vertices (two closed components of 546)
    assert topo.keep_vertices.size == 9383 and topo.keep_vertices.max() == 9382
    assert len(topo.faces) == 20908 - 2176
    counts = np.bincount(topo.vertex_segment, minlength=len(sc.SEGMENTS))
    assert (counts > 100).all()
    no_fingers = sc.smplx_topology(exclude_fingers=True)
    assert no_fingers.keep_vertices.size < topo.keep_vertices.size


def test_smplx_rest_pose_has_no_scored_collision(smplx_ready, tmp_path):
    z = np.zeros((2, 3), np.float32)
    params = {"body_pose": np.zeros((2, 63), np.float32), "betas": np.zeros((2, 10), np.float32),
              "global_orient": z, "transl": z}
    topo = sc.smplx_topology(exclude_fingers=False)
    verts = sc.smplx_vertices(params, device="cpu")
    m = sc.measure_mesh_sequence(verts, topo, backend="cpu", with_full_mesh=True)
    # the full mesh has the intrinsic eyeball/mouth/toe intersections (~1.5 % of faces) ...
    assert (m["full_mesh_faces"] / m["num_faces_full"] > 0.01).all()
    # ... the scored surface has none between non-adjacent parts
    assert (m["na_pairs"] == 0).all() and (m["depth"] == 0).all()
    # session front end
    session = tmp_path / "S"
    (session / "gvhmr" / "clip").mkdir(parents=True)
    torch.save({"smpl_params_global": {k: torch.as_tensor(v) for k, v in params.items()}},
               session / "gvhmr" / "clip" / "hmr4d_results.pt")
    res = sc.score_session(session, backend="cpu")
    assert res["self_collision_score"] == 100.0
    assert (session / "metrics/self_collision/clip_self_collision.json").is_file()
