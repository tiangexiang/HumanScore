"""HADM optimizations must apply the exact complete EMA state."""
import importlib

import pytest

from hmb.hadm_prepare import apply_ema_parameters


def test_ema_overwrites_all_parameters_and_buffers_like_upstream():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3))
    ema = {}
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        ema[name] = torch.full_like(value, 7)
    apply_ema_parameters(model, {"model": {"unused": "regular weights"}, "ema_state": ema})
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        assert torch.equal(value, ema[name])


def test_incomplete_ema_never_leaves_silent_random_weights():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 3)
    original = model.weight.clone()
    with pytest.raises(ValueError, match="missing or incompatible"):
        apply_ema_parameters(model, {"ema_state": {"weight": torch.zeros_like(model.weight)}})
    assert torch.equal(model.weight, original)
    with pytest.raises(ValueError, match="complete ema_state"):
        apply_ema_parameters(model, {"model": model.state_dict()})


def test_single_video_wrapper_uses_headless_adapter(monkeypatch, tmp_path):
    from types import SimpleNamespace
    module = importlib.import_module("metrics.multi-limbs.eval_extra_limbs")
    root = tmp_path / "hadm"
    for relative in ("tools/lazyconfig_train_net.py", "cfg.py", "weights.pth"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    calls = []
    monkeypatch.delenv("HMB_HADM_UPSTREAM", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(module.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(returncode=0))
    module.run_hadm_inference(str(root), str(tmp_path / "frames"), str(tmp_path / "out"),
                              config_file="cfg.py", checkpoint_path="weights.pth")
    command, options = calls[0]
    assert command[1].endswith("hmb/hadm_prepare.py")
    assert command[command.index("--checkpoint") + 1] == "weights.pth"
    assert options["env"]["WANDB_MODE"] == "disabled"
    # The adapter appends the HADM root to sys.path itself; on PYTHONPATH its unbuilt detectron2
    # source tree would shadow the installed, compiled detectron2.
    assert "PYTHONPATH" not in options["env"]
