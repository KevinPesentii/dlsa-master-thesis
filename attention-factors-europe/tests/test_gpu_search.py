"""search_validation.py --device cuda: the training step reads the panel's device copies and nothing else
changes on the CPU; a trial that runs out of GPU memory is run again with the deeper pieces recomputed;
run_matrix passes the device to the searches and never starts more workers than --jobs. The tests marked
`cuda` compare a GPU step with the CPU step and run only where torch sees a GPU (the GPU machine)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as rm  # noqa: E402
import search_validation as sv  # noqa: E402
from afe.data import slots  # noqa: E402
from afe.policy import longconv  # noqa: E402
from afe.policy.longconv import LongConvPolicy  # noqa: E402
from test_gap2_matrix import load  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def small_cfg(**policy) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "us_replication.yaml").read_text())
    cfg["training"].update(epochs=2, batch_days=40)
    cfg["policy"].update(residual_lookback=10, hidden=8, dropout=0.0, **policy)
    cfg["model"].update(embedding_dim=8)
    cfg["execution"].update(lag=1)                 # the executed book: reads R_pool and closed
    return cfg


@pytest.fixture()
def panel(tmp_path, returns, universe, features):
    r = returns.copy()
    r["traded"] = np.random.default_rng(0).random(len(r)) > 0.05      # some closes not traded
    r.to_parquet(tmp_path / "returns.parquet", index=False)
    universe.to_parquet(tmp_path / "universe.parquet", index=False)
    features.to_parquet(tmp_path / "features.parquet", index=False)
    return slots.load_slots(tmp_path, str(universe["month"].min().date()), str(returns["date"].max().date()))


def step(p, cfg, model, t0, t1):
    model.train()
    loss, parts, _ = sv.base.span(model, p, t0, t1, cfg)
    loss.backward()
    return loss.detach(), parts["net"].detach(), [q.grad.clone() for q in model.parameters()]


@pytest.mark.parametrize("window", ["rolling", "fixed"])
def test_a_panel_moved_to_the_cpu_steps_exactly_as_before(panel, window):
    cfg = small_cfg(residual_window=window, layers=2)
    on = slots.to_device(panel, "cpu")
    assert set(on.dev) == set(slots.ON_DEVICE) and panel.dev is None
    m = sv.val.fresh_model(cfg, 3, len(panel.features), 7)
    a = step(panel, cfg, m, 60, 100)
    m.zero_grad()
    b = step(on, cfg, m, 60, 100)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert all(torch.equal(x, y) for x, y in zip(a[2], b[2]))
    for name in ("R_pool", "closed") + (("R_win", "has_ret") if window == "fixed" else ()):
        moved = copy.copy(on)                       # the step reads the device copies, not the numpy arrays
        moved.dev = {**on.dev, name: torch.zeros_like(on.dev[name])}
        m.zero_grad()
        assert not torch.equal(step(moved, cfg, m, 60, 100)[1], a[1]), name


def test_out_of_memory_runs_the_trial_again_with_every_piece_recomputed(monkeypatch):
    calls = []

    def trial(job):
        calls.append(longconv.GPU_KEEP)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return {**job, "net_SR": 1.0}
    monkeypatch.setattr(sv, "evaluate_attention", trial)
    monkeypatch.setitem(sv.WORKER, "device", torch.device("cuda", 1))
    keep = longconv.GPU_KEEP
    r = sv.evaluate({"name": "x", "seed": 0})
    assert calls == [keep, 0] and longconv.GPU_KEEP == keep
    assert r["oom_retry"] and r["device"] == "cuda:1"
    monkeypatch.setitem(sv.WORKER, "device", torch.device("cpu"))
    calls.clear()
    with pytest.raises(torch.cuda.OutOfMemoryError):     # on the CPU nothing is retried
        sv.evaluate({"name": "x", "seed": 0})


def test_pieces_follow_the_device():
    assert longconv.pieces(torch.device("cpu")) == (longconv.CHUNK, longconv.KEEP)
    assert longconv.pieces(torch.device("cuda", 0)) == (longconv.GPU_CHUNK, longconv.GPU_KEEP)


def test_gpu_matrix_is_the_all_k_matrix_with_fewer_workers(tmp_path):
    allk, gpu = load("search_t4_allk_matrix"), load("search_t4_allk_gpu_matrix")
    assert gpu["name"] == allk["name"] and not gpu["specs"] and gpu["searches"].keys() == allk["searches"].keys()
    drop = ("workers", "mem_gb_per_worker")
    for name, s in gpu["searches"].items():
        assert {k: v for k, v in s.items() if k not in drop} == {k: v for k, v in allk["searches"][name].items()
                                                                  if k not in drop}
    old = rm.ROOT
    rm.ROOT = tmp_path
    try:
        jobs = {j.id: j for j in rm.expand(gpu, None, tmp_path, 1, False, max_jobs=16, device="cuda")}
        cmd = jobs["search_us_t4_K100"].make_cmd()
        assert cmd[cmd.index("--jobs") + 1] == "8" and cmd[cmd.index("--device") + 1] == "cuda"
        assert cmd[cmd.index("--gpu-procs") + 1] == "16"
        one = {j.id: j for j in rm.expand(gpu, None, tmp_path, 1, False, max_jobs=4, device="cuda")}
        cmd = one["search_us_t4_K100"].make_cmd()
        assert cmd[cmd.index("--jobs") + 1] == "4" and one["search_us_t4_K100"].slots == 4
        cpu = {j.id: j for j in rm.expand(allk, None, tmp_path, 2, False, max_jobs=46)}
        assert "--device" not in cpu["search_us_t4_K100"].make_cmd()
    finally:
        rm.ROOT = old


@cuda
@pytest.mark.parametrize("window", ["rolling", "fixed"])
def test_a_gpu_step_is_the_cpu_step_to_rounding(panel, window):
    """Dropout 0: no random number is drawn, so only the order of the float sums differs."""
    cfg = small_cfg(residual_window=window, layers=2)
    m = sv.val.fresh_model(cfg, 3, len(panel.features), 7)
    g = copy.deepcopy(m).to("cuda")
    a, b = step(panel, cfg, m, 60, 100), step(slots.to_device(panel, "cuda"), cfg, g, 60, 100)
    rel = lambda x, y: float((x - y.cpu()).norm() / x.norm().clamp_min(1e-30))  # noqa: E731
    assert rel(a[0], b[0]) < 1e-4 and rel(a[1], b[1]) < 1e-4
    assert all(rel(x, y) < 1e-3 for x, y in zip(a[2], b[2]))


@cuda
def test_deep_longconv_on_the_gpu_with_pieces_recomputed(monkeypatch):
    monkeypatch.setattr(longconv, "GPU_CHUNK", 7)
    monkeypatch.setattr(longconv, "GPU_KEEP", 0)               # every GPU piece recomputed in backward
    torch.manual_seed(3)
    p = LongConvPolicy(hidden=16, lookback=10, dropout=0.0, layers=3)
    x = torch.randn(40, 10)
    p(x).square().sum().backward()
    want = [q.grad.clone() for q in p.parameters()]
    g = copy.deepcopy(p).to("cuda")
    g.zero_grad()
    g(x.to("cuda")).square().sum().backward()
    assert all(torch.allclose(w, q.grad.cpu(), rtol=1e-4, atol=1e-5) for w, q in zip(want, g.parameters()))
