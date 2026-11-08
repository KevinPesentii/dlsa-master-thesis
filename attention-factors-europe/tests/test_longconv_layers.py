"""Table 4's "Nr layers" and "LongConv init" (policy.layers, policy.kernel_init).

The checks are the definition written out by hand: the causal convolution of Appendix A,
y[t, h] = sum_{s <= t} Kbar[h, t - s] u[s, h] + D_h u[t, h], as a double loop; the last
position of that sequence is the one-layer shortcut; the "random" kernel is the geometric
decay kernel without its decay factor; one layer is the model of every earlier run."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from afe.model.attention_pipeline import AttentionArb
from afe.policy import longconv
from afe.policy.longconv import LongConvLayer, LongConvPolicy

d, L, B = 4, 6, 3


def layer(seed=0, lam=0.05, init="geom_decay"):
    torch.manual_seed(seed)
    m = LongConvLayer(d, L, dropout=0.0, lambda_squash=lam, init=init)
    return m.eval()


def by_hand(m: LongConvLayer, u: torch.Tensor) -> torch.Tensor:
    """forward_seq written out: loops over t and s, nothing vectorised."""
    K = m.kernel.detach()
    Kbar = torch.sign(K) * torch.clamp(K.abs() - m.lam, min=0.0)
    y = torch.zeros_like(u)
    for b in range(u.shape[0]):
        for t in range(L):
            for h in range(d):
                y[b, t, h] = sum(Kbar[h, t - s] * u[b, s, h] for s in range(t + 1)) + m.skip[h] * u[b, t, h]
    return F.glu(m.out(F.gelu(y)), dim=-1) + u


def cf(u: torch.Tensor) -> torch.Tensor:
    """(B, L, d) -> the layer's channels-first (d, B, L)."""
    return u.permute(2, 0, 1)


def test_sequence_layer_is_the_causal_convolution():
    m, u = layer(), torch.randn(B, L, d)
    with torch.no_grad():
        assert torch.allclose(m.forward_seq(cf(u)), cf(by_hand(m, u)), atol=1e-5)


def test_a_later_day_never_moves_an_earlier_position():
    m, u = layer(), torch.randn(d, B, L)
    v = u.clone()
    v[..., 4:] += 10.0
    with torch.no_grad():
        assert torch.equal(m.forward_seq(u)[..., :4], m.forward_seq(v)[..., :4])
        assert not torch.allclose(m.forward_seq(u)[..., 4:], m.forward_seq(v)[..., 4:])


def test_last_position_read_equals_the_whole_sequence_read():
    m, u = layer(), torch.randn(d, B, L)
    with torch.no_grad():
        assert torch.allclose(m.forward_last(u), m.forward_seq(u)[..., -1].T, atol=1e-6)


def test_one_layer_shortcut_is_the_sequence_layer_on_the_affine_encoding():
    m, x = layer(), torch.randn(B, L)
    a, c = torch.randn(d), torch.randn(d)
    with torch.no_grad():
        assert torch.allclose(m(x, a, c), m.forward_last(cf(x[..., None] * a + c)), atol=1e-5)


def test_random_init_is_the_geometric_kernel_without_its_decay():
    geo, rnd = layer(seed=3).kernel.detach(), layer(seed=3, init="random").kernel.detach()
    for h in range(d):
        for t in range(L):
            assert math.isclose(geo[h, t].item(), rnd[h, t].item() * math.exp(-t / L * (d / 2) ** (h / d)),
                                 rel_tol=1e-5, abs_tol=1e-7)
    with pytest.raises(ValueError):
        LongConvLayer(d, L, 0.0, 0.0, init="xavier")


def test_one_layer_policy_is_the_model_of_every_earlier_run():
    torch.manual_seed(0)
    p = LongConvPolicy(hidden=d, lookback=L, dropout=0.0, lambda_squash=0.05)
    assert sorted(p.state_dict()) == sorted(["encoder.weight", "encoder.bias", "layer.kernel", "layer.skip",
                                             "layer.out.weight", "layer.out.bias", "decoder.weight", "decoder.bias"])
    torch.manual_seed(0)                       # the module order before `layers`: encoder, layer, decoder
    enc, lay, dec = torch.nn.Linear(1, d), LongConvLayer(d, L, 0.0, 0.05), torch.nn.Linear(d, 1)
    for name, ref in (("encoder", enc), ("layer", lay), ("decoder", dec)):
        for k, v in ref.state_dict().items():
            assert torch.equal(p.state_dict()[f"{name}.{k}"], v)


def test_a_deeper_stack_feeds_each_layer_the_whole_sequence_of_the_one_before():
    torch.manual_seed(1)
    p = LongConvPolicy(hidden=d, lookback=L, dropout=0.0, lambda_squash=0.05, layers=3).eval()
    w = torch.randn(2, B, L)
    a, c = p.encoder.weight[:, 0], p.encoder.bias
    with torch.no_grad():
        u = by_hand(p.layer, w.reshape(-1, L)[..., None] * a + c)
        u = by_hand(p.deeper[0], u)
        want = p.decoder(by_hand(p.deeper[1], u)[:, -1]).reshape(2, B)
        assert torch.allclose(p(w), want, atol=1e-5)


def test_gradient_reaches_every_layer():
    torch.manual_seed(2)
    p = LongConvPolicy(hidden=d, lookback=L, dropout=0.1, lambda_squash=0.0, layers=2)
    p(torch.randn(5, L)).sum().backward()
    assert all(q.grad is not None and q.grad.abs().sum() > 0 for q in p.parameters())


def test_pieces_recomputed_in_backward_change_nothing(monkeypatch):
    monkeypatch.setattr(longconv, "CHUNK", 5)                 # 12 windows: pieces of 5, 5 and 2
    monkeypatch.setattr(longconv, "KEEP", 0)                  # every piece recomputed
    torch.manual_seed(4)
    p = LongConvPolicy(hidden=d, lookback=L, dropout=0.2, lambda_squash=0.0, layers=3)
    x = torch.randn(12, L)
    torch.manual_seed(9)
    p(x).square().sum().backward()
    g1 = [q.grad.clone() for q in p.parameters()]
    p.zero_grad()
    torch.manual_seed(9)                                      # the same pieces, kept instead of recomputed
    a, c = p.encoder.weight[:, 0], p.encoder.bias
    p.decoder(torch.cat([p.stack(xc, a, c) for xc in x.split(5)])).square().sum().backward()
    assert all(torch.allclose(u, q.grad, atol=1e-6) for u, q in zip(g1, p.parameters()))
    p.eval()
    with torch.no_grad():                                     # and without dropout, pieces = one piece
        assert torch.allclose(p(x), p.decoder(p.stack(x, a, c)).squeeze(-1), atol=1e-6)


def test_attention_model_passes_layers_and_init_on():
    m = AttentionArb(n_features=3, n_factors=2, embedding_dim=4, hidden=d, lookback=L, layers=2,
                     kernel_init="random")
    assert len(m.policy.deeper) == 1
    with pytest.raises(ValueError):
        LongConvPolicy(hidden=d, lookback=L, layers=0)
