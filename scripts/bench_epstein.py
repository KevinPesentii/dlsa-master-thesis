"""
Cost model for the Epstein et al. (2025) attention-factor architecture.

Implements the forward/backward at the paper's Table 4 dimensions, verifies that the
low-rank residual/weight mapping equals the dense (I - beta^T w_F) form, counts
parameters and FLOPs, and times one epoch over a full 8-year training window.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# Paper dimensions (Table 4 + Section 4.1)
N = 500      # assets
M = 79       # features: 39 chars + 39 cross-sectional medians + rf
D = 32       # embedding / hidden dim (d, d_x)
K = 30       # attention factors (headline)
S = 30       # residual lookback
T = 8 * 252  # 8-year rolling training window in trading days
LAMBDA_RIDGE = 1e-4
LAMBDA_VAR = 100.0
TC, SC = 5e-4, 1e-4


class AttentionFactors(nn.Module):
    def __init__(self, m=M, d=D, k=K):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(m, d) / np.sqrt(m))
        self.Q = nn.Parameter(torch.randn(k, d) / np.sqrt(d))
        self.d = d

    def forward(self, X):                                  # X: (T, N, M)
        Xt = X @ self.W_K                                  # (T, N, d)
        scores = torch.einsum('kd,tnd->tkn', self.Q, Xt) / np.sqrt(self.d)
        wF = torch.softmax(scores, dim=-1)                 # (T, K, N) rows sum to 1
        gram = wF @ wF.transpose(1, 2)                     # (T, K, K)
        gram = gram + LAMBDA_RIDGE * torch.eye(gram.shape[-1])
        betaT = torch.linalg.solve(gram, wF).transpose(1, 2)  # (T, N, K)
        return wF, betaT


class LongConv(nn.Module):
    """Fu et al. (2023) long convolution, causal over the whole sequence."""
    def __init__(self, d=D, ker=S, lam_squash=1e-3):
        super().__init__()
        h = torch.arange(d).float()
        t = torch.arange(ker).float()
        decay = torch.exp(-t[None, :] / ker * (d / 2) ** (h[:, None] / d))
        self.kernel = nn.Parameter(torch.randn(d, ker) * decay)
        self.skip = nn.Parameter(torch.randn(d))
        self.inp = nn.Linear(1, d)
        self.out = nn.Linear(d, 1)
        self.ker, self.lam = ker, lam_squash

    def forward(self, eps):                                # eps: (T, N) residual returns
        u = self.inp(eps.t().unsqueeze(-1))                # (N, T, d)
        u = u.permute(0, 2, 1)                             # (N, d, T)
        kbar = torch.sign(self.kernel) * torch.clamp(self.kernel.abs() - self.lam, min=0)
        up = F.pad(u, (self.ker - 1, 0))
        y = F.conv1d(up, kbar.flip(-1).unsqueeze(1), groups=kbar.shape[0])
        y = y + self.skip[None, :, None] * u
        return self.out(y.permute(0, 2, 1)).squeeze(-1).t()  # (T, N)


class AttentionArb(nn.Module):
    def __init__(self):
        super().__init__()
        self.factors = AttentionFactors()
        self.policy = LongConv()

    def forward(self, X, R, dense=False):
        wF, betaT = self.factors(X)                        # (T,K,N), (T,N,K)
        if dense:
            I = torch.eye(N).expand(X.shape[0], N, N)
            w_eps = I - betaT @ wF                         # (T, N, N)  <- avoid this
            eps = torch.einsum('tij,tj->ti', w_eps, R)
        else:                                              # low-rank: never form N x N
            Fh = torch.einsum('tkn,tn->tk', wF, R)         # factor returns (T, K)
            eps = R - torch.einsum('tnk,tk->tn', betaT, Fh)
        w_port = self.policy(eps)                          # (T, N) residual-space weights
        if dense:
            w = torch.einsum('tji,tj->ti', w_eps, w_port)
        else:
            bw = torch.einsum('tkn,tn->tk', betaT.transpose(1, 2), w_port)
            w = w_port - torch.einsum('tkn,tk->tn', wF, bw)
        w = w / (w.abs().sum(dim=1, keepdim=True) + 1e-12)  # ||w||_1 = 1, asset space
        return w, eps


def objective(w, eps, R, rf=0.0):
    r_gross = (w[:-1] * R[1:]).sum(dim=1)
    cost = TC * (w[1:] - w[:-1]).abs().sum(dim=1) + SC * torch.clamp(-w[1:], min=0).sum(dim=1)
    r_net = r_gross - cost
    sharpe = (r_net.mean() - rf) / (r_net.std() + 1e-12)
    ev = (1 - eps.var(dim=0) / (R.var(dim=0) + 1e-12)).mean()
    return -(sharpe + LAMBDA_VAR * ev)


def flops_per_epoch():
    """Analytic MAC count for one forward pass over T dates, x2 for FLOPs, x3 for fwd+bwd."""
    f = {
        'embed  X @ W_K':      T * N * M * D,
        'scores Q @ Xt':       T * K * D * N,
        'gram   wF wF^T':      T * K * N * K,
        'solve  K x K':        T * (K ** 3 // 3 + K * K * N),
        'resid  (low rank)':   T * (K * N + N * K),
        'wmap   (low rank)':   T * (K * N + N * K),
        'longconv in-proj':    T * N * D,
        'longconv depthwise':  N * D * T * S,
        'longconv out-proj':   T * N * D,
    }
    dense_extra = T * N * K * N + T * N * N + T * N * N
    return f, dense_extra


def bench(dense=False, epochs=3):
    X = torch.randn(T, N, M)
    R = torch.randn(T, N) * 0.02
    model = AttentionArb()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)
    opt.zero_grad(); loss = objective(*model(X, R, dense=dense)[:2], R); loss.backward(); opt.step()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        w, eps = model(X, R, dense=dense)
        objective(w, eps, R).backward()
        opt.step()
    return (time.perf_counter() - t0) / epochs


if __name__ == '__main__':
    print(f"dims: N={N} M={M} d={D} K={K} lookback={S} window={T} days ({T/252:.0f}y)\n")

    m = AttentionArb()
    tot = sum(p.numel() for p in m.parameters())
    print("parameters")
    for n_, p in m.named_parameters():
        print(f"  {n_:24s} {tuple(p.shape)!s:12s} {p.numel():>7,}")
    print(f"  {'TOTAL':24s} {'':12s} {tot:>7,}  ({tot*4/1024:.1f} KB fp32)\n")

    # correctness: low-rank == dense
    Xs, Rs = torch.randn(40, N, M), torch.randn(40, N) * 0.02
    with torch.no_grad():
        w1, e1 = m(Xs, Rs, dense=False)
        w2, e2 = m(Xs, Rs, dense=True)
    print(f"low-rank vs dense (I - beta^T wF):  max|d eps| = {(e1-e2).abs().max():.2e}"
          f"   max|d w| = {(w1-w2).abs().max():.2e}\n")

    f, dense_extra = flops_per_epoch()
    tot_mac = sum(f.values())
    print("MACs per forward pass over the window")
    for k_, v in f.items():
        print(f"  {k_:22s} {v/1e9:8.3f} G   ({100*v/tot_mac:4.1f}%)")
    print(f"  {'TOTAL':22s} {tot_mac/1e9:8.3f} G")
    print(f"  fwd+bwd ~3x, 2 FLOP/MAC -> {6*tot_mac/1e9:.1f} GFLOP per epoch")
    print(f"  materialising w_eps (N x N) would add {dense_extra/1e9:.1f} GMAC "
          f"({dense_extra/tot_mac:.0f}x) and {T*N*N*4/1e9:.1f} GB of activations\n")

    mem = {'X (T,N,M)': T*N*M*4, 'longconv acts (N,d,T)': N*D*T*4,
           'wF (T,K,N)': T*K*N*4, 'betaT (T,N,K)': T*N*K*4,
           'DENSE w_eps (T,N,N)': T*N*N*4}
    print("peak resident tensors")
    for k_, v in mem.items():
        print(f"  {k_:24s} {v/1e6:9.1f} MB")
    print()

    t_lr = bench(dense=False)
    print(f"measured, {torch.get_num_threads()} CPU threads, no GPU:")
    print(f"  low-rank : {t_lr:6.2f} s / epoch  -> {6*tot_mac/1e9/t_lr:6.1f} GFLOPS achieved")
    print(f"  30 epochs (one annual refit): {30*t_lr/60:.1f} min")
    print(f"  24 refits (full 1998-2021 OOS pass): {24*30*t_lr/3600:.1f} h")
