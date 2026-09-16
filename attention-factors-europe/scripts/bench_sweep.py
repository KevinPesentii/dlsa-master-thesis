"""K-sweep, stage timing, and the cost of the naive dense implementation."""
import time, numpy as np, torch
import bench_epstein as B

torch.manual_seed(0)


def stage_times(T=B.T, N=B.N, K=B.K, reps=3):
    X = torch.randn(T, N, B.M); R = torch.randn(T, N) * 0.02
    m = B.AttentionArb()
    if K != B.K:
        m.factors.Q = torch.nn.Parameter(torch.randn(K, B.D) / np.sqrt(B.D))
    out = {}

    def t(fn, n=reps):
        fn(); t0 = time.perf_counter()
        for _ in range(n): fn()
        return (time.perf_counter() - t0) / n

    Xt = X @ m.factors.W_K
    out['embed'] = t(lambda: X @ m.factors.W_K)
    scores = torch.einsum('kd,tnd->tkn', m.factors.Q, Xt) / np.sqrt(B.D)
    wF = torch.softmax(scores, dim=-1)
    out['scores+softmax'] = t(lambda: torch.softmax(
        torch.einsum('kd,tnd->tkn', m.factors.Q, Xt) / np.sqrt(B.D), dim=-1))
    gram = wF @ wF.transpose(1, 2) + B.LAMBDA_RIDGE * torch.eye(K)
    out['gram'] = t(lambda: wF @ wF.transpose(1, 2))
    out['solve (LU)'] = t(lambda: torch.linalg.solve(gram, wF))
    out['solve (Cholesky)'] = t(lambda: torch.cholesky_solve(wF, torch.linalg.cholesky(gram)))
    betaT = torch.linalg.solve(gram, wF).transpose(1, 2)
    Fh = torch.einsum('tkn,tn->tk', wF, R)
    eps = R - torch.einsum('tnk,tk->tn', betaT, Fh)
    out['residuals (low rank)'] = t(lambda: R - torch.einsum(
        'tnk,tk->tn', betaT, torch.einsum('tkn,tn->tk', wF, R)))
    out['longconv'] = t(lambda: m.policy(eps))
    return out


print(f"stage timing, forward only, {torch.get_num_threads()} CPU threads, "
      f"T={B.T} N={B.N} K={B.K}\n")
st = stage_times()
tot = sum(v for k, v in st.items() if 'Cholesky' not in k)
for k, v in st.items():
    tag = '  (alternative)' if 'Cholesky' in k else f'  {100*v/tot:4.1f}%'
    print(f"  {k:24s} {v*1000:8.1f} ms{tag}")
print(f"  {'TOTAL fwd':24s} {tot*1000:8.1f} ms\n")

print("K sweep, full fwd+bwd epoch\n")
print(f"  {'K':>4s}  {'GMAC/fwd':>9s}  {'s/epoch':>8s}  {'rel':>5s}")
base = None
for K in [1, 5, 30, 100]:
    T, N, M, D, S = B.T, B.N, B.M, B.D, B.S
    mac = (T*N*M*D + T*K*D*N + T*K*N*K + T*(K**3//3 + K*K*N)
           + 2*T*(2*K*N) + 2*T*N*D + N*D*T*S)
    X = torch.randn(T, N, M); R = torch.randn(T, N) * 0.02
    m = B.AttentionArb()
    m.factors.Q = torch.nn.Parameter(torch.randn(K, D) / np.sqrt(D))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    def step():
        opt.zero_grad(); w, e = m(X, R); B.objective(w, e, R).backward(); opt.step()
    step(); t0 = time.perf_counter()
    for _ in range(2): step()
    dt = (time.perf_counter() - t0) / 2
    base = base or dt
    print(f"  {K:>4d}  {mac/1e9:9.2f}  {dt:8.2f}  {dt/base:5.2f}x")

print("\ndense (I - beta^T wF) vs low-rank, short window T=200\n")
B.T = 200
for dense in [False, True]:
    dt = B.bench(dense=dense, epochs=2)
    print(f"  {'dense  ' if dense else 'low-rank'}: {dt:6.2f} s/epoch "
          f"-> {dt*(2016/200)*720/3600:6.2f} h for a full 24-refit OOS pass")
