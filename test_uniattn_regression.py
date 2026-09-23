#!/usr/bin/env python3
"""Regression suite for attention_uniattn: correctness invariants + performance guards.

Run after ANY change to uniattn_pruning.py (and ideally in CI):

    python3 test_uniattn_regression.py                 # uses installed sglang module
    UNIATTN_MODULE=/path/to/uniattn_pruning.py python3 test_uniattn_regression.py

Correctness part runs on CPU or GPU. Performance part needs CUDA and is skipped
otherwise. Exit code 0 = all green.

WHY THESE SPECIFIC CHECKS
-------------------------
* lambda=0 must stay byte-identical to attention_score top-k -- that is the
  contract behind "switch off => zero impact".
* batched / Triton must stay byte-identical to the per-image reference -- the
  whole point is that optimisation never changes *which* tokens survive.
* PERF: if Triton is unavailable or its launch fails, the code silently falls
  back to the eager loop and still returns correct results -- ~58x slower with
  no error at all. Only an explicit "Triton path was actually taken" assertion
  catches that silent regression. Same for anyone reintroducing a per-image loop.
"""
import os, sys, time

import torch

FAIL = 0
SKIP = 0


def chk(cond, msg):
    global FAIL
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL += 1


def skip(msg):
    global SKIP
    print("  skip " + msg)
    SKIP += 1


def load_module():
    path = os.environ.get("UNIATTN_MODULE")
    if path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("uniattnmod", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    from sglang.srt.layers.attention import uniattn_pruning as m
    return m


M = load_module()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = 3072
RATE = 0.5
print(f"module={M.__file__ if hasattr(M,'__file__') else '?'} device={DEV} "
      f"HAS_TRITON={getattr(M,'_HAS_TRITON',None)}")

# ---------------------------------------------------------------- correctness
print("\n[correctness]")
torch.manual_seed(0)

# 1) lambda=0 is byte-identical to attention_score top-k (per image)
try:
    from sglang.srt.layers.attention.visual_token_pruning import (
        prune_visual_tokens_dominant_only as topk_ref,
    )
    has_ref = True
except Exception:
    has_ref = False

N = 216
emb1 = torch.randn(N, D, device=DEV)
sc1 = torch.randn(N, device=DEV)
if has_ref:
    _, k0 = M.prune_visual_tokens_uniattn(emb1, sc1, RATE, 0.0)
    _, kd = topk_ref(emb1, sc1, RATE)
    chk(torch.equal(k0, kd), "lambda=0 == attention_score top-k (per image)")
else:
    skip("attention_score reference not importable (standalone module run)")

# 2) budget / ordering / dtype
keep_num = max(1, int(N * (1 - RATE)))
em, km = M.prune_visual_tokens_uniattn(emb1, sc1, RATE, 0.5)
chk(km.numel() == keep_num, f"exact budget K={keep_num} (got {km.numel()})")
chk(bool((km.sort().values == km).all()), "keep_indices ascending (raster order)")
chk(em.shape == (keep_num, D), f"output shape {tuple(em.shape)}")
chk(em.dtype == emb1.dtype, "dtype preserved")

# 3) batched entry == per-image reference, across sizes and lambdas
sizes = [64, 64, 128, 216]
embs = [torch.randn(n, D, device=DEV) for n in sizes]
scs = [torch.randn(n, device=DEV) for n in sizes]
for lam in (0.0, 0.3, 0.7, 1.5):
    bat = M.prune_visual_tokens_uniattn_batch(embs, scs, RATE, lam)
    same = all(
        torch.equal(bat[i], M.prune_visual_tokens_uniattn(embs[i], scs[i], RATE, lam)[0])
        for i in range(len(embs))
    )
    chk(same, f"batch == per-image reference (lambda={lam})")

# 4) Triton path == eager path, byte-identical (GPU only)
if DEV == "cuda" and getattr(M, "_HAS_TRITON", False):
    for lam in (0.3, 0.7):
        for n, b in ((64, 3), (216, 5), (256, 4)):
            e = [torch.randn(n, D, device=DEV) for _ in range(b)]
            s = [torch.randn(n, device=DEV) for _ in range(b)]
            tri = M.prune_visual_tokens_uniattn_batch(e, s, RATE, lam)
            old = M._UNIATTN_MAX_N
            M._UNIATTN_MAX_N = 0                      # force eager
            try:
                eag = M.prune_visual_tokens_uniattn_batch(e, s, RATE, lam)
            finally:
                M._UNIATTN_MAX_N = old
            chk(all(torch.equal(a, c) for a, c in zip(tri, eag)),
                f"Triton == eager (N={n},B={b},lambda={lam})")
else:
    skip("Triton==eager check (needs CUDA + Triton)")

# 5) eager fallback still correct when Triton is disabled
old = M._UNIATTN_MAX_N
M._UNIATTN_MAX_N = 0
try:
    fb = M.prune_visual_tokens_uniattn_batch(embs, scs, RATE, 0.3)
    same = all(
        torch.equal(fb[i], M.prune_visual_tokens_uniattn(embs[i], scs[i], RATE, 0.3)[0])
        for i in range(len(embs))
    )
finally:
    M._UNIATTN_MAX_N = old
chk(same, "eager fallback path still byte-identical")

# 6) degenerate inputs
_, ke = M.prune_visual_tokens_uniattn(torch.zeros(0, D, device=DEV),
                                 torch.zeros(0, device=DEV), RATE, 0.5)
chk(ke.numel() == 0, "N=0 handled")
_, kr = M.prune_visual_tokens_uniattn(emb1, sc1, 0.0, 0.5)
chk(kr.numel() == N, "rate=0 keeps all")
out_all = M.prune_visual_tokens_uniattn_batch(embs, scs, 0.0, 0.3)
chk(all(torch.equal(a, b) for a, b in zip(out_all, embs)), "batch rate=0 passthrough")

# 7) lambda=0 must not need the similarity matmul (hoisted check).
#    Guard: run lambda=0 on a size that would OOM/blow up if [B,N,N] were built.
if DEV == "cuda":
    big = [torch.randn(1024, D, device=DEV) for _ in range(8)]
    bigs = [torch.randn(1024, device=DEV) for _ in range(8)]
    # Relative peak: with lambda>0 we additionally allocate the [B,N,D] stack,
    # its L2-normalised copy and the [B,N,N] matrix, so lambda=0 must come out
    # far cheaper. An absolute threshold would be wrong -- both paths allocate
    # the kept-embedding output, which dominates and is unavoidable.
    def peak_for(lam):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        res = M.prune_visual_tokens_uniattn_batch(big, bigs, RATE, lam)
        torch.cuda.synchronize()
        p = torch.cuda.max_memory_allocated() - before
        del res
        return p

    p0, p3 = peak_for(0.0), peak_for(0.3)
    chk(p0 < 0.5 * p3,
        f"lambda=0 skips similarity build (peak {p0/1e6:.0f}MB vs lambda>0 {p3/1e6:.0f}MB)")
    del big, bigs
    torch.cuda.empty_cache()
else:
    skip("lambda=0 no-similarity memory guard (needs CUDA)")

# ---------------------------------------------------------------- performance
print("\n[performance]")
if DEV != "cuda" or not getattr(M, "_HAS_TRITON", False):
    skip("performance guards (need CUDA + Triton)")
else:
    # Realistic 13-image request
    req = [196, 256, 256, 324, 256, 196, 256, 256, 196, 324, 256, 196, 256]
    e = [torch.randn(n, D, device=DEV, dtype=torch.bfloat16) for n in req]
    s = [torch.randn(n, device=DEV) for n in req]

    # GUARD 1: the Triton path is actually taken (catches silent fallback)
    used = []
    for n in sorted(set(req)):
        ss = torch.randn(4, n, device=DEV)
        sim = torch.randn(4, n, n, device=DEV)
        used.append(M._triton_greedy_select(ss, sim, max(1, int(n * 0.5)), 0.3) is not None)
    chk(all(used), f"Triton path used for all request sizes {sorted(set(req))}")

    def run(triton_on):
        old = M._UNIATTN_MAX_N
        M._UNIATTN_MAX_N = old if triton_on else 0
        try:
            M.prune_visual_tokens_uniattn_batch(e, s, RATE, 0.3)
        finally:
            M._UNIATTN_MAX_N = old

    for _ in range(2):
        run(True); run(False)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(3): run(False)
    torch.cuda.synchronize(); t_eager = (time.time() - t) / 3 * 1000
    torch.cuda.synchronize(); t = time.time()
    for _ in range(3): run(True)
    torch.cuda.synchronize(); t_tri = (time.time() - t) / 3 * 1000
    speedup = t_eager / max(t_tri, 1e-6)
    print(f"       13-img selection: eager={t_eager:.1f}ms triton={t_tri:.1f}ms "
          f"speedup={speedup:.1f}x")

    # GUARD 2: Triton must be meaningfully faster than the eager K-step loop.
    # Loose threshold on purpose: the absolute gap grows with GPU contention
    # (measured ~58x under load); 2x is the "did we lose the fusion" tripwire.
    chk(speedup >= 2.0, f"Triton >= 2x eager (got {speedup:.1f}x)")

    # GUARD 3: absolute budget for a standard request -- trips if someone
    # reintroduces per-image looping or an accidental O(K) launch pattern.
    chk(t_tri < 250.0, f"13-img selection under 250ms (got {t_tri:.1f}ms)")

print(f"\nregression: {'PASSED' if FAIL == 0 else f'FAILED ({FAIL})'}"
      f"{f', {SKIP} skipped' if SKIP else ''}")
sys.exit(0 if FAIL == 0 else 1)
