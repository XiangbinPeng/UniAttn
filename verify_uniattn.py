#!/usr/bin/env python3
"""Offline unit checks for the UniAttn selector. No engine / GPU / server needed.

Run after apply_uniattn.py. Exercises the installed
sglang.srt.layers.attention.uniattn_pruning against the untouched
prune_visual_tokens_dominant_only it is meant to generalise.
"""
import sys, torch

FAIL = 0
def chk(cond, msg):
    global FAIL
    print(("  ok  " if cond else "  XX  ") + msg)
    if not cond: FAIL += 1

from sglang.srt.layers.attention.uniattn_pruning import (
    prune_visual_tokens_uniattn,
    prune_visual_tokens_uniattn_batch,
)
from sglang.srt.layers.attention.visual_token_pruning import (
    prune_visual_tokens_dominant_only,
)

torch.manual_seed(0)
N, D, rate = 216, 3072, 0.4
keep_num = max(1, int(N * (1 - rate)))
emb = torch.randn(N, D)
scores = torch.randn(N)

# 1) lambda == 0 is byte-for-byte the existing top-k
e0, k0 = prune_visual_tokens_uniattn(emb, scores, rate, uniattn_lambda=0.0)
ed, kd = prune_visual_tokens_dominant_only(emb, scores, rate)
chk(torch.equal(k0, kd), f"lambda=0 keep_indices == dominant_only ({k0.numel()} tokens)")
chk(torch.equal(e0, ed), "lambda=0 embeddings == dominant_only")

# 2) exact budget and ascending (raster) order for lambda > 0
em, km = prune_visual_tokens_uniattn(emb, scores, rate, uniattn_lambda=0.5)
chk(km.numel() == keep_num, f"lambda>0 keeps exactly keep_num={keep_num} (got {km.numel()})")
chk(bool((km.sort().values == km).all()), "keep_indices ascending (raster order)")
chk(em.shape == (keep_num, D), f"embeddings shape {tuple(em.shape)} == ({keep_num}, {D})")
chk(em.dtype == emb.dtype, "output dtype preserved")

# 3) lambda > 0 actually diverges from top-k (diversity is doing something)
chk(not torch.equal(km, kd), "lambda=0.5 selection differs from top-k")

# 4) UniAttn set is more spread out: lower mean pairwise cosine than the top-k set
def mean_pair_cos(idx):
    x = torch.nn.functional.normalize(emb[idx].float(), dim=-1)
    s = x @ x.t()
    n = idx.numel()
    return (s.sum() - n) / (n * (n - 1))  # exclude diagonal
chk(mean_pair_cos(km) <= mean_pair_cos(kd) + 1e-6,
    f"UniAttn set less redundant: cos {mean_pair_cos(km):.4f} <= top-k {mean_pair_cos(kd):.4f}")

# 5) degenerate inputs
e_empty, k_empty = prune_visual_tokens_uniattn(torch.zeros(0, D), torch.zeros(0), rate, 0.5)
chk(k_empty.numel() == 0, "N=0 handled")
e_norate, k_norate = prune_visual_tokens_uniattn(emb, scores, 0.0, 0.5)
chk(k_norate.numel() == N, "rate=0 keeps all")

# 6) larger lambda spreads more (monotone-ish sanity, not strict)
c = {lam: float(mean_pair_cos(prune_visual_tokens_uniattn(emb, scores, rate, lam)[1]))
     for lam in (0.0, 0.5, 2.0)}
print(f"      mean pairwise cos by lambda: {c}")
chk(c[2.0] <= c[0.0] + 1e-6, "lambda=2 no more redundant than lambda=0")

# 7) batched entry point (what the engine calls) is byte-identical to the
#    per-image reference, across mixed sizes and lambdas.
embs = [torch.randn(n, D) for n in (64, 64, 128, 216)]
scs = [torch.randn(n) for n in (64, 64, 128, 216)]
for lam in (0.0, 0.3, 0.7):
    bat = prune_visual_tokens_uniattn_batch(embs, scs, rate, lam)
    same = all(
        torch.equal(bat[i], prune_visual_tokens_uniattn(embs[i], scs[i], rate, lam)[0])
        for i in range(len(embs))
    )
    chk(same, f"batch == per-image selection (lambda={lam})")

print("\nverify_uniattn " + ("PASSED" if FAIL == 0 else f"FAILED ({FAIL})"))
sys.exit(0 if FAIL == 0 else 1)
