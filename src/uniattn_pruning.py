"""Attention + diversity (UniAttn) visual-token selection.

A drop-in alternative to the ``attention_score`` strategy's top-k. Instead of
keeping the K highest-attention tokens outright -- which tend to cluster on
whatever region the ViT found salient -- this keeps tokens that are both
attended to *and* mutually dissimilar, using the Maximal Marginal Relevance
rule from information retrieval:

    score(i) = attention(i)  -  lambda * max_{j in selected} cos(emb_i, emb_j)

At each greedy step the token maximising this is added. The first term is the
relevance the existing strategy already trusts; the second penalises picking a
token that duplicates one already kept, so the survivors spread across the
image the way UniComp's uniqueness ranking does -- but seeded by attention
rather than replacing it.

Design choices:
* ``lambda == 0`` reduces *exactly* to attention_score top-k, so this is a
  strict superset and ``--uniattn-lambda 0`` is a free A/B baseline.
* Redundancy is cosine on the post-merger embeddings (what the LM consumes);
  cosine (not dot/L2) because magnitude ~ salience, already covered by attention.
* Scores are min-max normalised per image so one lambda means the same thing
  regardless of that image's attention scale.

PERFORMANCE
-----------
Greedy UniAttn is inherently sequential (each pick depends on the previous
``max_sim`` update), so the K steps can't be vectorised away. Two layers keep
it cheap under 8-encoder MPS concurrency:

1. **Request-level batching** (prune_visual_tokens_uniattn_batch): all same-sized
   images of a request run their K steps together on a ``[B, N]`` batch.

2. **Fused Triton kernel** (_triton_greedy_select): the entire K-step greedy
   loop for the whole batch runs *inside a single kernel* -- one program per
   image loops K times over its N tokens, reading similarity rows from global
   memory. This collapses the ~K*5 tiny kernel launches (whose launch overhead
   dominates under a contended MPS GPU) into **one launch**. Unlike CUDA graphs
   it needs no capture and does not touch global CUDA/RNG state, so it is safe
   under MPS. The kernel replicates the eager op order exactly -- argmax breaks
   ties on the lowest index, matching torch.argmax -- so selection is
   byte-identical. Any failure (Triton missing, oversized N, launch error)
   falls back to the eager loop, so behaviour never regresses.

The per-image ``prune_visual_tokens_uniattn`` is kept as the reference the batched,
eager and Triton paths are all verified against.
"""

import torch
from collections import defaultdict

_MAX_BATCH_GRAPHS = 64
# Triton kernel keeps the whole image in a register block of BLOCK_N; cap N so
# register pressure stays sane. Larger images fall back to the eager batch loop.
_UNIATTN_MAX_N = 1024

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _uniattn_greedy_kernel(
        s_ptr, sim_ptr, out_ptr, N, keep_num, uniattn_lambda, BLOCK_N: tl.constexpr
    ):
        b = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        NEG = -float("inf")

        s = tl.load(s_ptr + b * N + offs, mask=mask, other=NEG)

        # seed: first = lowest index achieving max(s)  (matches torch.argmax)
        m = tl.max(s, axis=0)
        first = tl.min(tl.where(s == m, offs, N), axis=0)
        chosen = offs == first
        max_sim = tl.load(sim_ptr + b * N * N + first * N + offs, mask=mask, other=0.0)

        for _ in tl.range(0, keep_num - 1):
            uniattn_score = tl.where(chosen, NEG, s - uniattn_lambda * max_sim)
            m = tl.max(uniattn_score, axis=0)
            nxt = tl.min(tl.where(uniattn_score == m, offs, N), axis=0)
            chosen = chosen | (offs == nxt)
            srow = tl.load(
                sim_ptr + b * N * N + nxt * N + offs, mask=mask, other=0.0
            )
            max_sim = tl.maximum(max_sim, srow)

        tl.store(out_ptr + b * N + offs, chosen.to(tl.int8), mask=mask)

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def prune_visual_tokens_uniattn(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    pruning_rate: float,
    uniattn_lambda: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """UniAttn selection for a single image. Reference for the batched paths."""
    N = embeddings.shape[0]
    if N == 0:
        return embeddings, torch.empty(0, dtype=torch.long, device=embeddings.device)
    if pruning_rate <= 0.0:
        return embeddings, torch.arange(N, device=embeddings.device)

    keep_num = max(1, int(N * (1.0 - pruning_rate)))
    if keep_num >= N:
        return embeddings, torch.arange(N, device=embeddings.device)

    if uniattn_lambda <= 0.0:
        _, topk = torch.topk(scores, keep_num, sorted=False)
        keep = topk.sort().values
        return embeddings[keep], keep

    device = embeddings.device
    s = scores.detach().float()
    s = (s - s.min()) / (s.max() - s.min()).clamp(min=1e-6)
    normed = torch.nn.functional.normalize(embeddings.detach().float(), p=2, dim=-1)
    sim = normed @ normed.t()

    penalty = torch.zeros(N, device=device)
    chosen = torch.zeros(N, dtype=torch.bool, device=device)
    first = torch.argmax(s)
    chosen[first] = True
    penalty[first] = float("-inf")
    max_sim = sim[first].clone()
    for _ in range(keep_num - 1):
        uniattn_score = s - uniattn_lambda * max_sim + penalty
        nxt = torch.argmax(uniattn_score)
        chosen[nxt] = True
        penalty[nxt] = float("-inf")
        max_sim = torch.maximum(max_sim, sim[nxt])

    keep = torch.nonzero(chosen, as_tuple=False).squeeze(1)
    return embeddings[keep], keep


def _greedy_batch(s, sim, ar, keep_num, uniattn_lambda):
    """Eager batched greedy: [B,N] scores + [B,N,N] sim -> [B,N] bool chosen.

    The reference op sequence; the Triton kernel mirrors it exactly.
    """
    b, n = s.shape
    chosen = torch.zeros(b, n, dtype=torch.bool, device=s.device)
    penalty = torch.zeros(b, n, device=s.device)
    first = torch.argmax(s, dim=-1)
    chosen[ar, first] = True
    penalty[ar, first] = float("-inf")
    max_sim = sim[ar, first].clone()
    for _ in range(keep_num - 1):
        uniattn_score = s - uniattn_lambda * max_sim + penalty
        nxt = torch.argmax(uniattn_score, dim=-1)
        chosen[ar, nxt] = True
        penalty[ar, nxt] = float("-inf")
        max_sim = torch.maximum(max_sim, sim[ar, nxt])
    return chosen


def _triton_greedy_select(s, sim, keep_num, uniattn_lambda):
    """Whole greedy loop in one Triton launch. Returns [B,N] bool, or None.

    None => caller uses the eager loop (non-CUDA, no Triton, oversized N, or any
    launch failure). Result is byte-identical to _greedy_batch.
    """
    if not _HAS_TRITON or not s.is_cuda:
        return None
    b, n = s.shape
    if n > _UNIATTN_MAX_N:
        return None
    try:
        s_c = s.contiguous()
        sim_c = sim.contiguous()
        out = torch.empty(b, n, dtype=torch.int8, device=s.device)
        block_n = triton.next_power_of_2(n)
        _uniattn_greedy_kernel[(b,)](
            s_c, sim_c, out, n, int(keep_num), float(uniattn_lambda), BLOCK_N=block_n
        )
        return out.bool()
    except Exception:
        return None


def prune_visual_tokens_uniattn_batch(
    embeddings: list,
    scores: list,
    pruning_rate: float,
    uniattn_lambda: float = 0.3,
) -> list:
    """UniAttn over every image of a request, batched by token count.

    Same-N images are stacked; their K greedy steps run together via a single
    fused Triton kernel when available, else the eager loop. Selection is
    byte-identical to prune_visual_tokens_uniattn per image.
    """
    out = [None] * len(embeddings)
    if pruning_rate <= 0.0:
        return list(embeddings)

    groups = defaultdict(list)
    for i, emb in enumerate(embeddings):
        groups[emb.shape[0]].append(i)

    for n, idxs in groups.items():
        keep_num = max(1, int(n * (1.0 - pruning_rate)))
        if n == 0 or keep_num >= n:
            for i in idxs:
                out[i] = embeddings[i]
            continue

        chunk = max(1, _MAX_BATCH_GRAPHS)
        for start in range(0, len(idxs), chunk):
            sub = idxs[start:start + chunk]
            b = len(sub)
            device = embeddings[sub[0]].device
            s = torch.stack([scores[i].detach().float() for i in sub])       # [B,N]

            if uniattn_lambda <= 0.0:
                # Pure attention top-k. Checked *before* building the similarity
                # inputs on purpose: the [B,N,D] stack, the L2 normalise and the
                # O(N^2*D) matmul below feed only the diversity term, so with
                # lambda=0 they would all be computed and thrown away. Top-k also
                # needs no min-max (a monotonic transform cannot change top-k), so
                # this runs on the raw scores exactly like the per-image reference.
                chosen = torch.zeros(b, n, dtype=torch.bool, device=device)
                sel = torch.topk(s, keep_num, dim=-1, sorted=False).indices
                chosen.scatter_(1, sel, True)
            else:
                lo = s.amin(-1, keepdim=True)
                s = (s - lo) / (s.amax(-1, keepdim=True) - lo).clamp(min=1e-6)
                emb = torch.stack([embeddings[i] for i in sub])              # [B,N,D]
                normed = torch.nn.functional.normalize(
                    emb.detach().float(), p=2, dim=-1
                )
                sim = normed @ normed.transpose(1, 2)                        # [B,N,N]
                chosen = _triton_greedy_select(s, sim, keep_num, uniattn_lambda)
                if chosen is None:
                    ar = torch.arange(b, device=device)
                    chosen = _greedy_batch(s, sim, ar, keep_num, uniattn_lambda)

            for slot, i in enumerate(sub):
                keep = torch.nonzero(chosen[slot], as_tuple=False).squeeze(1)
                out[i] = embeddings[i][keep]

    return out
