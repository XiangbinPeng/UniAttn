# UniAttn

> **Keep the *distinct* high-attention visual tokens** — attention-guided, diversity-aware
> visual token pruning. A drop-in alternative to attention-score top-k that keeps tokens which
> are both **highly attended** *and* **mutually distinct**, wired into sglang as
> `--image-pruning-strategy attention_uniattn`.

In multimodal LLMs a single image produces hundreds of visual tokens. When you must keep only a
fraction of them at a given pruning rate, the common approach is plain **top-k**: keep the K
tokens with the highest ViT attention scores. The problem: **high-attention tokens tend to
cluster in the same small salient region**, so top-k spends the whole budget on that one cluster
and drops information everywhere else in the image. That hurts tasks where information is spread
across the whole image — reading text in an image, cross-region reasoning.

**UniAttn's core idea: keep tokens that are both high-attention (relevant) and not redundant with
what's already kept (distinct).**

## Algorithm

Each image has N post-merger tokens; keep `K = max(1, int(N*(1-rate)))`. Tokens are selected
greedily, one per step, taking the highest-scoring token each time:

```
score(i) = attention(i)  −  λ · max_{j ∈ selected} cos(emb_i, emb_j)
              ↑ relevance                  ↑ redundancy vs. the kept set
```

- **attention(i)**: the same ViT attention score the `attention_score` strategy uses, min-max
  normalized within each image.
- **redundancy term**: cosine similarity to the *most similar* already-kept token (computed on
  the post-merger embeddings).
- **λ**: the diversity weight (`--uniattn-lambda`). **λ=0 reduces exactly to top-k**, giving a
  built-in A/B switch.

This is the **Maximal Marginal Relevance (MMR)** idea from information retrieval: attention as
the relevance term, cosine similarity as the redundancy penalty, yielding a set of
"high-attention but mutually distinct" tokens. No threshold, no back-filling — it is
budget-driven, so as long as K≤N it always keeps exactly K. See
[`docs/algorithm-and-optimization.md`](docs/algorithm-and-optimization.md).

## Engineering optimization

Greedy selection is inherently **sequential** (each step depends on the `max_sim` updated by the
previous one), so the K steps cannot be vectorized away. Under "8 encoders sharing 1 GPU via
CUDA MPS + high concurrency", the launch overhead of those K small kernels is amplified by
contention and latency explodes. Two layers of optimization:

1. **Per-request batching**: same-sized images within a request are done together, running the K
   steps on a `[B, N]` batch (not K × num_images).
2. **Fused Triton kernel**: the entire K-step greedy loop is moved *inside a single kernel* — one
   launch handles the whole batch, eliminating the K launch/queue overheads. It touches no CUDA
   Graph and no global CUDA/RNG state, so it is MPS-safe, with an eager fallback.

Measured **~58× speedup** under GPU contention, with selection results **byte-identical** to
eager (pinned by the regression suite). Details and pitfalls (why CUDA Graph was abandoned, why
per-step sync is not the bottleneck) are in
[`bench/TRITON_OPT_REPRODUCE.md`](bench/TRITON_OPT_REPRODUCE.md).

## sglang integration

Wired in as a standalone, idempotent, revertible anchored patch: one new algorithm file + 8
anchored edits, zero deletions. It reuses the `attention_score` scoring path and **does not
depend on unicomp**.

```bash
python3 apply_uniattn.py --check     # preflight, changes nothing
python3 apply_uniattn.py             # apply the patch (backs up to *.pre_uniattn_graft, idempotent)
python3 apply_uniattn.py --revert    # roll back
```

Enable it (encoder side):

```bash
--image-pruning-rate 0.5 --image-pruning-strategy attention_uniattn --uniattn-lambda 0.3
```

What each of the 8 anchors changes and why is in
[`docs/sglang-integration.md`](docs/sglang-integration.md).

## Benchmark

lmms-eval, 5 public sets (POPE / TextVQA / ScienceQA / MMMU / MME), 500 samples per task.
**4-metric mean** = arithmetic mean of POPE-acc / TextVQA / ScienceQA / MMMU.

| Strategy | Pruning rate | 4-metric mean |
|---|---|---|
| no pruning | 0.0 | 80.74 |
| attention_score (top-k) | 0.5 | 79.02 |
| unicomp | 0.5 | 78.52 |
| **attention_uniattn (λ0.3)** | **0.5** | **79.65** ← best at equal rate |

- At equal pruning rate, **UniAttn > atte > unic**; **UniAttn0.5 also beats atte0.4 across the
  board** (higher quality + lower ISL).
- Pruning rate dominates quality: UniAttn 0.5→0.6→0.8→0.9 = 79.65→77.37→72.74→63.85.
- Sweet spot: **rate 0.5 + λ0.3**.

Full comparison tables in [`bench/`](bench/README.md).

## Quick self-check

```bash
bash run_checks.sh          # read-only: env report + standalone regression + patch preflight (changes nothing)
bash run_checks.sh --apply  # also apply the patch + offline unit checks + regression
```

Read-only mode needs no engine and changes no files. The only hard dependency is torch; if triton
is missing / non-CUDA / N>1024 / a kernel launch fails, it falls back to eager (same result, just
slower). To reproduce on another machine, see [`REPRODUCE.md`](REPRODUCE.md).

## Project layout

```
UniAttn/
├── src/uniattn_pruning.py          # the algorithm: Triton kernel + batched entry + eager fallback + per-image reference
├── apply_uniattn.py                # anchored patcher (targets a stock engine, 8 anchors, --check / --revert)
├── verify_uniattn.py               # offline unit checks
├── test_uniattn_regression.py      # regression suite: correctness invariants + performance guards
├── run_checks.sh                   # one-command self-check
├── REPRODUCE.md                    # reproduction guide for another machine
├── docs/
│   ├── algorithm-and-optimization.md   # algorithm design + latency root cause + optimization + regression tests (deep dive)
│   └── sglang-integration.md           # how the 8 anchors wire into sglang
└── bench/                          # quality and performance benchmarks
```

## License

[MIT](LICENSE)

