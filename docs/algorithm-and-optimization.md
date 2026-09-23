# attention_uniattn: Algorithm Design, Quality Comparison, Latency Root Cause and Optimization — Summary

In one sentence: change visual token pruning from "just pick the top-k by attention" to the UniAttn greedy selection of tokens that are **both important and mutually non-redundant**,
which is optimal in quality at the same compression ratio; the high-concurrency latency problem it introduces is ultimately solved by **running the entire greedy loop inside a single Triton kernel**,
giving ~58× speedup under contention while keeping the **selection result bit-for-bit identical** — both "bit-for-bit identical" and "Triton is actually in effect" are nailed down by the
**regression tests** in Chapter 5, because otherwise the former silently breaking would void the entire quality table, and the latter silently falling back would merely be dozens of times slower without raising any error.

---

# 1. Algorithm: How UniAttn Works

## 1.1 What problem it solves

Visual token pruning needs to keep `K = max(1, int(N*(1-rate)))` tokens out of the N tokens of an image.
The existing `attention_score` strategy uses **pure top-k**: it takes the K highest ViT attention scores.

The problem: **tokens with high attention scores tend to cluster in the same small region** (where the ViT thinks it is most salient).
Top-k gives all the quota to this one cluster, so information at other locations in the image gets dropped wholesale — especially costly for "information spread across the whole image" tasks
(reading text in an image, cross-region reasoning).

## 1.2 Core formula

At each step select one token, the one with the **highest score**:

```
score(i) = attention(i)  −  λ · max_{j ∈ selected} cos(emb_i, emb_j)
             ↑ relevance          ↑ redundancy w.r.t. the selected set
```

- **attention(i)**: exactly the ViT attention scores used by `attention_score` (min-max normalized within each image).
- **redundancy term**: the cosine similarity between `i` and **the most similar token among those already selected**.
- **λ**: the diversity weight (`--uniattn-lambda`). A larger λ emphasizes "don't repeat" more strongly.

## 1.3 Execution flow (greedy, step by step)

1. **Seed**: pick the token with the highest attention (at this point the selected set is empty, so the redundancy term is 0).
2. **Iterate K−1 times**: for every not-yet-selected token compute the score above, add the largest one;
   then **update** every token's nearest similarity `max_sim` to the "selected set".
3. Stop once K tokens are picked, and output them in the original (raster) order.

## 1.4 Why it is designed this way (every choice has a reason)

| Design | Why |
|---|---|
| **"Deduct points" from attention rather than replacing it** | attention is a proven-effective relevance signal; UniAttn only adds a "don't repeat" constraint on top of it. When `λ=0` it **degenerates exactly to top-k**, so this is a **strict superset** of the original strategy, with a built-in switch and risk-free A/B. |
| **Use cosine similarity, not dot product / L2** | A vector's **magnitude** in a transformer ≈ token saliency, and "saliency" is already covered by the attention term. Cosine looks only at **direction (content)**, avoiding counting the same information twice. Clear division of labor: attention handles "how important", cosine handles "how similar the content is". |
| **Computed on the post-merger embedding, not the ViT key** | ① the embedding is the representation **actually consumed by the language model**, so it is the most fitting way to measure "will dropping it affect generation"; ② measured on this model, information retention is clearly higher (rate 0.6: ViT key 51.9% vs embedding 67.4%); ③ engineering-wise the embedding is readily available, so **the ViT needs to output no extra signal**. |
| **Take max (nearest neighbor) for redundancy, not the mean** | Judging "duplicate" means "does it collide with some already-selected token" — colliding with one already counts as duplicate. Taking the mean would be diluted by a pile of unrelated tokens. |
| **No backfill needed (vs UniComp)** | UniComp's NMS is **threshold(Uc)-driven**: the number of selected representatives is variable and may fall short of the budget → it must backfill/truncate and also auto-solve Uc. UniAttn is **budget-driven**: the loop runs a fixed K−1 steps, picking the remaining best each step, so as long as K≤N it **necessarily selects exactly K** — no threshold, no backfill, naturally satisfying the engine's fixed-length output contract. |

## 1.5 Mechanism comparison with the other two strategies

| Strategy | Selection criterion | Selection method | Uses attention |
|---|---|---|---|
| `attention_score` | attention | top-k (all at once) | Yes |
| `unicomp` | uniqueness (1−mean cosine) | threshold Uc + greedy NMS + backfill | No |
| **`attention_uniattn`** | attention − λ·cosine redundancy | greedy UniAttn (no threshold, no backfill) | Yes |

---

# 2. Quality Comparison Data

## 2.1 Evaluation setup
lmms-eval, 5 public datasets: POPE / TextVQA(val) / ScienceQA(img) / MMMU(val) / MME, **500 samples per task**,
hitting the master router. **4-task mean** = the arithmetic mean of POPE-acc / TextVQA / ScienceQA / MMMU (a single quality reference value).
**ISL** = the prompt_tokens of the same 16-image production request (**determined solely by the compression ratio, independent of strategy**, since all three share the same keep_num formula).

## 2.2 Overall table

| Strategy | Compression | Retention | ISL(16 imgs) | **4-task mean** | POPE-acc | TextVQA | ScienceQA | MMMU | MME-perc | MME-cog |
|---|---|---|---|---|---|---|---|---|---|---|
| No compression | 0.0 | 100% | 12910 | **80.74** | 86.8 | 77.2 | 97.4 | 61.6 | 358 | 185 |
| attention_score | 0.4 | 60% | 11438 | 79.08 | 87.8 | 72.1 | 97.4 | 59.0 | 358 | 152 |
| attention_score | 0.5 | 50% | 11076 | 79.02 | 87.0 | 73.3 | 96.8 | 59.0 | 359 | 148 |
| unicomp | 0.4 | 60% | 11438 | 79.35 | 86.0 | 75.2 | 94.8 | 61.4 | 358 | 168 |
| unicomp | 0.5 | 50% | 11076 | 78.52 | 85.4 | 75.1 | 94.0 | 59.6 | 352 | 142 |
| **attention_uniattn(λ0.3)** | **0.5** | **50%** | **11076** | **79.65** | 87.0 | 74.6 | 95.8 | 61.2 | 347 | 160 |
| attention_uniattn(λ0.3) | 0.6 | 40% | 10701 | 77.37 | 86.6 | 71.1 | 93.6 | 58.2 | 358 | 132 |
| attention_uniattn(λ0.3) | 0.8 | 20% | 9965 | 72.74 | 85.6 | 62.2 | 88.6 | 54.6 | 344 | 88 |
| attention_uniattn(λ0.3) | 0.9 | 10% | 9603 | 63.85 | 79.6 | 42.6 | 81.2 | 52.0 | 334 | 100 |

## 2.3 λ sweep (rate 0.5)

| λ | 4-task mean | Note |
|---|---|---|
| 0 | 79.09 | Nearly identical item-by-item to attention_score(79.02) → **a sanity check that the wiring introduces zero regression** |
| **0.3** | **79.65** | **Optimal (sweet spot)** |
| 0.7 | 79.14 | Too strong, ScienceQA/MME-cog fall back (but MMMU is highest at 61.6, matching no compression) |

## 2.4 Key conclusions

1. **At the same compression ratio, attention_uniattn is best**: rate 0.5 → **UniAttn 79.65 > atte 79.02 > unic 78.52**.
2. **The gain lands on "information spread across the whole image / reasoning" tasks** (relative to atte0.5): MMMU +2.2, MME-cog +12, TextVQA +1.3;
   the cost is a slight drop on salient-object tasks (ScienceQA −1.0, MME-perc −12) — consistent with the mechanism that "the diversity penalty squeezes out a few salient tokens".
3. **UniAttn0.5 beats atte0.4 across the board**: quality 79.65 > 79.08, and ISL is lower (11076 < 11438) → **both better and cheaper**.
4. **Compression ratio is the dominant factor for quality**: UniAttn 0.5→0.6→0.8→0.9 = 79.65→77.37→72.74→63.85.
   So **UniAttn0.6 cannot catch up to atte0.4** (77.37 vs 79.08, because it retains 40% vs 60%) — the algorithmic advantage cannot make up the token gap.
5. **ISL benefit is limited and diminishing**: visual tokens account for only about 9% of this request (the text RAG is very long), so pruning 40–50% of visual tokens only reduces ISL by 11–14%.
6. **Best cost/performance sweet spot: rate 0.5 + attention_uniattn(λ0.3)**.

---

# 3. Latency: Where It Is Slower Than attention_score

## 3.1 First, scope it: only the "select tokens" step differs

Both strategies **share the exact same ViT forward pass + attention score extraction** (attention_uniattn also needs the ViT to produce scores;
`extract_vit_attention_score` is auto-enabled for it too). So **the only timing difference is the "which K to select" step**.

## 3.2 The two computation shapes are completely different

**attention_score (top-k)**
```python
_, topk = torch.topk(scores, keep_num)   # all at once
```
- **One fused kernel**, O(N log K), **fully parallel**, touches no embedding, builds no matrix, no loop.
- Measured: **~0.3 ms** (N=256), essentially negligible.

**attention_uniattn (greedy UniAttn)**
```python
sim = normed @ normed.T          # [N,N] cosine matrix: O(N^2 · D)
for _ in range(keep_num - 1):    # K steps, serial, mutually dependent
    nxt = argmax(s - λ·max_sim)
    max_sim = maximum(max_sim, sim[nxt])
```
Two **unavoidable** costs on top of top-k:
1. **The O(N²·D) similarity matrix** (about 1e8 FLOPs/image at N≈200, D=3072) — top-k has no such step at all.
2. **K greedy steps, inherently serial**: each step depends on the `max_sim` updated by the previous one, **cannot be vectorized away**; K≈N·(1−rate),
   about a hundred-plus steps at rate 0.5, and each step still launches several small kernels.
- Measured: **~7 ms (N=256), ~28 ms (N=1024)**, i.e. **10–30×** higher than top-k.

## 3.3 Why "fine at low concurrency, explodes at high concurrency"

The key is the deployment shape: **8 encoder processes share 1 GPU via CUDA MPS**.

- **Low concurrency**: the K-step small kernels fit into the GPU queue, the selection cost is a few ms, drowned out by the ViT forward + prefill, and end-to-end **basically imperceptible**.
- **High concurrency**: 8 encoders simultaneously fire **a large number of serial small kernels** on this one card, contending with each other;
  each step's **kernel launch overhead + queuing** is amplified, and after accumulating over K steps it **degrades nonlinearly** — something that is 7 ms idle can reach hundreds of ms or even seconds under full load.

**By contrast, why the other two strategies don't have this problem**:
- `attention_score` = **1 kernel**, produces almost no contention;
- `unicomp` = **batched + fixed 16 steps** (not K steps), few kernels and independent of K.

→ **The shape "many small serial kernels" is UniAttn's unique pain point and the real source of latency.**

---

# 4. How It Was Solved (including two misjudgments, both backed by measurements)

## 4.1 Step 1: per-image → per-request batching (~6×)

The original implementation did greedy **per image**: an N-image request needed `K × num_images` serial small kernels.
Changed to processing **same-sized images within one request together**, running the K steps on `[B, N]` batched tensors (not K×num_images),
so each kernel is larger and the total serial step count is cut by roughly a factor of "num_images".
- Measured (under GPU contention, selection time for a 13-image request): **23 s → 3.7 s, about 6×**; the selection result is **bit-for-bit identical** to per-image.

## 4.2 Misjudgment #1: a measurement artifact (must remember)

The first microbenchmark measured UniAttn at ~370 ms per image, and we thought the function itself was orders of magnitude slow — **wrong**.
Cause: the benchmark script ran as a **non-MPS "rogue process"**, serialized while competing for the same card as the 8 MPS encoders.
**After joining the same MPS server (setting `CUDA_MPS_PIPE_DIRECTORY`), idle is only ~7 ms.**
> Lesson: in this environment all microbenchmarks **must run as MPS clients**, otherwise the data is fake.

## 4.3 Misjudgment #2: thinking the bottleneck was per-step synchronization

Suspecting that the GPU→CPU sync of `int(torch.argmax())` each step was the culprit, we switched to "fully on-device, zero per-step sync".
Measured under GPU contention: **the sync-free version 938 ms vs the original 625 ms (N=256) — not faster, actually slower**.
→ **Synchronization is not the root cause; the root cause is the launch/queuing overhead of K serial kernels.**

## 4.4 Tried CUDA Graph → abandoned (unusable and dangerous under MPS)

The idea was to capture the K-step loop with a CUDA Graph and replay it as a single call. Two fatal problems in GPU testing:
1. **Capture fails entirely under MPS deployment** (CUDA Graph is basically incompatible with MPS);
2. **A failed capture attempt pollutes the CUDA context** — afterward an ordinary `torch.randn + matmul` directly raises
   `Offset increment outside graph capture encountered unexpectedly`.
   Inside the encoder this would **crash the ViT forward of subsequent requests**.
→ **Abandon CUDA Graph.** (Do not go down this path again.)

## 4.5 Final solution: a single Triton kernel running the entire greedy loop (~58×)

Idea: since the pain point is "K kernel launches", **move the entire K-step loop inside one kernel**.

- **grid = B** (one program per image), each program **loops K steps inside the kernel**:
  compute `UniAttn = s − λ·max_sim` (selected set to −inf), take argmax, update chosen and max_sim,
  reading similarity rows directly from global memory.
- **The whole batch's selection needs only 1 kernel launch** → completely eliminates the K launch/queuing overheads.
- **Bit-for-bit identical result**: the kernel's arithmetic and ordering are exactly the same as eager, and the **argmax tie rule is implemented as "take the smallest index",
  matching `torch.argmax`**.
- **Safe**: no CUDA Graph involved, **does not touch global CUDA/RNG state** (ordinary CUDA ops verified normal after running), usable under MPS.
- **Has a fallback**: Triton missing / N>1024 / launch exception → automatically falls back to batched eager, **behavior never regresses**.

### Measured effect (under GPU contention, "select tokens" time for a 13-image request)

| Path | Time |
|---|---|
| eager (K serial steps) | **1926 ms** |
| **Triton (1 launch)** | **32.8 ms** |
| **Speedup** | **≈ 58×** |

Correctness: across all tested (N, B, λ) combinations, Triton's selection result is **bit-for-bit identical** to the per-image eager reference
→ **quality data is unaffected, no re-testing needed**.

## 4.6 A bonus cut: skip similarity build when λ≤0

In the batched entry, the `if uniattn_lambda <= 0.0` branch is **pure top-k** and does not read similarity at all; but this check was originally
**placed after the similarity**, so the `[B,N,D]` stack, its L2-normalized copy, and the `O(N²·D)` matmul
**were all computed and then thrown away**. Simply **moving the check before them** fixes it.

Also note: top-k **needs no min-max normalization** (a monotonic transform does not change the top-k result), so the fast path works directly on the
raw scores, consistent with the per-image reference implementation.

- Measured (8 images of N=1024): peak memory **285 MB → 50 MB** (the remaining 50 MB is the required output tensor itself, unavoidable).
- Scope: only affects the `--uniattn-lambda 0` A/B baseline path; the logic of the normal λ>0 path is unchanged.

---

# 5. Regression Tests: Locking Down Both Correctness and Performance

## 5.1 Why this test suite exists

The **entire value premise** of this round of optimization is "made it faster but **the result is exactly the same**". The moment someone changes the kernel and
the selected tokens change, the quality table in Chapter 2 is **entirely void**, and it **runs without any error at all** —
such regressions cannot be caught by human review, so the invariants must be nailed down with tests.

The same goes for performance, and it is even more insidious: **when Triton silently falls back to eager, the result is still correct, no exception is thrown, it's just dozens of times slower**.
In production this shows up as "occasional latency explosions", with a perfectly clean log. So performance also needs assertions.

> In one sentence: **correctness tests** guarantee "the optimization did not change behavior", **performance tests** guarantee "the optimization is actually still in effect".

## 5.2 Correctness invariants (runnable on CPU / GPU)

| Check | Why check this |
|---|---|
| `λ=0` **bit-for-bit identical** to `attention_score`'s top-k | This is the contract of "the switch off has zero impact" itself; also the sanity check that the original wiring was correct |
| batched entry == per-image reference implementation (λ ∈ {0, 0.3, 0.7, 1.5}) | Batching is the first layer of optimization; must prove it only "computes together" rather than "changes the algorithm" |
| **Triton == eager, bit-for-bit** (6 N/B/λ combinations) | The core assertion of the second layer of optimization. Control-group method: temporarily set `_UNIATTN_MAX_N` to 0 to force eager |
| eager fallback path still bit-for-bit identical | The fallback is the safety net, and the safety net itself must be correct |
| budget exactly K / output ascending (raster order) / shape / dtype | The engine's fixed-length + order contract; breaking it directly shifts positions |
| Degenerate inputs: N=0, rate=0, batched rate=0 pass-through | Edge cases must not raise exceptions |
| **Memory guard** for the λ=0 fast path | Proves the early check in 4.6 actually takes effect (otherwise it only "looks moved earlier") |

## 5.3 Performance guards (require CUDA + Triton)

| Guard | What it catches |
|---|---|
| **Assert the Triton path is actually taken** | **The single most critical one.** A silent fallback raises no error and the result is still correct — only an explicit assertion catches it. Method: for every N appearing in the request, call `_triton_greedy_select` directly and assert it **does not return None** |
| Speedup ≥ 2× | The trip line for "was the fusion accidentally removed". The threshold is intentionally loose — measured 58× under contention, 19× idle, 2× is just the floor |
| 13-image request selection time < 250 ms | Absolute budget. If someone reintroduces the per-image loop, or accidentally restores O(K) launches, it trips |

## 5.4 How to run + measured output

```bash
# use the sglang installed in the repo
python3 test_uniattn_regression.py
# or point directly at a uniattn_pruning.py file (runs standalone, no engine needed)
UNIATTN_MODULE=/path/to/uniattn_pruning.py python3 test_uniattn_regression.py
```
Exit code 0 = all green. The correctness part runs on CPU; the performance part auto-skips when CUDA/Triton is absent.

Measured result on GPU (all green):

```
13-img selection: eager=28.6ms triton=1.5ms speedup=19.1x     # idle
ok  Triton path used for all request sizes [196, 256, 324]
ok  lambda=0 skips similarity build (peak 50MB vs lambda>0 285MB)
regression: PASSED
```

## 5.5 A lesson from a bug: the memory guard must use a **relative** comparison

The first version asserted the extra peak memory of λ=0 as "< 24 MB", but the measured 50.4 MB **failed the test** —
yet **the code was correct**: that 50 MB is the **output tensor** of the kept embeddings (8×512×3072×4 ≈ 50 MB),
which both paths must allocate and cannot avoid; the absolute threshold was wrong from the start.

Changed to a **relative comparison**: `peak of λ=0 < 0.5 × peak of λ>0` (λ>0 additionally has the `[B,N,D]` stack + normalized copy +
`[B,N,N]` matrix) → 50 MB vs 285 MB, passes comfortably.

> Lesson: before writing a performance/resource assertion, first figure out "which part of the measured quantity is the **unavoidable baseline**",
> otherwise you are testing the baseline rather than the optimization.

---

# 6. Conclusions and Recommendations

1. **Algorithmically**: `attention_uniattn` is best in quality at the **same compression ratio** (rate0.5: 79.65 > atte 79.02 > unic 78.52),
   and **UniAttn0.5 beats atte0.4 across the board** (better + lower ISL). Recommended operating point: **rate 0.5 + λ0.3**.
2. **Do not trade performance by raising the compression ratio**: compression ratio dominates quality; UniAttn0.6 already cannot catch up to atte0.4, and 0.8/0.9 drop off a cliff.
3. **On latency**: UniAttn is inherently more expensive than top-k (one extra N² similarity matrix + K serial selection steps), and this gap **cannot be fully eliminated**;
   but through "**per-request batching + a single Triton kernel**", the amplification effect under high concurrency has been essentially removed (~58×),
   with unchanged results and a safe fallback.
4. **Switch defaults off**: without specifying `--image-pruning-strategy attention_uniattn`, behavior is exactly the same as before, zero impact;
   once enabled, `--uniattn-lambda` **defaults to 0.3** (the measured sweet spot).
5. **Regression tests are the premise for this optimization to hold long-term**: the correctness tests nail down "the result is bit-for-bit unchanged" (otherwise the quality table is void),
   the performance tests nail down "Triton is actually in use" (a silent fallback raises no error and is merely dozens of times slower). Must be run after any change to `uniattn_pruning.py`.

---

# 7. Appendix: Parameters, Files, and Pitfall Checklist

## Enable parameters (encoder side)
```bash
--image-pruning-rate 0.5 \
--image-pruning-strategy attention_uniattn \
--uniattn-lambda 0.3          # default is 0.3; passing 0 = exactly degenerates to attention_score top-k
```
Deployment-script env switches: `ENCODER_IMAGE_PRUNING_RATE` / `ENCODER_IMAGE_PRUNING_STRATEGY` / `ENCODER_uniattn_lambda`

## Code and docs (repo structure)
```
UniAttn/
  src/uniattn_pruning.py               # Algorithm: Triton kernel + batched entry + eager fallback + per-image reference
  apply_uniattn.py                     # Clean standalone patcher (against the stock engine, 8 anchor points, --check/--revert)
  verify_uniattn.py                    # Offline unit tests (incl. λ=0==top-k, batch==per-image equivalence)
  test_uniattn_regression.py           # Regression tests: correctness invariants + performance guards (see Chapter 5)
  docs/algorithm-and-optimization.md   # This document
  docs/sglang-integration.md           # How the 8 anchor points wire into sglang
  bench/SUMMARY_TABLE.md               # Overall quality+ISL table for three strategies × each compression ratio
  bench/QUALITY_COMPARISON.md          # Quality comparison across five configs
  bench/COMPARE_uniattn06_vs_atte04.md # uniattn0.6 vs atte0.4
  bench/TRITON_OPT.md                  # Triton optimization notes
  bench/TRITON_OPT_REPRODUCE.md        # Complete reproducible record of the optimization
```
After wiring into sglang, the algorithm file lands at the engine's
`python/sglang/srt/layers/attention/uniattn_pruning.py`.

## Pitfall checklist (be sure to note when reproducing)
1. **Microbenchmarks must run as MPS clients** (set `CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-epd`), otherwise you get fake slow data.
2. **Do not use CUDA Graph**: capture fails under MPS and pollutes the CUDA context.
3. **Per-step sync is not the bottleneck**: removing it is actually slower, don't spend time on it.
4. **Deployment must bring up D and master together**: D's launcher waits after "observer live" to form a cohort with the master group before starting the encoder;
   bringing up only D will hang forever with 0 encoders.
5. **Do not kill -9 the MPS process**: it leaves unreapable defunct zombies (the container PID1 doesn't reap); clean MPS with
   `echo quit | nvidia-cuda-mps-control` + deleting the pipe directory.
6. **Run only one deploy at a time**: stacking a second one while the previous is still in health-wait causes the two to clear each other's encoders.
7. **A pod restart resets the deployment script under /workspace**: the two env-passthrough lines for strategy/UniAttn-lambda added to the encoder block need to be re-applied.
8. **Before writing a resource assertion, distinguish the "unavoidable baseline"**: the λ=0 memory guard must use a relative comparison (see 5.5);
   an absolute threshold would misjudge the mandatory output tensor as a regression.
9. **Do not use `-m` for long git commit messages**: parentheses/quotes in the message get eaten by the shell, raising `pathspec ... did not match`.
   Write to a file then `git commit -F <file>`.
