# attention_uniattn high-concurrency latency optimization — full reproducible record

Purpose: to freeze "why it was slow, what was tried, how it was finally solved, how to verify, how to deploy" in full, for precise reproduction later.

---

## 0. Environment and prerequisites (must be identical for reproduction)

- Engine: stock sglang build.
- Topology: EPD 4P1D, two nodes.
  - **master (P node)**: Router(8081) + Prefill×4 (10000-10003, each TP1) + Decode(9998, TP4 + DP-attention + EP MoE) + MTP(NEXTN). 8×L20.
  - **encoder (D node)**: 8 encoder processes (30000-30007) **sharing 1 L20, via CUDA MPS**. Pruning runs inside `get_image_feature` on the encoder side.
- The attention_uniattn patch is already installed (this package's apply_uniattn.py, 8 anchor points + the new file uniattn_pruning.py).
- Key point: **8 processes sharing 1 GPU via MPS + high concurrency** is the fundamental constraint behind this latency problem and the optimization trade-offs.

---

## 1. Problem symptom

Under load testing (high concurrency), `attention_uniattn`'s latency/throughput is **orders of magnitude** worse than `attention_score` and `unicomp`;
a single request at low concurrency is normal (12 images ~0.85s). That is, the problem only surfaces at high concurrency.

---

## 2. Root cause analysis (includes two pitfalls / misjudgments — do not repeat them when reproducing)

### True cause
Greedy UniAttn is **serial** (each step depends on the previous step's `max_sim`), and the K=keep_num≈N·(1-rate) steps cannot be vectorized.
Done per-image, one N-image request needs `K×num_images` serial small kernels; even batching by request ([B,N]) down to K steps,
the **kernel-launch overhead of these K steps** gets amplified by contention under the 8-encoder shared-GPU MPS + high concurrency → latency explodes.
Contrast: `attention_score` = a single topk kernel, `unicomp` = a batched fixed 16 steps → neither produces this kind of "many small serial kernels", so both are stable.

### Pitfall one (measurement artifact, must read)
The first micro-benchmark measured UniAttn at ~370ms/image, leading to the belief that the function itself was orders of magnitude slow — **wrong**. The reason: the benchmark script ran as a
**non-MPS "rogue process"**, competing for the same card with the 8 MPS encoders and getting serialized.
**Lesson: all micro-benchmarks must run as an MPS client**, i.e. set:
`CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-epd CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-epd-log CUDA_VISIBLE_DEVICES=0`
(joining the same MPS server as the encoders). After switching to an MPS client, the idle cost was only ~7ms/256tok.

### Pitfall two (wrong direction)
Suspecting the per-step `int(torch.argmax())` device sync was to blame, it was changed to "fully on-device, zero per-step sync".
Measured under GPU contention: the de-synced version 938ms vs the original 625ms (N=256) — **not faster, actually slower**.
**Conclusion: sync is not the root cause; the K serial kernel launches are.**

---

## 3. CUDA Graph attempt and failure (why it was abandoned — must remember)

Idea: capture the K-step loop with CUDA Graph and replay it as 1. See the old `_graph_greedy_select` for the implementation (static buffers + cached by (N,B,keep_num,λ) + eager fallback).
**Two fatal problems measured on GPU**:
1. **All captures fail under the MPS deployment** (0/N succeed) — CUDA Graph is basically incompatible with MPS.
2. **A failed capture attempt pollutes the CUDA context**: afterward a plain `torch.randn + matmul` directly throws
   `RuntimeError: Offset increment outside graph capture encountered unexpectedly.`
   → inside the encoder this would crash the ViT forward pass of subsequent requests and drag down the encoder.

**Decision: CUDA Graph is unusable under MPS, abandoned.** (Do not go down this path again.)

---

## 4. Final solution: put the entire greedy loop into a single Triton kernel

Idea: grid = B (one program per image), **each program loops K steps inside the kernel** to do the greedy selection,
reading the similarity rows directly from global memory — **the whole batch selection is just 1 kernel launch**, fully eliminating the K-launch overhead;
it does not touch CUDA Graph / RNG global state, so it is **safe under MPS** (measured: ordinary CUDA ops work fine after capture).

### 4.1 Core kernel (the key to being bit-for-bit identical to eager: argmax ties take the smallest index, matching torch.argmax)

```python
import triton, triton.language as tl

@triton.jit
def _uniattn_greedy_kernel(s_ptr, sim_ptr, out_ptr, N, keep_num, uniattn_lambda,
                       BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    NEG = -float("inf")
    s = tl.load(s_ptr + b*N + offs, mask=mask, other=NEG)
    # seed: first = the smallest index reaching max(s) (= torch.argmax semantics)
    m = tl.max(s, axis=0)
    first = tl.min(tl.where(s == m, offs, N), axis=0)
    chosen = offs == first
    max_sim = tl.load(sim_ptr + b*N*N + first*N + offs, mask=mask, other=0.0)
    for _ in tl.range(0, keep_num - 1):
        uniattn = tl.where(chosen, NEG, s - uniattn_lambda * max_sim)
        m = tl.max(uniattn, axis=0)
        nxt = tl.min(tl.where(uniattn == m, offs, N), axis=0)
        chosen = chosen | (offs == nxt)
        srow = tl.load(sim_ptr + b*N*N + nxt*N + offs, mask=mask, other=0.0)
        max_sim = tl.maximum(max_sim, srow)
    tl.store(out_ptr + b*N + offs, chosen.to(tl.int8), mask=mask)
```

Key points:
- `BLOCK_N = triton.next_power_of_2(N)`, `mask=offs<N`; masked positions take s = -inf → never selected.
- `tl.range(0, keep_num-1)` uses a **runtime** loop bound (no unroll, no per-K recompilation).
- Inputs need to be contiguous fp32: `s [B,N]`, `sim [B,N,N]`; output `out [B,N] int8` → `.bool()`.
- Exactly the same expressions and order as eager, so the **result is bit-for-bit identical**.

### 4.2 Integration and fallback (inside `prune_visual_tokens_uniattn_batch`)
- λ=0 → straight top-k (equivalent to attention_score).
- λ>0 → `_triton_greedy_select(s, sim, keep_num, λ)`: if **non-CUDA / no Triton / N>1024 / launch error** it returns None,
  and the caller falls back to batched eager `_greedy_batch`. **Behavior never regresses.**
- See this package's `src/uniattn_pruning.py` for the full implementation (includes `_uniattn_greedy_kernel`/`_triton_greedy_select`/`_greedy_batch`/the two entry functions).

---

## 5. Verification method (copy as-is)

All run as an **MPS client** (see the three env vars in pitfall one), loading the `uniattn_pruning.py` under test directly via `importlib` (without touching the live install).

### 5.1 Correctness (Triton == per-image eager, bit-for-bit identical)
For several (N,B,λ): compare each per-image output of `prune_visual_tokens_uniattn_batch` (going through Triton) against the
`embeddings[keep]` of `prune_visual_tokens_uniattn` (per-image eager) with `torch.equal`. Also confirm `_triton_greedy_select` actually returns (no fallback) and that ordinary CUDA ops work fine after capture.
Result: **CORRECT=True, 8/8 go through Triton, POST cuda op OK**.

### 5.2 Speedup (under GPU contention)
Create a background GPU load (spin up another MPS client process `while True: a=(a@b).relu()...` to saturate the card),
and for the same 13-image request, time `_TRITON_MAX_N=0` (force eager) vs default (Triton) over 5 rounds each and take the mean.

---

## 6. Measured numbers (under GPU contention, "token selection" time for a 13-image request)

| path | time |
|---|---|
| eager (K serial steps) | **1926 ms** |
| Triton (1 launch) | **32.8 ms** |
| **speedup** | **≈ 58×** |

- Correctness: for all (N,B,λ), the Triton selection result is **bit-for-bit identical** to per-image eager.
- Safety: unlike CUDA Graph, Triton does not pollute the context (`randn+matmul` works fine after capture).
- Quality: since the selection result is unchanged, **the previous quality numbers (rate0.5: uniattn(λ0.3)≈79.6 ≥ atte≈79.0 ≥ unic≈78.5) still hold and need no re-measurement**.

---

## 7. Apply / deploy (exact steps)

1. Change only one file: overwrite the optimized `src/uniattn_pruning.py` onto the engine's installed location
   `/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/uniattn_pruning.py`
   (the rest of the wiring qwen3_vl/server_args/vit_batch_utils is unchanged; if the engine is a fresh stock build, first run `python3 apply_uniattn.py` to apply the full patch set, then overwrite this one file).
2. `python3 -c "import py_compile;py_compile.compile(<that file>,doraise=True)"` to confirm it compiles.
3. Restart the deployment scripts on **D + master together** (the cohort must start on both sides, otherwise the encoders get stuck at observer-live and do not start):
   - D: `ENCODER_IMAGE_PRUNING_RATE=<r> ENCODER_IMAGE_PRUNING_STRATEGY=attention_uniattn ENCODER_UNIATTN_LAMBDA=<λ> bash <your encoder deploy script>`
   - master: `bash <your deploy script>`
   - Note: if the deploy script gets reset to the original by the environment, you need to re-apply the two env-passthrough lines for the encoder block's strategy/uniattn-lambda (see this package's README / restart script).
4. Verify: all 8 encoders /health return 200; master router/decode/prefill + cohort_ready=true; end-to-end image requests are correct.

---

## 8. Changed files

```
Only src/uniattn_pruning.py:
  + _uniattn_greedy_kernel        (Triton, entire greedy loop)
  + _triton_greedy_select     (wrapper + fallback)
    _greedy_batch             (batched eager, for fallback/reference)
    prune_visual_tokens_uniattn   (per-image eager, single-test baseline)
    prune_visual_tokens_uniattn_batch (entry: λ0→topk / otherwise Triton→eager fallback)
The 3 wiring files are unchanged.
```

---

## 9. One-sentence reproduction points
- Environment: stock engine + 8encoder/1GPU/MPS.
- Do not use CUDA Graph (capture fails under MPS and pollutes the context).
- Use "stuff the entire greedy loop into a single Triton kernel (1 launch, argmax ties take the smallest index)", with eager fallback.
- Micro-benchmarks must run as an MPS client.
- ~58× speedup under contention, results bit-for-bit unchanged.
