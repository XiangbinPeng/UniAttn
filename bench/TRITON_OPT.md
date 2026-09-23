# attention_uniattn high-concurrency latency optimization: fused Triton kernel

## Background
Greedy UniAttn is **serial** (each step depends on the previous step's max_sim), so the K steps cannot be vectorized. Done per-image, one N-image request needs
`K×num_images` serial small kernels; batching by request brings it down to K steps. But under **8 encoders sharing 1 GPU (MPS) + high concurrency**,
the **kernel-launch overhead** of these K steps still gets amplified by contention → visibly higher end-to-end latency.

## Tried but not viable: CUDA Graph
The original idea was to use CUDA Graph to replay the K steps as one. Measured on GPU: **all captures fail under the MPS deployment**, and **a failed capture attempt
pollutes the CUDA context** — afterward even ordinary `torch.randn/matmul` throws `Offset increment outside graph capture`.
Inside the encoder, this would crash the ViT forward pass of subsequent requests. **So CUDA Graph is unusable under MPS and was abandoned.**

## Adopted: put the entire greedy loop into a single Triton kernel
`_uniattn_greedy_kernel`: grid = B (one program per image), each program loops K steps **inside** the kernel:
- reads `s[b,:]` and the similarity row `sim[b,nxt,:]` from global memory;
- each step computes `uniattn = s - λ·max_sim` (selected positions set to -inf), takes argmax (**ties take the smallest index, matching torch.argmax**),
  updates chosen and max_sim;
- writes out `chosen[b,:]`.

**The whole batch selection is just 1 kernel launch**, fully eliminating the K-launch overhead; it does not touch CUDA Graph / RNG global state, so it is safe under MPS.
With eager fallback: Triton missing / N>1024 / launch error → fall back to batched eager, behavior does not regress.

## Measured (under GPU contention, "token selection" time for a 13-image request)
| path | time |
|---|---|
| eager (K serial steps) | **1926 ms** |
| Triton (1 launch) | **32.8 ms** |
| **speedup** | **≈ 58×** |

Correctness: the Triton selection result is **bit-for-bit identical** to the per-image eager reference (all N/B/λ); and ordinary CUDA ops work fine after capture
(unlike CUDA Graph, which pollutes the context).

## Conclusions
- The high-concurrency latency problem is **solved**: UniAttn's selection cost drops from "K serial kernel launches" to "1 kernel", about 58× speedup under contention.
- **Results unchanged** (bit-for-bit identical), the quality numbers stay consistent with before and need no re-measurement.
- λ=0 still goes through top-k; non-CUDA / no Triton / large N automatically fall back to eager, zero risk.

## Changes involved
Only `src/uniattn_pruning.py`: added `_uniattn_greedy_kernel` (Triton) + `_triton_greedy_select` wrapper + fallback;
`prune_visual_tokens_uniattn_batch` prefers Triton. The rest of the wiring (qwen3_vl / server_args / vit_batch_utils) is unchanged.
