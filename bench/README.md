# Benchmark

Quality comparison of UniAttn (`attention_uniattn`) against `attention_score` (top-k) and
`unicomp` on the same lmms-eval dataset suite, plus performance records for the fused Triton kernel.

Evaluation setup: lmms-eval, 5 public sets POPE / TextVQA(val) / ScienceQA(img) / MMMU(val) / MME,
500 samples per task; **4-metric mean** = arithmetic average of POPE-acc / TextVQA / ScienceQA / MMMU.

## Key conclusions

- **UniAttn is best at the same pruning rate**: at rate 0.5, **UniAttn(λ0.3) 79.65 > atte 79.02 > unic 78.52**.
- **UniAttn0.5 beats atte0.4 across the board**: higher quality (79.65 vs 79.08) and lower ISL — better and cheaper.
- **Pruning rate dominates quality**: UniAttn 0.5→0.6→0.8→0.9 = 79.65→77.37→72.74→63.85,
  at high pruning rates the algorithmic advantage cannot make up for the token gap.
- **Cost/performance sweet spot: rate 0.5 + λ0.3**.
- **Performance**: the fused Triton kernel collapses the entire greedy loop into a single kernel launch,
  giving about **58×** speedup under GPU contention, with selection results **bit-for-bit identical** to eager.

## Files

| File | Contents |
|---|---|
| `SUMMARY_TABLE.md` | Master table of quality + ISL for three strategies × each pruning rate |
| `QUALITY_COMPARISON.md` | Quality comparison of the five configs atte0.5 / unic0.5 / uniattn0.5 / 0.8 / 0.9 |
| `COMPARE_uniattn06_vs_atte04.md` | uniattn0.6 vs atte0.4 (shows the pruning rate dominates) |
| `RESULTS.md` | Results of the first rate0.5 × λ sweep (λ∈{0,0.3,0.7}) |
| `TRITON_OPT.md` | Triton optimization notes |
| `TRITON_OPT_REPRODUCE.md` | Full reproducible record of the optimization (root cause, misjudgments, final solution) |
| `uniattn_sweep.sh` / `uniattn_quality_sweep.sh` | Restart + evaluation scripts for sweeping configs (deployment-specific, adjust per environment) |
| `restart_enc_uniattn.py` | Restart the encoder from a snapshot to switch configs (deployment-specific, adjust per environment) |
| `collect3.py` / `collect4.py` | Aggregate `results/` into tables |

> Paths, ports, and model paths in the scripts are placeholder deployment templates; adjust them yourself when changing environments.
