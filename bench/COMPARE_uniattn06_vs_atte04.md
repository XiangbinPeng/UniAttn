# Quality comparison: no pruning / atte0.4 / uniattn0.5(λ0.3) / uniattn0.6(λ0.3)

**Key conclusion: uniattn0.6 cannot catch up to atte0.4.** 4-metric accuracy mean uniattn0.6=77.37 vs atte0.4=79.08
(**−1.71**). The root cause is that uniattn0.6 retains only 40% of tokens while atte0.4 retains 60% — a 1.5× difference in pruning intensity,
and the algorithmic advantage cannot make up the token gap. If you want "no worse than atte0.4", use **uniattn0.5 (79.65 > 79.08, and lower ISL)**.

lmms-eval, 5 sets × 500 samples each, uniattn configs use λ=0.3, date 2026-08-18.

## Raw scores

| benchmark | no pruning | atte0.4 | uniattn0.5 λ.3 | uniattn0.6 λ.3 |
|---|---|---|---|---|
| POPE acc | 86.8 | 87.8 | 87.0 | 86.6 |
| POPE F1 | 86.9 | 88.0 | 87.3 | 86.8 |
| TextVQA | 77.2 | 72.1 | 74.6 | 71.1 |
| ScienceQA | 97.4 | 97.4 | 95.8 | 93.6 |
| MMMU | 61.6 | 59.0 | 61.2 | 58.2 |
| MME-perc (0-2000) | 358 | 358 | 347 | 358 |
| MME-cog | 185 | 152 | 160 | 132 |

## ISL + 4-metric accuracy mean (POPE-acc, TextVQA, ScienceQA, MMMU)

| config | pruning rate | retention | ISL(16 imgs) | 4-metric mean | vs atte0.4 |
|---|---|---|---|---|---|
| no pruning | 0.0 | 100% | 12910 | **80.74** | +1.66 |
| **atte0.4** | 0.4 | 60% | 11438 | **79.08** | — |
| uniattn0.5 λ0.3 | 0.5 | 50% | 11076 | **79.65** | **+0.57** |
| uniattn0.6 λ0.3 | 0.6 | 40% | 10701 | 77.37 | **−1.71** |

## Analysis

### uniattn0.6 vs atte0.4: falls short
Slightly lower across the board: POPE −1.2 / TextVQA −1.0 / ScienceQA −3.8 / MMMU −0.8 / MME-cog −20 /
MME-perc tied. ScienceQA and MME-cog are the biggest drags. Mean −1.71, exceeding the 500-sample noise.

### Why: the pruning rate dominates
uniattn0.6 retains 40%, atte0.4 retains 60% — uniattn0.6 prunes 1.5× harder. The selection algorithm (UniAttn's
attention+diversity) does have an advantage at an equal budget, but it cannot make up the 20-percentage-point token gap.

### To beat atte0.4: use uniattn0.5, not uniattn0.6
uniattn0.5(79.65) is **both** higher quality than atte0.4 (+0.57) **and** lower ISL (11076 < 11438) —
a win-win. uniattn0.6's niche is "save ~737 more tokens (ISL), at the cost of −1.71 quality".

### In one sentence
- For **quality no worse than atte0.4**: choose **uniattn0.5(λ0.3)** (better + cheaper).
- For **more extreme token savings**: uniattn0.6 is usable, but accept about a 1.7-point quality drop.

> Note: this run temporarily restarted to the uniattn0.6 config to measure its quality, and automatically restored the **no-pruning** deployment afterward.
