# Quality comparison: atte0.5 / unic0.5 / uniattn0.5 / uniattn0.8 / uniattn0.9

**Key conclusion: uniattn0.8 (λ=0.3) cannot match atte0.5 on quality.** 4-metric accuracy mean
atte0.5=79.02, uniattn0.8=72.74 (**−6.28**). The pruning rate (retaining 20% vs 50%) is the dominant factor;
the algorithmic advantage only holds at the same pruning rate — at the same rate 0.5, uniattn0.5(79.65) overtakes atte0.5 (+0.63).

---

## Setup

- Framework **lmms-eval**, 5 public sets: POPE / TextVQA(val) / ScienceQA(img) / MMMU(val) / MME
- `--limit 500` per task (500 samples), `num_concurrent=8`, hitting the master router
  `<master-router>:8081` (<multimodal-LLM>)
- All uniattn configs use λ=0.3; restart all 8 encoders before each config to clear the cache
- Date: 2026-08-15
- The controls `nopruning` / `atte0.5` / `unic0.5`(=unic0.5b) reuse existing results at the same rate and same harness

Ratio of visual tokens retained per config: atte0.5 / unic0.5 / uniattn0.5 = **50%**; uniattn0.8 = **20%**;
uniattn0.9 = **10%**. (ISL for a 16-image request: no pruning 12910 → 0.5=11076 → 0.7=10341 → 0.8=9965 → 0.9=9603)

---

## Raw scores

| benchmark | no pruning | atte0.5 | unic0.5 | uniattn0.5 λ.3 | uniattn0.8 λ.3 | uniattn0.9 λ.3 |
|---|---|---|---|---|---|---|
| POPE acc | 86.8 | 87.0 | 85.4 | 87.0 | 85.6 | 79.6 |
| POPE F1 | 86.9 | 87.3 | 85.9 | 87.3 | 85.5 | 78.9 |
| TextVQA | 77.2 | 73.3 | 75.1 | 74.6 | 62.2 | 42.6 |
| ScienceQA | 97.4 | 96.8 | 94.0 | 95.8 | 88.6 | 81.2 |
| MMMU | 61.6 | 59.0 | 59.6 | 61.2 | 54.6 | 52.0 |
| MME-perc (0-2000) | 358 | 359 | 352 | 347 | 344 | 334 |
| MME-cog | 185 | 148 | 142 | 160 | 88 | 100 |

## Retention relative to no pruning (%)

| benchmark | atte0.5 | unic0.5 | uniattn0.5 λ.3 | uniattn0.8 λ.3 | uniattn0.9 λ.3 |
|---|---|---|---|---|---|
| POPE acc | 100.2% | 98.4% | 100.2% | 98.6% | 91.7% |
| POPE F1 | 100.5% | 98.9% | 100.5% | 98.5% | 90.9% |
| TextVQA | 94.9% | 97.3% | 96.7% | 80.5% | 55.2% |
| ScienceQA | 99.4% | 96.5% | 98.4% | 91.0% | 83.4% |
| MMMU | 95.8% | 96.8% | 99.4% | 88.6% | 84.4% |
| MME-perc | 100.3% | 98.1% | 96.8% | 96.0% | 93.2% |
| MME-cog | 79.7% | 77.0% | 86.5% | 47.3% | 54.1% |

## 4-metric accuracy mean (POPE-acc, TextVQA, ScienceQA, MMMU)

| config | retention | mean | vs atte0.5 |
|---|---|---|---|
| no pruning | 100% | 80.75 | +1.73 |
| **atte0.5** | 50% | **79.02** | — |
| unic0.5 | 50% | 78.53 | −0.49 |
| **uniattn0.5 λ0.3** | 50% | **79.65** | **+0.63** |
| uniattn0.8 λ0.3 | 20% | 72.74 | −6.28 |
| uniattn0.9 λ0.3 | 10% | 63.85 | −15.17 |

---

## Analysis

### 1. Does uniattn0.8 match atte0.5? — No

uniattn0.8 is lower than atte0.5 across the board, and lags far behind on key tasks:
TextVQA −11.1, ScienceQA −8.2, MME-cog −60, MMMU −4.4, POPE −1.4, 4-metric mean −6.28.

Root cause: **2.5× fewer retained tokens** (20% vs 50%). No matter how good the selection algorithm is, it cannot recover the
30 percentage points of visual information lost. **Quality is determined first by the pruning rate, and only secondarily by which tokens are chosen.**

### 2. The algorithmic advantage only holds at "the same pruning rate"

At the same rate 0.5: **uniattn0.5(79.65) > atte0.5(79.02) > unic0.5(78.53)**. UniAttn's
"attention + diversity" really is best under an equal token budget, especially on MMMU(61.2) and MME-cog(160).
Once the pruning rate is raised, this advantage is far from enough to offset the loss from fewer tokens.

### 3. The collapse point is between rate 0.8 and 0.5

From 0.5→0.8, the mean drops off a cliff 79.65→72.74; TextVQA (reading image text) and MME-cog (cognition) are the most sensitive
— such tasks need to retain enough tokens covering the whole image, and a 20% budget is not enough. At 0.9 it collapses further to 63.85.

### 4. If the goal is "preserve atte0.5's quality at a higher pruning rate"

With the current data this is **not achievable**: both rate 0.8/0.9 are far below atte0.5. If the pruning rate must be raised, what is needed is
a stronger information-retention mechanism (such as reconnecting VisionZip's contextual merging to merge the dropped tokens instead of
simply discarding them); merely swapping the selection algorithm + raising the rate is not enough.

### 5. Boundaries

- n=500 per task, ±1~1.5% noise per single metric; but the 6~11 point gap of 0.5→0.8 far exceeds the noise and is a real trend.
- λ is fixed at 0.3; in theory rate 0.8 could get a separate λ sweep, but a 6-point gap is unlikely to be closed by tuning λ (the pruning rate dominates).
- Only quality was measured, not latency/throughput.
