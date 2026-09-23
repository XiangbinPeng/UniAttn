# Pruning strategy summary: quality scores × post-pruning ISL

Quality and post-pruning ISL for three visual-token pruning strategies (attention_score / unicomp / attention_uniattn)
at the pruning rates each was actually used with, including the no-pruning baseline. Same lmms-eval dataset suite (POPE / TextVQA /
ScienceQA / MMMU / MME, 500 samples each), dates 2026-08-15/17.

| strategy | pruning rate | retention | ISL(16 imgs) | 4-metric mean | POPE-acc | TextVQA | ScienceQA | MMMU | MME-perc | MME-cog |
|---|---|---|---|---|---|---|---|---|---|---|
| no pruning | 0.0 | 100% | 12910 | **80.74** | 86.8 | 77.2 | 97.4 | 61.6 | 358 | 185 |
| attention_score | 0.4 | 60% | 11438 | 79.08 | 87.8 | 72.1 | 97.4 | 59.0 | 358 | 152 |
| attention_score | 0.5 | 50% | 11076 | 79.02 | 87.0 | 73.3 | 96.8 | 59.0 | 359 | 148 |
| unicomp | 0.4 | 60% | 11438 | 79.35 | 86.0 | 75.2 | 94.8 | 61.4 | 358 | 168 |
| unicomp | 0.5 | 50% | 11076 | 78.52 | 85.4 | 75.1 | 94.0 | 59.6 | 352 | 142 |
| **attention_uniattn(λ0.3)** | 0.5 | 50% | 11076 | **79.65** | 87.0 | 74.6 | 95.8 | 61.2 | 347 | 160 |
| attention_uniattn(λ0.3) | 0.8 | 20% | 9965 | 72.74 | 85.6 | 62.2 | 88.6 | 54.6 | 344 | 88 |
| attention_uniattn(λ0.3) | 0.9 | 10% | 9603 | 63.85 | 79.6 | 42.6 | 81.2 | 52.0 | 334 | 100 |

## Notes

- **4-metric mean** = arithmetic average of the four accuracies (%) POPE-acc / TextVQA / ScienceQA / MMMU, a single quality reference value;
  MME is counted separately (perc = 0-2000 perception score, cog = cognition score).
- **ISL(16 imgs)** = measured prompt_tokens of the same 16-image production request (including the text RAG context).
  **ISL is determined only by the pruning rate, independent of strategy** — all three strategies use the identical
  `keep_num=int(N*(1-rate))` formula at the same rate, so at rate 0.5 all three are 11076 and at rate 0.4 all are 11438. Visual tokens are only about
  9% of that request (the text context is very long), so pruning 40~50% of visual tokens only lowers ISL by about 11~14%.
- All uniattn configs use λ=0.3.

## Key points

1. **attention_uniattn is best at the same pruning rate**: at rate 0.5, uniattn(79.65) > atte(79.02) > unic(78.52),
   and it is the only pruning scheme at 0.5 that beats every config at rate 0.4.
2. **The pruning rate is the dominant factor for quality**: uniattn drops off a cliff from 0.5→0.8→0.9, mean 79.65→72.74→63.85;
   at high pruning rates the algorithmic advantage is far from enough to offset the loss from fewer tokens (uniattn0.8 cannot match atte0.5).
3. **The ISL benefit diminishes with pruning rate and is limited**: from rate 0.5→0.9, ISL only drops from 11076 to 9603 (because visual tokens are
   a small share); yet the quality it trades away grows ever larger — the cost/performance inflection point is around rate 0.5.
4. **The sweet spot is rate 0.5 + attention_uniattn(λ0.3)**: ISL down to 11076, quality 79.65 (closest to no pruning 80.74).
