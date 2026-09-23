# Reproducing attention_uniattn on Another Machine

Goal: starting from this folder, get `attention_uniattn` **installed, verified, benchmarked, deployed, and re-evaluated for quality** on a fresh machine.
Just follow the steps in order; each one tells you "how to know it passed".

---

## 0. Reference Environment (the setup that was verified end-to-end)

| Item | Version |
|---|---|
| sglang | stock build (**no unicomp patch required**) |
| torch | `2.9.1+cu129` |
| triton | `3.5.1` |
| GPU | NVIDIA **L20** |
| Deployment topology | EPD 4P1D across two nodes; on the D node, 8 encoders share a single card via **CUDA MPS** |

**The only hard dependency is torch.** A missing triton / non-CUDA environment / `N > 1024` / a failed kernel launch all fall back automatically to the eager batched loop
— **results are unchanged, just slower** (about 58x under contention). So "does it run" and "does it run fast" are two separate questions on a new machine; step 2 reports on each separately.

> It will very likely work even if the engine version changes: the patch is **anchor-based**. Each of the 8 anchors asserts that it appears **exactly once** in the target file,
> and a mismatch is reported clearly at the `--check` stage — which anchor and how many times it was found — so it **never patches blindly**.

---

## 1. One Command for the Full Self-Check

```bash
cd UniAttn
bash run_checks.sh            # read-only: environment report + standalone regression test + patch pre-check (changes nothing)
bash run_checks.sh --apply    # then apply the patch + offline unit test + regression against the installed module
```

Read-only mode **does not require the engine to be installed and does not modify any file**; run this first on a new machine. It tells you clearly whether triton is present, whether
CUDA is present (which decides whether the fast path is taken), and whether all 8 anchors match.

Below are the manual versions of each internal step, for troubleshooting.

---

## 2. Manual Steps

### 2.1 Apply the Patch

```bash
python3 apply_uniattn.py --check     # pre-check, must be PASSED; prints the per-anchor status for all 8 anchors
python3 apply_uniattn.py             # apply the patch (original files backed up as *.pre_uniattn_graft, zero deletions, idempotent)
python3 apply_uniattn.py --revert    # to roll back when needed
```

When the engine is not in the default site-packages: `SGLANG_DIR=/path/to/sglang python3 apply_uniattn.py`.

What changes (8 anchors, all under `sglang/srt/`):

```
+ layers/attention/uniattn_pruning.py        new file, the algorithm itself (byte-for-byte identical to the file in the source repo)
M models/qwen3_vl.py                     3 spots: import / __init__ reads strategy+uniattn_lambda / attn_prune branch
M server_args.py                         3 spots: extract-score condition / uniattn_lambda field / --uniattn-lambda CLI
M disaggregation/vit_batch_utils.py      2 spots: Gate-A split-rate / Gate-B batch-max, both allowing attention_uniattn through
```

### 2.2 Verify Correctness

```bash
python3 verify_uniattn.py            # requires an installed and patched engine; must be PASSED
python3 test_uniattn_regression.py   # regression test; must be PASSED
# You can also run the regression without installing the engine:
UNIATTN_MODULE=src/uniattn_pruning.py python3 test_uniattn_regression.py
```

For what the regression test does and why each check is written the way it is, see chapter 5 of `docs/algorithm-and-optimization.md`. **The two to watch**:

- `lambda=0 == attention_score top-k`: this is the very contract that "turning the switch off has zero impact".
- `Triton path used for all request sizes [...]`: **a silent fallback raises no error, still produces correct results, and is only tens of times slower**,
  so this is the only assertion that catches it. If it FAILs on a new machine, the fast path is not active (check the triton/CUDA report from step 1 first).

On a machine without CUDA the performance items are skipped automatically, and the output looks like `regression: PASSED, 4 skipped` — this is normal;
it means correctness passed and performance was not measured.

### 2.3 Measure Performance (optional, but recommended)

```bash
python3 test_uniattn_regression.py   # prints the eager-vs-triton timing and speedup for a 13-image request
```

Reference values: idle `eager=28.6ms triton=1.5ms speedup=19.1x`; **under GPU contention 1926ms → 32.8ms (~58x)**.

> ⚠️ **In this environment the microbenchmark must be run as an MPS client**: `export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-epd`.
> Otherwise your script is a "non-MPS rogue process" that competes with the 8 encoders for the card and gets serialized, and the slow numbers you measure **are fake**
> (I got burned by this the first time — measured 371ms/image, when the real number is ~7ms).

### 2.4 Deploy

Encoder-side parameters:

```bash
--image-pruning-rate 0.5 \
--image-pruning-strategy attention_uniattn \
--uniattn-lambda 0.3            # 0.3 is the default; passing 0 degrades exactly to attention_score top-k
```

When using your own encoder deploy script, go through env vars:
`ENCODER_IMAGE_PRUNING_RATE` / `ENCODER_IMAGE_PRUNING_STRATEGY` / `ENCODER_UNIATTN_LAMBDA`
(if the deploy script gets reset to its original by the environment, the two env pass-through lines in the encoder block need to be re-applied,
otherwise only rate is passed and the strategy silently falls back to `attention_score` — hit this pitfall once and wasted a whole run).

To quickly switch config when encoder processes are already running: `python3 bench/restart_enc_uniattn.py 0.5 attention_uniattn 0.3`
(inherits the remaining parameters from the current process snapshot and restarts the 8 encoders).

Deployment rules (all learned the hard way):

1. **The D node and master must start together**: after "observer live", the D launcher waits to form a cohort with master
   before it starts the 8 encoders; starting D alone will hang forever at 0 encoders (once misdiagnosed as broken MPS, and a pod was restarted for nothing).
2. **Do not `kill -9` MPS processes**: this leaves defunct zombies that the container PID1 does not reap; clean up MPS with
   `echo quit | nvidia-cuda-mps-control` + delete the pipe directory.
3. **Run only one deploy at a time**: stacking a second one while the first is still in health-wait makes the two clean up each other's encoders.
4. **Run `bash <your deploy script>` directly**, do not use `run.sh --external-script` (on failure it falls through to
   `exec run_rllm.py`, leaving a pile of junk processes).

Health checks: each encoder's `/health`, plus `cohort_ready` at the observer `:18080/membership`.

### 2.5 Re-evaluate Quality (only needed when the selection logic changed)

```bash
bash bench/uniattn_quality_sweep.sh      # lmms-eval sweep over a set of configs
python3 bench/collect4.py            # aggregate results/ into a table
```

Datasets: POPE / TextVQA(val) / ScienceQA(img) / MMMU(val) / MME, **`--limit 500` per task**, hitting the master router.
Baseline data is in `bench/SUMMARY_TABLE.md`.

> **This round's Triton optimization does not require re-evaluating quality**: the selection results are **bit-for-bit identical** to eager (there is an assertion for this in the regression test).
> Only when you change the "which tokens to select" logic do you need to re-run quality.

---

## 3. The 5 Things Most Likely to Trip You Up When Switching Machines

1. **triton not installed / too old** → silently runs eager, functionally correct but tens of times slower. Check the triton line in step 1 of `run_checks.sh`,
   and `Triton path used for all request sizes` in the regression test.
2. **microbenchmark not attached to MPS** → measures fake slow numbers (see 2.3).
3. **deploy script reset by a pod restart** → strategy silently falls back to `attention_score`, and you think you are testing UniAttn but you are not (see 2.4).
4. **engine version drift makes anchors mismatch** → `apply_uniattn.py --check` points out which anchor and how many times it was found;
   just adjust the corresponding `*_OLD` constants to match the new-version source — do not bypass the assertions.
5. **Do not try CUDA Graph**: capture fails outright under MPS, and **a failed capture pollutes the CUDA context**,
   after which even a plain `torch.randn + matmul` reports `Offset increment outside graph capture`,
   which then breaks the subsequent ViT forward as well. This road is a dead end — do not go down it again.

---

## 4. File Reference

| Path | Purpose |
|---|---|
| `src/uniattn_pruning.py` | The algorithm itself: fused Triton kernel + batched entry point + eager fallback + per-image reference implementation. **Byte-for-byte identical to the source repo's `python/sglang/srt/layers/attention/uniattn_pruning.py`** |
| `apply_uniattn.py` | The anchor patcher (against the stock engine, 8 spots, `--check` / `--revert`) |
| `verify_uniattn.py` | Offline unit test (requires an installed and patched engine) |
| `test_uniattn_regression.py` | Regression test: correctness invariants + performance guard. **Byte-for-byte identical to the repo's `test/srt/test_attention_uniattn_pruning.py`** |
| `run_checks.sh` | Runs all of the above self-checks in one command |
| `bench/` | Quality results, Triton optimization notes, config-switching and aggregation scripts |
| `docs/` | Algorithm and engineering optimization summary, sglang integration notes |
