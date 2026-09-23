# How to integrate UniAttn into sglang

UniAttn is exposed as `--image-pruning-strategy attention_uniattn`, sitting alongside the
existing `attention_score` and reusing the latter's ViT attention-scoring pipeline. It **does
not depend on unicomp**. Integration is done through a **clean, self-contained, idempotent, and
reversible** anchor-based patcher, `apply_uniattn.py`: it adds 1 new algorithm file plus 8 anchor
edits to the stock engine, with **zero deletions**.

---

## 1. One-command integration

```bash
python3 apply_uniattn.py --check     # Pre-check: prints the status of all 8 anchors one by one, changes nothing
python3 apply_uniattn.py             # Apply the patch (original files backed up as *.pre_uniattn_graft, idempotent)
python3 verify_uniattn.py            # Offline unit test, must be PASSED
python3 apply_uniattn.py --revert    # Roll back
```

When the engine is not in the default site-packages: `SGLANG_DIR=/path/to/sglang python3 apply_uniattn.py`.

For every anchor, the patcher asserts that it **appears exactly once** in the target file. If it
does not match, it reports clearly at the `--check` stage which anchor it was and how many times it
occurred, and **never patches blindly**. This means it can locate issues safely even when the engine
version drifts.

---

## 2. What was changed (against the stock engine, 8 anchors, zero deletions)

```
+ srt/layers/attention/uniattn_pruning.py   New: the UniAttn selector (the algorithm itself)
M srt/models/qwen3_vl.py                     3 spots
M srt/server_args.py                         3 spots
M srt/disaggregation/vit_batch_utils.py      2 spots
```

| # | File | Anchor | Purpose |
|---|---|---|---|
| 1 | `models/qwen3_vl.py` | import | Introduce `prune_visual_tokens_uniattn_batch` next to the stock dominant-only import |
| 2 | `models/qwen3_vl.py` | `__init__` | Stock only reads `image_pruning_rate`; additionally read `image_pruning_strategy` and `uniattn_lambda` |
| 3 | `models/qwen3_vl.py` | `attn_prune` branch | `attention_uniattn` takes the **per-request batched** entry point; other strategies keep the original per-image `dominant_only` loop |
| 4 | `server_args.py` | extract-score condition | Make `attention_uniattn` also automatically enable `extract_vit_attention_score` (it likewise needs ViT to emit scores) |
| 5 | `server_args.py` | field | Add the `uniattn_lambda: float = 0.3` field |
| 6 | `server_args.py` | CLI | Register `--uniattn-lambda`, and add `attention_uniattn` to the help text of `--image-pruning-strategy` |
| 7 | `disaggregation/vit_batch_utils.py` | Gate-A (split-rate) | Allow `attention_uniattn` through (using the same keep_num formula as `attention_score`) |
| 8 | `disaggregation/vit_batch_utils.py` | Gate-B (batch-max) | Same as above, treating `attention_uniattn` like `attention_score` |

> The stock engine's `--image-pruning-strategy` has no choices restriction, so `attention_uniattn` can be passed directly.

---

## 3. Enablement parameters (encoder side)

```bash
--image-pruning-rate 0.5 \
--image-pruning-strategy attention_uniattn \
--uniattn-lambda 0.3          # Defaults to 0.3; passing 0 degrades exactly to attention_score top-k
```

- **Off by default**: when `--image-pruning-strategy attention_uniattn` is not specified, the engine
  behaves **byte-for-byte identically** to before the patch — it does not import or construct any
  UniAttn tensors, and takes the original `dominant_only` branch.
- **λ defaults to 0.3**: measured sweet spot (highest mean of the 4 accuracy metrics at rate 0.5, see `bench/`).

To pass these through from environment variables (paired with your own deployment scripts):
`ENCODER_IMAGE_PRUNING_RATE` / `ENCODER_IMAGE_PRUNING_STRATEGY` / `ENCODER_UNIATTN_LAMBDA`.

---

## 4. Why this integration is safe

- **Reuse rather than replace**: UniAttn only adds a "diversity" selection layer on top of the
  `attention_score` scoring results. With `λ=0` it degrades exactly to the original top-k, so it
  naturally comes with an A/B switch and can be rolled out gradually with no risk.
- **Anchor-based patching**: every edit is pinned to a unique anchor, is `--revert`-able, and has a
  `.pre_uniattn_graft` backup, so you can quickly tell whether it still matches after an engine upgrade.
- **Regression must run after patching**: `test_uniattn_regression.py` pins down that "selection
  results are bit-for-bit unchanged" and that "the Triton fast path is actually in effect"; see
  Chapter 5 of `docs/algorithm-and-optimization.md`.
