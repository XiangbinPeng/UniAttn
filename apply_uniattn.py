#!/usr/bin/env python3
"""Add a standalone ``attention_uniattn`` visual-token-pruning strategy to sglang.

CLEAN / STANDALONE: anchors target a *stock* engine (no unicomp required).
UniAttn reuses the existing attention_score scoring path, so it only needs the
stock `prune_visual_tokens_dominant_only` machinery to be present.

One new file + anchored edits, each asserting its anchor occurs exactly once,
applied on top of whatever the engine currently is. Zero deletions. Idempotent.
Reversible via `.pre_uniattn_graft` backups.

  python3 apply_uniattn.py            apply
  python3 apply_uniattn.py --check    preflight only, change nothing
  python3 apply_uniattn.py --revert   restore the backups

Verified against a stock sglang build (no unicomp patch required).
"""
import argparse, os, py_compile, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIX = ".pre_uniattn_graft"
NEW_FILE = "layers/attention/uniattn_pruning.py"


def find_srt():
    if os.environ.get("SGLANG_DIR"):
        base = os.environ["SGLANG_DIR"]
    else:
        base = subprocess.run(
            [sys.executable, "-c", "import os,sglang;print(os.path.dirname(sglang.__file__))"],
            capture_output=True, text=True).stdout.strip()
    if not base or not os.path.isdir(base):
        sys.exit("sglang not found; set SGLANG_DIR=/path/to/site-packages/sglang")
    return os.path.join(base, "srt"), base


# ---------------------------------------------------------------- edits ----
# 1) import the UniAttn selector next to the stock dominant-only import
QWEN_IMPORT_OLD = '''from sglang.srt.layers.attention.visual_token_pruning import (
    prune_visual_tokens_dominant_only,
)'''
QWEN_IMPORT_NEW = '''from sglang.srt.layers.attention.visual_token_pruning import (
    prune_visual_tokens_dominant_only,
)
from sglang.srt.layers.attention.uniattn_pruning import (
    prune_visual_tokens_uniattn_batch,
)'''

# 2) __init__: stock only reads image_pruning_rate; also read strategy + uniattn_lambda
QWEN_INIT_OLD = '''        self.image_pruning_rate = getattr(
            get_global_server_args(), "image_pruning_rate", 0.0
        )'''
QWEN_INIT_NEW = '''        _args = get_global_server_args()
        self.image_pruning_rate = getattr(_args, "image_pruning_rate", 0.0)
        self.image_pruning_strategy = getattr(
            _args, "image_pruning_strategy", "attention_score"
        )
        self.uniattn_lambda = getattr(_args, "uniattn_lambda", 0.3)'''

# 3) attention_uniattn batches over the whole request (K steps, not K*num_images);
#    other strategies keep the stock per-image dominant_only loop.
QWEN_BRANCH_OLD = '''        result_parts = []
        for emb, score, grid_thw_per_image in zip(
            embeds_split, scores_split, grid_thw_list
        ):
            # dominant_only is the default strategy; with_merge is available
            # via prune_visual_tokens_with_merge() but not yet exposed as config.
            # dominant_only provides better throughput while with_merge preserves
            # more spatial context at the cost of additional computation.
            pruned_emb, _ = prune_visual_tokens_dominant_only(
                emb, score, self.image_pruning_rate
            )
            result_parts.append(pruned_emb)

        return torch.cat(result_parts, dim=0)'''
QWEN_BRANCH_NEW = '''        if self.image_pruning_strategy == "attention_uniattn":
            pruned = prune_visual_tokens_uniattn_batch(
                list(embeds_split),
                list(scores_split),
                self.image_pruning_rate,
                self.uniattn_lambda,
            )
            return torch.cat(pruned, dim=0)

        result_parts = []
        for emb, score, grid_thw_per_image in zip(
            embeds_split, scores_split, grid_thw_list
        ):
            # dominant_only is the default strategy; with_merge is available
            # via prune_visual_tokens_with_merge() but not yet exposed as config.
            # dominant_only provides better throughput while with_merge preserves
            # more spatial context at the cost of additional computation.
            pruned_emb, _ = prune_visual_tokens_dominant_only(
                emb, score, self.image_pruning_rate
            )
            result_parts.append(pruned_emb)

        return torch.cat(result_parts, dim=0)'''

# 4) auto-enable ViT attention-score extraction for attention_uniattn too
ARGS_EXTRACT_OLD = '''        if self.image_pruning_rate > 0 and self.image_pruning_strategy == "attention_score":
            self.extract_vit_attention_score = True'''
ARGS_EXTRACT_NEW = '''        if self.image_pruning_rate > 0 and self.image_pruning_strategy in (
            "attention_score",
            "attention_uniattn",
        ):
            self.extract_vit_attention_score = True'''

# 5) new server-arg field (inserted between strategy and extract fields)
ARGS_FIELDS_OLD = '''    image_pruning_strategy: str = "attention_score"
    extract_vit_attention_score: bool = False'''
ARGS_FIELDS_NEW = '''    image_pruning_strategy: str = "attention_score"
    # UniAttn selection weight for image_pruning_strategy == "attention_uniattn".
    # score(i) = attention(i) - uniattn_lambda * max_j-in-kept cos(emb_i, emb_j).
    # 0.0 reduces exactly to attention_score top-k.
    # 0.3 is the measured sweet spot (best 4-task accuracy mean at rate 0.5).
    uniattn_lambda: float = 0.3
    extract_vit_attention_score: bool = False'''

# 6) register --uniattn-lambda right after --image-pruning-strategy
ARGS_CLI_OLD = '''        parser.add_argument(
            "--image-pruning-strategy",
            type=str,
            default=ServerArgs.image_pruning_strategy,
            help="Pruning strategy: 'attention_score' or 'uniform'.",
        )'''
ARGS_CLI_NEW = '''        parser.add_argument(
            "--image-pruning-strategy",
            type=str,
            default=ServerArgs.image_pruning_strategy,
            help="Pruning strategy: 'attention_score', 'attention_uniattn', or 'uniform'.",
        )
        parser.add_argument(
            "--uniattn-lambda",
            type=float,
            default=ServerArgs.uniattn_lambda,
            help="Diversity weight for the 'attention_uniattn' pruning strategy. "
            "Each greedy step keeps argmax of attention(i) - uniattn_lambda * "
            "max cosine similarity to already-kept tokens. 0 reduces to plain "
            "attention_score top-k; larger trades attention for spatial spread.",
        )'''

# 7) Gate A: split-rate gate must let attention_uniattn through (same keep_num rule)
GATE_A_OLD = '''    if image_pruning_rate <= 0.0 or image_pruning_strategy != "attention_score":
        return 0.0'''
GATE_A_NEW = '''    if image_pruning_rate <= 0.0 or image_pruning_strategy not in (
        "attention_score",
        "attention_uniattn",
    ):
        return 0.0'''

# 8) Gate B: ViT batch max-size guard — treat attention_uniattn like attention_score
GATE_B_OLD = '''        image_pruning_rate > 0.0
        and image_pruning_strategy == "attention_score"
        and not attention_prune_batch_enabled'''
GATE_B_NEW = '''        image_pruning_rate > 0.0
        and image_pruning_strategy in ("attention_score", "attention_uniattn")
        and not attention_prune_batch_enabled'''

EDITS = [
    ("models/qwen3_vl.py", QWEN_IMPORT_OLD, QWEN_IMPORT_NEW, "1/8 qwen3_vl: import UniAttn selector"),
    ("models/qwen3_vl.py", QWEN_INIT_OLD, QWEN_INIT_NEW, "2/8 qwen3_vl: __init__ reads strategy + uniattn_lambda"),
    ("models/qwen3_vl.py", QWEN_BRANCH_OLD, QWEN_BRANCH_NEW, "3/8 qwen3_vl: attn_prune batches attention_uniattn over request"),
    ("server_args.py", ARGS_EXTRACT_OLD, ARGS_EXTRACT_NEW, "4/8 server_args: extract score for attention_uniattn"),
    ("server_args.py", ARGS_FIELDS_OLD, ARGS_FIELDS_NEW, "5/8 server_args: uniattn_lambda field"),
    ("server_args.py", ARGS_CLI_OLD, ARGS_CLI_NEW, "6/8 server_args: --uniattn-lambda CLI"),
    ("disaggregation/vit_batch_utils.py", GATE_A_OLD, GATE_A_NEW, "7/8 GATE-A: split-rate gate lets attention_uniattn through"),
    ("disaggregation/vit_batch_utils.py", GATE_B_OLD, GATE_B_NEW, "8/8 GATE-B: batch-max guard treats attention_uniattn like attention_score"),
]
TOUCHED = sorted({rel for rel, *_ in EDITS})


def preflight(srt):
    ok = True
    print(f"target : {srt}")
    src = os.path.join(HERE, "src", "uniattn_pruning.py")
    print(f"  {'ok ' if os.path.exists(src) else 'XX '} src/uniattn_pruning.py present")
    ok &= os.path.exists(src)
    for rel, old, new, label in EDITS:
        p = os.path.join(srt, rel)
        if not os.path.exists(p):
            print(f"  XX  {label}: {rel} missing"); ok = False; continue
        s = open(p, encoding="utf-8").read()
        if new in s:
            print(f"  ok  {label}: already applied")
        elif s.count(old) == 1:
            print(f"  ok  {label}: anchor found")
        else:
            print(f"  XX  {label}: anchor occurs {s.count(old)}x (need exactly 1)"); ok = False
    if subprocess.run(["pgrep", "-f", "launch_server"], capture_output=True).returncode == 0:
        print("  !!  a launch_server process is running; stop it before patching")
    return ok


def apply(srt, base):
    src = os.path.join(HERE, "src", "uniattn_pruning.py")
    dst = os.path.join(srt, NEW_FILE)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        open(dst + SUFFIX + ".WAS_ABSENT", "w").close()
    shutil.copy2(src, dst)
    print(f"  installed {NEW_FILE}")

    for rel in TOUCHED:
        p = os.path.join(srt, rel)
        bak = p + SUFFIX
        if not os.path.exists(bak):
            shutil.copy2(p, bak); print(f"  backed up {rel}")

    for rel, old, new, label in EDITS:
        p = os.path.join(srt, rel)
        s = open(p, encoding="utf-8").read()
        if new in s:
            print(f"  skip    {label} (already applied)"); continue
        assert s.count(old) == 1, f"{label}: anchor occurs {s.count(old)}x"
        open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
        print(f"  applied {label}")

    subprocess.run(["find", base, "-name", "__pycache__", "-type", "d",
                    "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
    for rel in TOUCHED + [NEW_FILE]:
        py_compile.compile(os.path.join(srt, rel), doraise=True)
    print("  bytecode purged; all touched files compile")


def revert(srt, base):
    for rel in TOUCHED + [NEW_FILE]:
        p = os.path.join(srt, rel)
        bak, absent = p + SUFFIX, p + SUFFIX + ".WAS_ABSENT"
        if os.path.exists(bak):
            shutil.move(bak, p); print(f"  restored {rel}")
        elif os.path.exists(absent):
            if os.path.exists(p): os.remove(p)
            os.remove(absent); print(f"  removed  {rel} (was new)")
        else:
            print(f"  no backup for {rel}, left as-is")
    subprocess.run(["find", base, "-name", "__pycache__", "-type", "d",
                    "-exec", "rm", "-rf", "{}", "+"], capture_output=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    srt, base = find_srt()
    if a.revert:
        revert(srt, base); sys.exit(0)
    good = preflight(srt)
    if a.check:
        print("\npreflight " + ("PASSED" if good else "FAILED")); sys.exit(0 if good else 1)
    if not good:
        sys.exit("\npreflight failed; nothing changed")
    print()
    apply(srt, base)
    print("\nnow run:  python3 verify_uniattn.py && python3 test_uniattn_regression.py")
