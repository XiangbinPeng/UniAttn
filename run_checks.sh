#!/usr/bin/env bash
# One-shot self-check for the attention_uniattn port. Run from anywhere.
#
#   bash run_checks.sh            # read-only: env report + standalone regression + patch preflight
#   bash run_checks.sh --apply    # also patch the installed sglang, then verify + regression
#
# Read-only mode changes nothing and needs no engine -- safe to run first on a
# new machine to see whether the environment can even support the fast path.
set -uo pipefail
cd "$(dirname "$0")"

FAIL=0
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
note() { printf '   %s\n' "$1"; }

step "1/5 environment"
python3 - <<'PY'
import sys
print(f"   python  {sys.version.split()[0]}")
try:
    import torch
    print(f"   torch   {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   gpu     {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   torch   MISSING ({e})")
try:
    import triton
    print(f"   triton  {triton.__version__}   <- fused kernel needs this")
except Exception as e:
    print(f"   triton  MISSING ({e})  -> will fall back to eager (correct but ~58x slower)")
try:
    import os, sglang
    print(f"   sglang  {getattr(sglang,'__version__','?')}  at {os.path.dirname(sglang.__file__)}")
except Exception as e:
    print(f"   sglang  not importable ({e})  -> only the standalone regression can run")
PY
note "reference environment: torch 2.9.1+cu129 / triton 3.5.1 / stock sglang / NVIDIA L20"

step "2/5 standalone regression (src/uniattn_pruning.py, no engine needed)"
UNIATTN_MODULE=src/uniattn_pruning.py python3 test_uniattn_regression.py || FAIL=1

step "3/5 patch preflight (changes nothing)"
if python3 -c "import sglang" 2>/dev/null || [ -n "${SGLANG_DIR:-}" ]; then
  python3 apply_uniattn.py --check || FAIL=1
else
  note "SKIP: sglang not importable here -- nothing to patch."
  note "      run this on the engine host, or set SGLANG_DIR=/path/to/sglang."
fi

if [ "${1:-}" != "--apply" ]; then
  printf '\n\033[1mread-only checks %s\033[0m  (re-run with --apply to patch the engine)\n' \
    "$([ $FAIL -eq 0 ] && echo PASSED || echo FAILED)"
  exit $FAIL
fi

step "4/5 apply patch + offline unit checks"
python3 apply_uniattn.py || exit 1
python3 verify_uniattn.py || FAIL=1

step "5/5 regression against the installed module"
python3 test_uniattn_regression.py || FAIL=1

printf '\n\033[1mall checks %s\033[0m\n' "$([ $FAIL -eq 0 ] && echo PASSED || echo FAILED)"
[ $FAIL -eq 0 ] && note "enable with: --image-pruning-rate 0.5 --image-pruning-strategy attention_uniattn  (lambda defaults to 0.3)"
exit $FAIL
