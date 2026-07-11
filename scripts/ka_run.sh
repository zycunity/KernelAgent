#!/usr/bin/env bash
# ka_run.sh — run one KernelAgent optimization, persist to GCS, then tear down.
#
# Idempotent by construction: runs in its own process (env changes never leak to
# your shell), resolves paths from the script location (any cwd), cleans the
# candidate before AND after the run so repeated runs of the same candidate start
# clean. Saves the optimized kernel + logs + a condition-encoded run_meta to GCS.
#
# Usage:
#   scripts/ka_run.sh -c <candidate-dir> [--think off|low|high|max] [--rounds N]
#                     [--strategy beam_search|greedy] [--user NAME] [--gcs <dir>]
# Identity: GCS lands under /gcs/<user>/. <user> = --user > $KA_USER > $USER >
#   whoami > anon. In the pod $USER is unset (root, non-login) so pass --user or
#   export KA_USER=<you>, else output goes to /gcs/anon/.
# Examples:
#   scripts/ka_run.sh -c /work/candidates/grouped_gemm --think off --rounds 3
#   scripts/ka_run.sh -c /work/candidates/din_attention --think low --strategy greedy
set -uo pipefail   # not -e: we handle failures so a failed run is still saved

usage() { sed -n '2,17p' "$0"; exit "${1:-0}"; }

# ---- defaults ----
CAND=""; THINK="off"; ROUNDS=3; STRAT="beam_search"; GCS_ROOT=""; KA_USER="${KA_USER:-}"

# ---- args ----
while [ $# -gt 0 ]; do
  case "$1" in
    -c|--candidate) CAND="$2"; shift 2 ;;
    --think)        THINK="$2"; shift 2 ;;
    --rounds)       ROUNDS="$2"; shift 2 ;;
    --strategy)     STRAT="$2"; shift 2 ;;
    --user)         KA_USER="$2"; shift 2 ;;
    --gcs)          GCS_ROOT="$2"; shift 2 ;;
    -h|--help)      usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done
[ -n "$CAND" ] || { echo "ERROR: -c <candidate-dir> required" >&2; usage 1; }

# ---- identity: /gcs/<user>/ (pod has no $USER -> whoami=root -> anon) ----
KA_USER="${KA_USER:-${USER:-$(whoami 2>/dev/null || echo anon)}}"
GCS_ROOT="${GCS_ROOT:-/gcs/$KA_USER}"

# ---- resolve paths from script location (works from any cwd) ----
KA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAND_DIR="$(cd "$CAND" 2>/dev/null && pwd)" || { echo "ERROR: candidate dir not found: $CAND" >&2; exit 1; }
NAME="$(basename "$CAND_DIR")"
for f in problem.py input.py test.py; do
  [ -f "$CAND_DIR/$f" ] || { echo "ERROR: $CAND_DIR missing $f" >&2; exit 1; }
done

# ---- idempotent env: reset, then set per --think (GLM-5.2 honors reasoning_effort) ----
unset OPENAI_DISABLE_THINKING OPENAI_REASONING_EFFORT
export OPENAI_MODEL="${OPENAI_MODEL:-glm-5.2-504b}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://glm52-504b.ray-clusters:8000/v1}"
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-16384}"
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/local/nvidia/lib64}"
case "$THINK" in
  off)  export OPENAI_DISABLE_THINKING=1 ;;
  max)  : ;;                                   # unbounded (both unset) — impractical on slow GLM
  low|high|medium) export OPENAI_REASONING_EFFORT="$THINK" ;;
  *) echo "ERROR: --think must be off|low|high|max" >&2; exit 1 ;;
esac

# ---- teardown BEFORE (clean start) ----
clean() { rm -rf "$CAND_DIR/opt_manager_logs" "$CAND_DIR"/optimized_kernel_*.py "$CAND_DIR/run.log"; }
clean

# ---- run (tee full stdout to run.log for live view) ----
LOG="$CAND_DIR/run.log"
echo ">> ka_run: candidate=$NAME think=$THINK strategy=$STRAT rounds=$ROUNDS model=$OPENAI_MODEL"
( cd "$KA_ROOT/examples" && python run_opt_manager.py --kernel-dir "$CAND_DIR" --strategy "$STRAT" --max-rounds "$ROUNDS" ) 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

# ---- extract just the final result summary ----
RESULT="$(grep -iE 'Best time:|Speedup vs|OPTIMIZATION (SUCCESSFUL|FAILED)|workers succeeded' "$LOG" | tail -8 || true)"

# ---- condition-encoded GCS dest + run_meta (full provenance) ----
DT="$(grep -oE 'torch\.(bfloat16|float32|float16)' "$CAND_DIR/problem.py" | head -1 | cut -d. -f2)"
TS="$(date +%Y%m%d-%H%M%S)"
DEST="$GCS_ROOT/$NAME/${DT}-think-${THINK}-${STRAT}-r${ROUNDS}-${TS}"
mkdir -p "$DEST"
{
  echo "date:       $(date)"
  echo "candidate:  $NAME"
  echo "dtype:      $DT"
  echo "think:      $THINK   strategy: $STRAT   rounds: $ROUNDS"
  echo "model:      $OPENAI_MODEL   base: $OPENAI_BASE_URL"
  echo "env:        DISABLE_THINKING=${OPENAI_DISABLE_THINKING:-} EFFORT=${OPENAI_REASONING_EFFORT:-} MAXTOK=$OPENAI_MAX_TOKENS"
  echo "ka_git_sha: $(git -C "$KA_ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "exit:       $STATUS"
  echo "--- shapes (problem.py) ---"; grep -E '^[A-Z_]+ = ' "$CAND_DIR/problem.py" | sed 's/^/  /'
  echo "--- result ---"; echo "$RESULT" | sed 's/^/  /'
} > "$DEST/run_meta.txt"

# artifacts: optimized kernel + opt logs + the candidate definition (not the giant run.log)
cp -r "$CAND_DIR/opt_manager_logs" "$DEST"/ 2>/dev/null || true
cp "$CAND_DIR"/optimized_kernel_*.py "$CAND_DIR"/{problem,input,test}.py "$DEST"/ 2>/dev/null || true

echo; echo "=== saved -> $DEST ==="; cat "$DEST/run_meta.txt"

# ---- teardown AFTER (leave candidate pristine for the next run) ----
clean
exit "$STATUS"
