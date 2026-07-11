#!/usr/bin/env bash
# ka_run.sh — run one KernelAgent optimization, persist to GCS, then tear down.
#
# Idempotent by construction: runs in its own process (env changes never leak to
# your shell), resolves paths from the script location (any cwd), cleans the
# candidate before AND after the run so repeated runs of the same candidate start
# clean. Saves the optimized kernel + logs + a condition-encoded run_meta to GCS.
#
# Usage:
#   scripts/ka_run.sh -c <candidate-dir> [--provider glm|anthropic] [--model ID]
#                     [--effort L] [--think off|low|high|max] [--rounds N]
#                     [--strategy beam_search|greedy] [--user NAME] [--gcs <dir>]
# Backend: --provider glm (default, in-cluster GLM-5.2) | anthropic (Claude).
#   anthropic needs ANTHROPIC_API_KEY in env (never bake keys into the script).
#   --model defaults: glm -> glm-5.2-504b, anthropic -> claude-opus-4-8.
#   --think: GLM reasoning_effort; on anthropic off=no thinking, else adaptive on.
#   --effort low|medium|high|max|xhigh (anthropic depth; unset=API default high).
# Identity: GCS lands under /gcs/<user>/. <user> = --user > $KA_USER > $USER >
#   whoami > anon. Pod has no $USER (root, non-login) so pass --user / export KA_USER.
# Examples:
#   scripts/ka_run.sh -c /work/candidates/grouped_gemm --think off --rounds 3
#   scripts/ka_run.sh -c /work/candidates/din_attention --provider anthropic --model claude-opus-4-8
set -uo pipefail   # not -e: we handle failures so a failed run is still saved

usage() { sed -n '2,22p' "$0"; exit "${1:-0}"; }

# ---- defaults ----
CAND=""; THINK="off"; ROUNDS=3; STRAT="beam_search"; GCS_ROOT=""; KA_USER="${KA_USER:-}"
PROVIDER="glm"; MODEL=""; EFFORT=""

# ---- args ----
while [ $# -gt 0 ]; do
  case "$1" in
    -c|--candidate) CAND="$2"; shift 2 ;;
    --provider)     PROVIDER="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --effort)       EFFORT="$2"; shift 2 ;;
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

# ---- idempotent env: reset, then set per --provider / --think ----
unset OPENAI_DISABLE_THINKING OPENAI_REASONING_EFFORT ANTHROPIC_EFFORT ANTHROPIC_THINKING
export OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-16384}"   # cross-provider output cap (anthropic honors it too)
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/local/nvidia/lib64}"
case "$PROVIDER" in
  glm)
    export KA_DEFAULT_PROVIDER=openai
    export OPENAI_MODEL="${MODEL:-glm-5.2-504b}"
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://glm52-504b.ray-clusters:8000/v1}"   # in-cluster GLM; OPENAI_API_KEY from pod secret
    ;;
  anthropic|claude)
    export KA_DEFAULT_PROVIDER=anthropic
    export OPENAI_MODEL="${MODEL:-claude-opus-4-8}"   # KA passes this straight through as the model id
    unset OPENAI_BASE_URL                             # anthropic client uses its own endpoint
    [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "ERROR: --provider anthropic needs ANTHROPIC_API_KEY in env (export it; never bake keys into the script)" >&2; exit 1; }
    [ -n "$EFFORT" ] && export ANTHROPIC_EFFORT="$EFFORT"   # low|medium|high|max|xhigh (unset = API default high)
    [ "$THINK" != off ] && export ANTHROPIC_THINKING=1      # --think != off -> adaptive thinking on (opus-4.8's only way to think)
    ;;
  *) echo "ERROR: --provider must be glm|anthropic" >&2; exit 1 ;;
esac
case "$THINK" in
  off)  export OPENAI_DISABLE_THINKING=1 ;;
  high) export OPENAI_REASONING_EFFORT=high ;;  # GLM's ONLY bounded thinking level
  max)  : ;;                                    # unbounded (both unset) — impractical on slow GLM
  low|medium)
    export OPENAI_REASONING_EFFORT="$THINK"
    # GLM-5.2 chat_template: effective_effort = 'high' if effort=='high' else 'max'.
    # low/medium collapse to Max (unbounded) -> over-think -> timeout. Kept for
    # non-GLM backends that honor them; on GLM use --think high or off.
    echo ">> WARN: GLM-5.2 maps reasoning_effort '$THINK' -> Max (unbounded); expect timeout. Use --think high|off." >&2 ;;
  *) echo "ERROR: --think must be off|low|medium|high|max" >&2; exit 1 ;;
esac

# ---- teardown BEFORE (clean start) ----
clean() { rm -rf "$CAND_DIR/opt_manager_logs" "$CAND_DIR"/optimized_kernel_*.py "$CAND_DIR/run.log"; }
clean

# ---- run (tee full stdout to run.log for live view) ----
LOG="$CAND_DIR/run.log"
[ "$PROVIDER" != glm ] && [ "$THINK" != off ] && echo ">> NOTE: provider=$PROVIDER ignores --think (no thinking param sent); running $OPENAI_MODEL default mode." >&2
echo ">> ka_run: provider=$PROVIDER model=$OPENAI_MODEL candidate=$NAME think=$THINK strategy=$STRAT rounds=$ROUNDS"
( cd "$KA_ROOT/examples" && python run_opt_manager.py --kernel-dir "$CAND_DIR" --strategy "$STRAT" --max-rounds "$ROUNDS" ) 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

# ---- extract just the final result summary ----
RESULT="$(grep -iE 'Best time:|Speedup vs|OPTIMIZATION (SUCCESSFUL|FAILED)|workers succeeded' "$LOG" | tail -8 || true)"

# ---- condition-encoded GCS dest + run_meta (full provenance) ----
DT="$(grep -oE 'torch\.(bfloat16|float32|float16)' "$CAND_DIR/problem.py" | head -1 | cut -d. -f2)"
TS="$(date +%Y%m%d-%H%M%S)"
# model slug in the path so a cross-model A/B (e.g. glm-5.2-504b vs claude-*) lands
# in separate dirs instead of colliding on everything-but-timestamp.
MSLUG="$(printf '%s' "${OPENAI_MODEL:-unknown}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9.' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')"
DEST="$GCS_ROOT/$NAME/${DT}-${MSLUG}-think-${THINK}-${STRAT}-r${ROUNDS}-${TS}"
mkdir -p "$DEST"
{
  echo "date:       $(date)"
  echo "candidate:  $NAME"
  echo "dtype:      $DT"
  echo "think:      $THINK   strategy: $STRAT   rounds: $ROUNDS"
  echo "provider:   ${KA_DEFAULT_PROVIDER:-?}   model: $OPENAI_MODEL   base: ${OPENAI_BASE_URL:-(anthropic native)}"
  echo "env:        DISABLE_THINKING=${OPENAI_DISABLE_THINKING:-} EFFORT=${OPENAI_REASONING_EFFORT:-} MAXTOK=$OPENAI_MAX_TOKENS ANTHROPIC_EFFORT=${ANTHROPIC_EFFORT:-} ANTHROPIC_THINKING=${ANTHROPIC_THINKING:-}"
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
