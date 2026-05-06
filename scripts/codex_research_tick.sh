#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$ROOT" in
  "$HOME"/Documents/*)
    echo "Refusing to run from ~/Documents; macOS File Provider can deny Codex/Git access there. Use a non-Documents local checkout of this repo." >&2
    exit 1
    ;;
esac

MODEL="${CODEX_RESEARCH_MODEL:-gpt-5.5}"
EFFORT="${CODEX_RESEARCH_EFFORT:-low}"
ALLOW_DIRTY=0
PRINT_PROMPT=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/codex_research_tick.sh [--print-prompt] [--dry-run] [--allow-dirty]

Runs one local Codex research pass. Codex reads program.md, chooses one
bounded SynLoc experiment, edits the smallest needed surface, verifies, and
records the next action. Hugging Face Jobs remain CUDA execution only.

Environment overrides:
  CODEX_RESEARCH_MODEL   default: gpt-5.5
  CODEX_RESEARCH_EFFORT  default: low
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --print-prompt) PRINT_PROMPT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [ "${GITHUB_ACTIONS:-}" = "true" ] || [ "${CI:-}" = "true" ] || [ -n "${HF_JOB_ID:-}" ] || [ -n "${HUGGINGFACE_JOB_ID:-}" ]; then
  echo "Refusing to run Codex research outside the local control-plane machine." >&2
  exit 1
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI is not on PATH; local research tick cannot start." >&2
  exit 1
fi

if [ "$ALLOW_DIRTY" -ne 1 ] && [ -n "$(git status --short)" ]; then
  echo "Working tree is dirty. Commit/stash/clean first, or pass --allow-dirty intentionally." >&2
  git status --short >&2
  exit 1
fi

prompt_file="$(mktemp "${TMPDIR:-/tmp}/auto-research-2-codex-prompt.XXXXXX")"
trap 'rm -f "$prompt_file"' EXIT

cat > "$prompt_file" <<'EOF'
You are the local Codex researcher for this auto-research-2 checkout.

Read these first:

1. AGENTS.md
2. program.md
3. CURRENT.md
4. LEDGER.md
5. IDEAS.md

You are one iteration of the outer research loop. Finish one bounded experiment
or one concrete blocker update, then leave the repo ready for the next loop
iteration. Do not treat a successful single pass as the end of autonomy.

- Preserve the Karpathy-style shape: markdown brain, train.py as the central
  editable research surface, fixed helpers only for mechanics.
- Choose one concrete SynLoc track/pose/keypoint or direct ground-point
  experiment with expected movement in official SSKit mAP-LocSim.
- If implementation is warranted, edit the smallest needed surface.
- Use Hugging Face Jobs only for bounded CUDA execution, not reasoning.
- Submit cloud work through repo-local helpers such as scripts/run_hf_train.sh
  or commands that clone the repo and apply repo-relative patches. Do not pass
  /Users/... paths, ~/Documents paths, or ad hoc gists into cloud jobs.
- If you launch a cloud job, follow it to an official score or an explicit
  blocker before ending; do not leave a live job ambiguous.
- If a job reaches official evaluation and then artifact upload fails, record
  the printed AUTONOMY_RESULT score and the upload blocker; do not rerun only to
  bypass strict upload unless the experiment score itself was not printed.
- Record keep/discard facts in LEDGER.md and update CURRENT.md.
- When running from a clean branch/worktree, commit kept code/state changes or
  revert discarded experiment code while preserving the ledger/current facts.
- Before ending a mutating turn, run scripts/verify.sh plus any narrower check.
EOF

if [ "$PRINT_PROMPT" -eq 1 ]; then
  cat "$prompt_file"
  exit 0
fi

cmd=(codex -m "$MODEL" -c "model_reasoning_effort=\"$EFFORT\"" -a never -s danger-full-access --search -C "$ROOT" exec -)

if [ "$DRY_RUN" -eq 1 ]; then
  printf '%q ' "${cmd[@]}"
  printf '< %q\n' "$prompt_file"
  exit 0
fi

echo "Starting local Codex research tick with model=${MODEL}, reasoning=${EFFORT}" >&2
"${cmd[@]}" < "$prompt_file"
