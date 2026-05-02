#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${CODEX_RESEARCH_MODEL:-gpt-5.5}"
EFFORT="${CODEX_RESEARCH_EFFORT:-xhigh}"
ALLOW_DIRTY=0
ALLOW_ACTIVE_JOB=0
PRINT_PROMPT=0

usage() {
  cat <<'EOF'
Usage: scripts/codex_research_tick.sh [--print-prompt] [--allow-dirty] [--allow-active-job]

Runs one local Codex research tick. Research and code edits happen on this
machine with Codex; Hugging Face Jobs remain CUDA execution substrate only.

Environment overrides:
  CODEX_RESEARCH_MODEL   default: gpt-5.5
  CODEX_RESEARCH_EFFORT  default: xhigh
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --print-prompt)
      PRINT_PROMPT=1
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      ;;
    --allow-active-job)
      ALLOW_ACTIVE_JOB=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
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

if [ "$ALLOW_ACTIVE_JOB" -ne 1 ]; then
  python3 - <<'PY'
import json
from pathlib import Path

state = json.loads(Path("autonomy/state.json").read_text(encoding="utf-8"))
active = state.get("active_job")
if active:
    label = active.get("label") or active.get("id") or "unknown"
    raise SystemExit(f"Active HF job is still running ({label}); do not start a new research tick.")
PY
fi

prompt_file="$(mktemp "${TMPDIR:-/tmp}/codex-research-prompt.XXXXXX.md")"
last_message="autonomy/codex_last_message.md"
trap 'rm -f "$prompt_file"' EXIT

python3 scripts/codex_research_prompt.py > "$prompt_file"

if [ "$PRINT_PROMPT" -eq 1 ]; then
  cat "$prompt_file"
  exit 0
fi

echo "Starting local Codex research tick with model=${MODEL}, reasoning=${EFFORT}" >&2
codex exec \
  -m "$MODEL" \
  -c "model_reasoning_effort=\"$EFFORT\"" \
  -a never \
  -s danger-full-access \
  --search \
  -C "$ROOT" \
  -o "$last_message" \
  - < "$prompt_file"
