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

INTERVAL_SECONDS="${CODEX_RESEARCH_INTERVAL_SECONDS:-300}"
ITERATIONS="${CODEX_RESEARCH_ITERATIONS:-0}"
MAX_RUNTIME_SECONDS="${CODEX_RESEARCH_MAX_RUNTIME_SECONDS:-0}"
DRY_RUN=0
ALLOW_DIRTY=0
PULL=0

usage() {
  cat <<'EOF'
Usage: scripts/codex_research_loop.sh [options]

Runs repeated local Codex research passes on the control-plane Mac.
Defaults to one pass every 300 seconds, i.e. 12 local chances/hour.

Options:
  --iterations N          Number of passes to run; 0 means forever.
  --interval-seconds N    Sleep interval between passes; default: 300.
  --max-runtime-seconds N Stop after roughly N seconds; 0 means no time cap.
  --pull                  Run git pull --ff-only before each pass.
  --allow-dirty           Allow a dirty tree intentionally.
  --dry-run               Print actions without running Codex.
  -h, --help              Show this help.
EOF
}

is_non_negative_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --iterations) ITERATIONS="${2:-}"; shift ;;
    --interval-seconds) INTERVAL_SECONDS="${2:-}"; shift ;;
    --max-runtime-seconds) MAX_RUNTIME_SECONDS="${2:-}"; shift ;;
    --pull) PULL=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ! is_non_negative_int "$ITERATIONS"; then echo "--iterations must be a non-negative integer" >&2; exit 2; fi
if ! is_non_negative_int "$INTERVAL_SECONDS"; then echo "--interval-seconds must be a non-negative integer" >&2; exit 2; fi
if ! is_non_negative_int "$MAX_RUNTIME_SECONDS"; then echo "--max-runtime-seconds must be a non-negative integer" >&2; exit 2; fi
if [ "$DRY_RUN" -eq 1 ] && [ "$ITERATIONS" -eq 0 ]; then ITERATIONS=1; fi

if [ "${GITHUB_ACTIONS:-}" = "true" ] || [ "${CI:-}" = "true" ] || [ -n "${HF_JOB_ID:-}" ] || [ -n "${HUGGINGFACE_JOB_ID:-}" ]; then
  echo "Refusing to run the local research loop outside the control-plane machine." >&2
  exit 1
fi

ensure_clean_tree() {
  if [ "$ALLOW_DIRTY" -eq 1 ]; then return; fi
  if [ -n "$(git status --short)" ]; then
    echo "Working tree is dirty. Commit/stash/clean first, or pass --allow-dirty intentionally." >&2
    git status --short >&2
    exit 1
  fi
}

sync_remote() {
  if [ "$PULL" -ne 1 ]; then
    echo "Skipping git pull --ff-only (default local-first loop)." >&2
    return
  fi
  if [ "$DRY_RUN" -eq 1 ]; then echo "+ git pull --ff-only" >&2; return; fi
  git pull --ff-only
}

run_research_tick() {
  local args=()
  if [ "$ALLOW_DIRTY" -eq 1 ]; then args+=(--allow-dirty); fi
  if [ "$DRY_RUN" -eq 1 ]; then args+=(--dry-run); fi
  if [ "${#args[@]}" -gt 0 ]; then
    scripts/codex_research_tick.sh "${args[@]}"
  else
    scripts/codex_research_tick.sh
  fi
}

iteration=0
started_at="$(date +%s)"
while :; do
  iteration=$((iteration + 1))
  echo "=== local Codex research pass ${iteration} ===" >&2
  ensure_clean_tree
  sync_remote
  ensure_clean_tree
  run_research_tick
  ensure_clean_tree

  if [ "$ITERATIONS" -gt 0 ] && [ "$iteration" -ge "$ITERATIONS" ]; then break; fi
  if [ "$MAX_RUNTIME_SECONDS" -gt 0 ]; then
    now="$(date +%s)"
    if [ $((now - started_at)) -ge "$MAX_RUNTIME_SECONDS" ]; then break; fi
  fi
  if [ "$DRY_RUN" -eq 1 ] && [ "$ITERATIONS" -eq 0 ]; then break; fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ sleep ${INTERVAL_SECONDS}" >&2
    continue
  fi
  echo "Sleeping ${INTERVAL_SECONDS}s before next local research chance." >&2
  sleep "$INTERVAL_SECONDS"
done
