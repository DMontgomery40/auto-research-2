#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INTERVAL_SECONDS="${CODEX_RESEARCH_INTERVAL_SECONDS:-300}"
ITERATIONS="${CODEX_RESEARCH_ITERATIONS:-0}"
DISPATCH=1
DISPATCH_BLOCKED=0
DRY_RUN=0
ALLOW_DIRTY=0
NO_PULL=0

usage() {
  cat <<'EOF'
Usage: scripts/codex_research_loop.sh [options]

Runs the local AK-style research loop on the control-plane Mac:
  local Codex research tick -> commit/push state/code -> wake execution heartbeat

Defaults to one iteration every 300 seconds, i.e. 12 local research chances/hour.
Use --iterations N for a bounded run. Omit it to run until interrupted.

Options:
  --iterations N          Number of loop iterations to run; 0 means forever.
  --interval-seconds N    Sleep interval between iterations; default: 300.
  --no-dispatch           Do not wake the GitHub Actions heartbeat.
  --dispatch-blocked      Wake heartbeat even for blocked_* phases.
  --no-pull               Skip git pull --ff-only before each iteration.
  --allow-dirty           Allow a dirty tree for intentional dry runs.
  --dry-run               Print actions without running Codex or dispatching.
  -h, --help              Show this help.

Environment overrides:
  CODEX_RESEARCH_INTERVAL_SECONDS  default: 300
  CODEX_RESEARCH_ITERATIONS        default: 0
  CODEX_RESEARCH_MODEL             passed through to codex_research_tick.sh
  CODEX_RESEARCH_EFFORT            passed through to codex_research_tick.sh
EOF
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --iterations)
      ITERATIONS="${2:-}"
      shift
      ;;
    --interval-seconds)
      INTERVAL_SECONDS="${2:-}"
      shift
      ;;
    --no-dispatch)
      DISPATCH=0
      ;;
    --dispatch-blocked)
      DISPATCH_BLOCKED=1
      ;;
    --no-pull)
      NO_PULL=1
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      ;;
    --dry-run)
      DRY_RUN=1
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

if ! is_non_negative_int "$ITERATIONS"; then
  echo "--iterations must be a non-negative integer" >&2
  exit 2
fi

if ! is_non_negative_int "$INTERVAL_SECONDS"; then
  echo "--interval-seconds must be a non-negative integer" >&2
  exit 2
fi

if [ "$DRY_RUN" -eq 1 ] && [ "$ITERATIONS" -eq 0 ]; then
  ITERATIONS=1
fi

if [ "${GITHUB_ACTIONS:-}" = "true" ] || [ "${CI:-}" = "true" ] || [ -n "${HF_JOB_ID:-}" ] || [ -n "${HUGGINGFACE_JOB_ID:-}" ]; then
  echo "Refusing to run the local research loop outside the control-plane machine." >&2
  exit 1
fi

if [ "$DISPATCH" -eq 1 ] && ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required for heartbeat dispatch; pass --no-dispatch to skip it." >&2
  exit 1
fi

ensure_clean_tree() {
  if [ "$ALLOW_DIRTY" -eq 1 ]; then
    return
  fi
  if [ -n "$(git status --short)" ]; then
    echo "Working tree is dirty. Commit/stash/clean first, or pass --allow-dirty intentionally." >&2
    git status --short >&2
    exit 1
  fi
}

state_label() {
  python3 - <<'PY'
import json
from pathlib import Path

state = json.loads(Path("autonomy/state.json").read_text(encoding="utf-8"))
phase = state.get("phase") or "unknown"
active = state.get("active_job")
active_label = "none"
if active:
    active_label = active.get("label") or active.get("id") or "unknown"
print(f"phase={phase} active_job={active_label}")
PY
}

active_job_present() {
  python3 - <<'PY'
import json
from pathlib import Path

state = json.loads(Path("autonomy/state.json").read_text(encoding="utf-8"))
raise SystemExit(0 if state.get("active_job") else 1)
PY
}

dispatch_reason() {
  DISPATCH_BLOCKED_VALUE="$DISPATCH_BLOCKED" python3 - <<'PY'
import json
import os
from pathlib import Path

state = json.loads(Path("autonomy/state.json").read_text(encoding="utf-8"))
phase = state.get("phase") or ""
active = state.get("active_job")
dispatch_blocked = os.environ.get("DISPATCH_BLOCKED_VALUE") == "1"
if active:
    print(f"active HF job needs heartbeat inspection: {active.get('label') or active.get('id') or 'unknown'}")
    raise SystemExit(0)
if phase.startswith("blocked_") and not dispatch_blocked:
    print(f"blocked phase; no heartbeat dispatch: {phase}")
    raise SystemExit(1)
if not phase:
    print("missing phase; no heartbeat dispatch")
    raise SystemExit(1)
print(f"actionable phase: {phase}")
PY
}

sync_remote() {
  if [ "$NO_PULL" -eq 1 ]; then
    echo "Skipping git pull --ff-only (--no-pull)." >&2
    return
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ git pull --ff-only" >&2
    return
  fi
  git pull --ff-only
}

run_research_tick() {
  local args=()
  if [ "$ALLOW_DIRTY" -eq 1 ]; then
    args+=(--allow-dirty)
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    args+=(--dry-run)
  fi
  scripts/codex_research_tick.sh "${args[@]}"
}

dispatch_heartbeat_if_actionable() {
  local reason
  if [ "$DISPATCH" -ne 1 ]; then
    echo "Heartbeat dispatch disabled." >&2
    return
  fi
  if ! reason="$(dispatch_reason)"; then
    echo "$reason" >&2
    return
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "+ gh workflow run autonomy.yml --ref main  # $reason" >&2
    return
  fi
  echo "Waking autonomy heartbeat: $reason" >&2
  gh workflow run autonomy.yml --ref main
}

iteration=0
while :; do
  iteration=$((iteration + 1))
  echo "=== local research loop iteration ${iteration}: $(state_label) ===" >&2

  ensure_clean_tree
  sync_remote
  ensure_clean_tree

  if active_job_present; then
    echo "Active HF job present; skipping local Codex research and waking heartbeat for inspection." >&2
  else
    run_research_tick
    ensure_clean_tree
  fi

  dispatch_heartbeat_if_actionable

  if [ "$ITERATIONS" -gt 0 ] && [ "$iteration" -ge "$ITERATIONS" ]; then
    break
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    break
  fi
  echo "Sleeping ${INTERVAL_SECONDS}s before next local research chance." >&2
  sleep "$INTERVAL_SECONDS"
done
