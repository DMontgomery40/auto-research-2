#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FLAVOR="${HF_FLAVOR:-t4-small}"
TIMEOUT="${HF_TIMEOUT:-2h}"
MODE="${TRAIN_MODE:-baseline}"
DATASET_REPO="${HF_DATASET_REPO:-dmontgomery40/auto-research-2-synloc-data}"
MODEL_REPO="${HF_MODEL_REPO:-dmontgomery40/auto-research-2-synloc-models}"
PYTHON_VERSION="${HF_PYTHON:-3.10}"
GIT_REMOTE="${HF_GIT_REMOTE:-origin}"
GIT_REF="${HF_GIT_REF:-$(git rev-parse HEAD)}"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: TRAIN_MODE=baseline scripts/run_hf_train.sh [--dry-run] [-- extra hf jobs args]

Submits train.py as a detached Hugging Face UV Job. This helper is mechanics
only: choose the experiment in program.md/local Codex, then run one bounded job.

Environment:
  TRAIN_MODE       baseline, transformer_baseline, rfdetr_baseline, finetune, keypoint, or point_regressor; default baseline
  HF_FLAVOR        default t4-small
  HF_TIMEOUT       default 2h
  HF_PYTHON        default 3.10
  HF_DATASET_REPO  default dmontgomery40/auto-research-2-synloc-data
  HF_MODEL_REPO    default dmontgomery40/auto-research-2-synloc-models
  HF_GIT_REMOTE    git remote used to build the train.py raw URL; default origin
  HF_GIT_REF       committed git ref used for train.py; default current HEAD
  HF_TRAIN_SCRIPT_URL
                  override the reachable train.py URL for HF Jobs
EOF
}

github_raw_base() {
  local remote_url="$1"
  local repo=""

  case "$remote_url" in
    https://github.com/*)
      repo="${remote_url#https://github.com/}"
      repo="${repo%.git}"
      ;;
    git@github.com:*)
      repo="${remote_url#git@github.com:}"
      repo="${repo%.git}"
      ;;
    ssh://git@github.com/*)
      repo="${remote_url#ssh://git@github.com/}"
      repo="${repo%.git}"
      ;;
  esac

  if [ -z "$repo" ]; then
    return 1
  fi

  printf 'https://raw.githubusercontent.com/%s' "$repo"
}

remote_url="$(git config --get "remote.${GIT_REMOTE}.url" || true)"
if [ -n "${HF_TRAIN_SCRIPT_URL:-}" ]; then
  SCRIPT_URL="$HF_TRAIN_SCRIPT_URL"
else
  if [ -z "$remote_url" ]; then
    echo "No git remote named ${GIT_REMOTE}; set HF_GIT_REMOTE or HF_TRAIN_SCRIPT_URL." >&2
    exit 1
  fi
  raw_base="$(github_raw_base "$remote_url")" || {
    echo "Remote ${GIT_REMOTE} is not a GitHub URL; set HF_TRAIN_SCRIPT_URL to a reachable train.py URL." >&2
    exit 1
  }
  SCRIPT_URL="${raw_base}/${GIT_REF}/train.py"
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      ;;
  esac
  shift
done

cmd=(
  hf jobs uv run
  --flavor "$FLAVOR"
  --timeout "$TIMEOUT"
  --python "$PYTHON_VERSION"
  --detach
  --secrets HF_TOKEN
  --env "TRAIN_MODE=$MODE"
  --env "HF_DATASET_REPO=$DATASET_REPO"
  --env "HF_MODEL_REPO=$MODEL_REPO"
  "$SCRIPT_URL"
)

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  cmd=("${cmd[@]:0:${#cmd[@]}-1}" "${EXTRA_ARGS[@]}" "$SCRIPT_URL")
fi

if [ "$DRY_RUN" -eq 1 ]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is required. Install/authenticate Hugging Face CLI first." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is dirty. Commit/stash/clean first because HF Jobs runs committed remote code, not local files." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && ! hf auth whoami >/dev/null 2>&1; then
  echo "HF_TOKEN is not visible and hf CLI is not logged in; export HF_TOKEN before submitting." >&2
  exit 1
fi

if [ -z "${HF_TRAIN_SCRIPT_URL:-}" ]; then
  if ! git ls-remote "$GIT_REMOTE" | awk -v ref="$GIT_REF" '$1 == ref { found = 1 } END { exit found ? 0 : 1 }'; then
    echo "Git ref ${GIT_REF} is not present on remote ${GIT_REMOTE}; push it before submitting to HF Jobs." >&2
    exit 1
  fi
fi

echo "Submitting train.py to Hugging Face Jobs: mode=${MODE} flavor=${FLAVOR} timeout=${TIMEOUT} python=${PYTHON_VERSION} ref=${GIT_REF} script=${SCRIPT_URL}" >&2
"${cmd[@]}"
