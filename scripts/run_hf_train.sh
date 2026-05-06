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
EOF
}

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
  train.py
)

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  cmd=("${cmd[@]:0:${#cmd[@]}-1}" "${EXTRA_ARGS[@]}" train.py)
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

echo "Submitting train.py to Hugging Face Jobs: mode=${MODE} flavor=${FLAVOR} timeout=${TIMEOUT} python=${PYTHON_VERSION}" >&2
"${cmd[@]}"
