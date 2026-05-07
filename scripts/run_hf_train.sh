#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${HF_ENV_FILE:-$ROOT/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

FLAVOR="${HF_FLAVOR:-t4-small}"
TIMEOUT="${HF_TIMEOUT:-2h}"
MODE="${TRAIN_MODE:-baseline}"
DATASET_REPO="${HF_DATASET_REPO:-dmontgomery40/auto-research-2-synloc-data}"
MODEL_REPO="${HF_MODEL_REPO:-dmontgomery40/auto-research-2-synloc-models}"
PYTHON_VERSION="${HF_PYTHON:-3.10}"
GIT_REMOTE="${HF_GIT_REMOTE:-origin}"
GIT_REF="${HF_GIT_REF:-$(git rev-parse HEAD)}"
DRY_RUN=0
PREFLIGHT=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: TRAIN_MODE=baseline scripts/run_hf_train.sh [--dry-run|--preflight] [-- extra hf jobs args]

Submits train.py as a detached Hugging Face UV Job. This helper is mechanics
only: choose the experiment in program.md/local Codex, then run one bounded job.

Environment:
  TRAIN_MODE       baseline, detector_class_audit, transformer_baseline, rfdetr_baseline, finetune, keypoint, or point_regressor; default baseline
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

check_hf_job_write_scope() {
  if [ -z "${HF_TOKEN:-}" ]; then
    return
  fi
  set +e
  python3 - <<'PY'
import json
import os
import sys
import urllib.request

source = os.getenv("HF_WHOAMI_JSON_FILE")
try:
    if source:
        with open(source, "r", encoding="utf-8") as handle:
            info = json.load(handle)
    else:
        try:
            from huggingface_hub import HfApi
        except Exception:
            req = urllib.request.Request(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                info = json.load(response)
        else:
            info = HfApi(token=os.environ["HF_TOKEN"]).whoami()
except Exception as exc:
    print(f"Could not inspect HF_TOKEN job.write permission: {exc}", file=sys.stderr)
    sys.exit(3)

access = info.get("auth", {}).get("accessToken", {})
if access.get("role") != "fineGrained":
    sys.exit(0)
fine_grained = access.get("fineGrained", {})
permissions = set(fine_grained.get("global") or [])
for scoped in fine_grained.get("scoped") or []:
    permissions.update(scoped.get("permissions") or [])
if "job.write" in permissions:
    sys.exit(0)

name = info.get("name") or "current user"
print(
    f"HF_TOKEN for {name} is authenticated but lacks job.write; local hf CLI cannot create Jobs. "
    "Use the Hugging Face Jobs connector/app or update .env with a token that has job.write.",
    file=sys.stderr,
)
sys.exit(2)
PY
  status="$?"
  set -e
  case "$status" in
    0) return ;;
    2) exit 1 ;;
    *) return ;;
  esac
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
    --preflight)
      PREFLIGHT=1
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

if [ -z "${HF_TRAIN_SCRIPT_URL:-}" ]; then
  if ! git ls-remote "$GIT_REMOTE" | awk -v ref="$GIT_REF" '$1 == ref { found = 1 } END { exit found ? 0 : 1 }'; then
    echo "Git ref ${GIT_REF} is not present on remote ${GIT_REMOTE}; push it before submitting to HF Jobs." >&2
    exit 1
  fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
  if [ "$PREFLIGHT" -eq 1 ]; then
    if ! command -v hf >/dev/null 2>&1; then
      echo "hf CLI is required. Install/authenticate Hugging Face CLI first." >&2
      exit 1
    fi
    if [ -z "${HF_TOKEN:-}" ]; then
      hf_whoami="$(hf auth whoami 2>&1 || true)"
      if [ -z "$hf_whoami" ] || printf '%s\n' "$hf_whoami" | grep -qi '^Not logged in'; then
        echo "HF_TOKEN is not visible and hf CLI is not logged in; export HF_TOKEN before submitting." >&2
        exit 1
      fi
    fi
    check_hf_job_write_scope
    if [ -n "$(git status --porcelain)" ]; then
      echo "Working tree is dirty. Commit/stash/clean first because HF Jobs runs committed remote code, not local files." >&2
      exit 1
    fi
  fi
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

if [ -z "${HF_TOKEN:-}" ]; then
  hf_whoami="$(hf auth whoami 2>&1 || true)"
  if [ -z "$hf_whoami" ] || printf '%s\n' "$hf_whoami" | grep -qi '^Not logged in'; then
    echo "HF_TOKEN is not visible and hf CLI is not logged in; export HF_TOKEN before submitting." >&2
    exit 1
  fi
fi
check_hf_job_write_scope

echo "Submitting train.py to Hugging Face Jobs: mode=${MODE} flavor=${FLAVOR} timeout=${TIMEOUT} python=${PYTHON_VERSION} ref=${GIT_REF} script=${SCRIPT_URL}" >&2
"${cmd[@]}"
