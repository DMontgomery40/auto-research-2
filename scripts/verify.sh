#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -f README.md
test -f AGENTS.md
test -f program.md
test -f CURRENT.md
test -f LEDGER.md
test -f IDEAS.md
test -f BUDGET.md
test -f COUNCIL.md
test -f .gitignore

if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo ".env is tracked; remove it from git" >&2
  exit 1
fi

if [ -d refs ] && git status --short --ignored refs | grep -v '^!! refs/' >/dev/null; then
  echo "refs/ should stay ignored" >&2
  exit 1
fi

python3 -m py_compile scripts/ask_council.py
python3 -m py_compile scripts/download_synloc.py
python3 -m py_compile scripts/evaluate_synloc.py
python3 -m py_compile scripts/autonomy_tick.py
python3 -m py_compile cloud/synloc_smoke.py
python3 -m py_compile cloud/synloc_cache.py
python3 -m py_compile cloud/synloc_baseline_yolo.py

python3 - <<'PY'
from pathlib import Path

for path in sorted(Path("cloud").glob("*.py")):
    in_pep723 = False
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line == "# /// script":
            in_pep723 = True
            continue
        if line == "# ///":
            in_pep723 = False
            continue
        if in_pep723 and "\"git+" in line:
            raise SystemExit(
                f"{path}:{line_no}: PEP 723 URL dependency must use 'package @ git+...'"
            )
PY

python3 - <<'PY'
import json
from pathlib import Path

state = json.loads(Path("autonomy/state.json").read_text())
assert state["phase"], "autonomy phase missing"
assert state["hf_dataset_repo"], "HF dataset repo missing"
assert state["hf_model_repo"], "HF model repo missing"
PY

echo "verify ok"
