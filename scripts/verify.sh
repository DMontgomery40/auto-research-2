#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -f README.md
test -f AGENTS.md
test -f program.md
test -f CURRENT.md
test -f LEDGER.md
test -f IDEAS.md
test -f COUNCIL.md
test -f .gitignore
test -f train.py
test -f scripts/run_hf_train.sh

for removed in GOAL.md JOURNAL.md COUNCIL_DOSSIER.md RESEARCH_PRIORS.md BUDGET.md scripts/autonomy_tick.py scripts/codex_research_prompt.py .github/workflows/autonomy.yml autonomy/state.json autonomy/events.jsonl; do
  if [ -e "$removed" ]; then
    echo "$removed should not exist in the collapsed Karpathy-style harness" >&2
    exit 1
  fi
done

if git ls-files --error-unmatch .env >/dev/null 2>&1; then echo ".env is tracked; remove it from git" >&2; exit 1; fi
if [ -d refs ] && git status --short --ignored refs | grep -v '^!! refs/' >/dev/null; then echo "refs/ should stay ignored" >&2; exit 1; fi

python3 -m py_compile scripts/ask_council.py
python3 -m py_compile scripts/download_synloc.py
python3 -m py_compile scripts/evaluate_synloc.py
python3 -m py_compile train.py
python3 -m py_compile cloud/synloc_cache.py

bash -n scripts/codex_research_tick.sh
bash -n scripts/codex_research_loop.sh
bash -n scripts/run_hf_train.sh

scripts/codex_research_tick.sh --allow-dirty --dry-run >/tmp/auto-research-2-codex-tick.txt
scripts/codex_research_loop.sh --allow-dirty --dry-run --iterations 2 >/tmp/auto-research-2-codex-loop.txt 2>&1
scripts/run_hf_train.sh --dry-run >/tmp/auto-research-2-hf-train.txt

python3 - <<'PYCHECK'
from pathlib import Path
required_docs = {
    "program.md": ["gpt-5.5", "low", "train.py", "mAP-LocSim", "0.9809895759", "first-yolo-train", "scripts/run_hf_train.sh", "cheapest option that actually works", "outer shell loop is the loop"],
    "README.md": ["program.md", "train.py", "scripts/run_hf_train.sh", "Hugging Face Jobs are CUDA execution substrate only", "cheapest option that actually works"],
    "AGENTS.md": ["program.md", "train.py", "Hugging Face Jobs are CUDA execution substrate only", "0.9809895759", "cheapest option that actually works"],
    "CURRENT.md": ["0.9809895759040843", "3.572767401302389e-06", "track/pose/keypoint", "cheapest option that actually works"],
    "LEDGER.md": ["first-yolo-train", "discard", "0.000825082508250825"],
}
for filename, needles in required_docs.items():
    text = Path(filename).read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing: raise SystemExit(f"{filename} missing required collapsed-loop clauses: {missing}")
for filename in ["program.md", "README.md", "AGENTS.md", "CURRENT.md", "IDEAS.md"]:
    text = Path(filename).read_text(encoding="utf-8")
    forbidden = ["autonomy/state.json", "autonomy/events.jsonl", "scripts/autonomy_tick.py", ".github/workflows/autonomy.yml", "devkit_oracle_review", "train_result_review", "BUDGET.md", "$50", "$28"]
    bad = [item for item in forbidden if item in text]
    if bad: raise SystemExit(f"{filename} still references removed controller machinery: {bad}")
tick = Path("/tmp/auto-research-2-codex-tick.txt").read_text(encoding="utf-8")
for needle in ["codex", "gpt-5.5", "model_reasoning_effort", "low", "--search", "exec"]:
    if needle not in tick: raise SystemExit(f"codex tick dry-run missing {needle}")
loop = Path("/tmp/auto-research-2-codex-loop.txt").read_text(encoding="utf-8")
if "codex" not in loop or "exec" not in loop: raise SystemExit("codex loop dry-run did not reach the Codex tick")
if "local Codex research pass 1" not in loop or "local Codex research pass 2" not in loop: raise SystemExit("codex loop dry-run did not prove more than one pass")
if "default local-first loop" not in loop: raise SystemExit("codex loop dry-run did not prove local-first default")
hf = Path("/tmp/auto-research-2-hf-train.txt").read_text(encoding="utf-8")
for needle in ["hf", "jobs", "uv", "run", "--detach", "--secrets", "HF_TOKEN", "train.py"]:
    if needle not in hf: raise SystemExit(f"HF train dry-run missing {needle}")
PYCHECK

echo "verify ok"
