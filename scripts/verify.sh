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

python3 - <<'PYCHECK'
import ast
from pathlib import Path

import numpy as np

train_text = Path("train.py").read_text(encoding="utf-8")
tree = ast.parse(train_text)
functions = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in {"crop_bounds", "jitter_bbox_xywh"}
]
module = ast.Module(body=functions, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {"np": np}
exec(compile(module, "train.py:point_box_helpers", "exec"), namespace)
crop_bounds = namespace["crop_bounds"]
jitter_bbox_xywh = namespace["jitter_bbox_xywh"]

cases = [
    ((10.0, 20.0, 30.0, 40.0), 100, 100, 0.15),
    ((150.0, 20.0, 30.0, 40.0), 100, 100, 0.15),
    ((-80.0, -40.0, 20.0, 20.0), 100, 100, 0.15),
    ((30.0, 140.0, 20.0, 20.0), 100, 100, 0.15),
    ((99.8, 99.8, 0.2, 0.2), 100, 100, 0.15),
]
for bbox, width, height, padding in cases:
    left, top, right, bottom = crop_bounds(bbox, width, height, padding)
    if not (0 <= left < right <= width):
        raise SystemExit(f"invalid horizontal crop for {bbox}: {(left, top, right, bottom)}")
    if not (0 <= top < bottom <= height):
        raise SystemExit(f"invalid vertical crop for {bbox}: {(left, top, right, bottom)}")

rng = np.random.default_rng(20260505)
for bbox, width, height, padding in cases:
    for _ in range(20):
        x, y, w, h = jitter_bbox_xywh(
            bbox,
            image_width=width,
            image_height=height,
            center_frac=0.20,
            scale_frac=0.30,
            rng=rng,
        )
        if not (0 <= x < width and 0 < w <= width and x + w <= width):
            raise SystemExit(f"invalid jittered horizontal box for {bbox}: {(x, y, w, h)}")
        if not (0 <= y < height and 0 < h <= height and y + h <= height):
            raise SystemExit(f"invalid jittered vertical box for {bbox}: {(x, y, w, h)}")

if 'os.getenv("RFDETR_MODEL_CLASS", "RFDETRLarge")' not in train_text:
    raise SystemExit("RF-DETR SoccerNet lane must default to RFDETRLarge; the checkpoint is not base-width")
if "model_class(pretrain_weights=str(checkpoint_path))" not in train_text:
    raise SystemExit("RF-DETR SoccerNet lane should load checkpoints through the RF-DETR public constructor")
if "model.model.model.load_state_dict(state)" in train_text:
    raise SystemExit("RF-DETR SoccerNet lane should not manually strict-load unknown architecture state")
if 'raise RuntimeError(f"Result upload failed for {run_id}") from exc' not in train_text:
    raise SystemExit("train.py must fail the run when result upload fails")
if 'def emit_autonomy_result(summary: dict[str, Any]) -> None:' not in train_text:
    raise SystemExit("train.py must emit AUTONOMY_RESULT through a shared helper")
if "emit_autonomy_result(summary)\n    upload_result(summary[\"run_id\"], upload_root)" not in train_text:
    raise SystemExit("train.py must print AUTONOMY_RESULT before strict artifact upload")
if "print(\"AUTONOMY_RESULT \" + json.dumps(summary, sort_keys=True))" in train_text:
    raise SystemExit("train.py must not print AUTONOMY_RESULT only after run_* returns")
PYCHECK

bash -n scripts/codex_research_tick.sh
bash -n scripts/codex_research_loop.sh
bash -n scripts/run_hf_train.sh

if ! rg -q 'Refusing to run from ~/Documents' scripts/codex_research_tick.sh scripts/codex_research_loop.sh; then
  echo "Codex loop scripts must refuse the damaged ~/Documents checkout path" >&2
  exit 1
fi
if ! rg -q '/Users/\.\.\. paths, ~/Documents paths, or ad hoc gists' scripts/codex_research_tick.sh; then
  echo "Codex tick prompt must ban machine-specific cloud job paths" >&2
  exit 1
fi
if ! rg -q 'AUTONOMY_RESULT score and the upload blocker' scripts/codex_research_tick.sh; then
  echo "Codex tick prompt must record scores even when artifact upload is blocked" >&2
  exit 1
fi
if ! rg -q 'HF_GIT_REF' scripts/run_hf_train.sh; then
  echo "HF train helper must pin train.py to a committed git ref" >&2
  exit 1
fi
if ! rg -q 'raw.githubusercontent.com' scripts/run_hf_train.sh; then
  echo "HF train helper must submit a reachable train.py URL, not a local path" >&2
  exit 1
fi
if ! rg -q 'Working tree is dirty.*HF Jobs runs committed remote code' scripts/run_hf_train.sh; then
  echo "HF train helper must refuse dirty local-only code before cloud submission" >&2
  exit 1
fi
if ! rg -q 'Not logged in' scripts/run_hf_train.sh; then
  echo "HF train helper must detect hf CLI's Not logged in output before submission" >&2
  exit 1
fi

scripts/codex_research_tick.sh --allow-dirty --dry-run >/tmp/auto-research-2-codex-tick.txt
scripts/codex_research_loop.sh --allow-dirty --dry-run --iterations 2 >/tmp/auto-research-2-codex-loop.txt 2>&1
scripts/run_hf_train.sh --dry-run >/tmp/auto-research-2-hf-train.txt

python3 - <<'PYCHECK'
from pathlib import Path
required_docs = {
    "program.md": ["gpt-5.5", "low", "train.py", "mAP-LocSim", "0.9809895759", "first-yolo-train", "scripts/run_hf_train.sh", "cheapest option that actually works", "outer shell loop is the loop"],
    "README.md": ["program.md", "train.py", "scripts/run_hf_train.sh", "Hugging Face Jobs are CUDA execution substrate only", "cheapest option that actually works", "outside `~/Documents`"],
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
if "https://raw.githubusercontent.com/" not in hf:
    raise SystemExit("HF train dry-run must use a reachable raw GitHub train.py URL")
for forbidden in ["/Users/", "~/Documents", "/Documents/", " file://"]:
    if forbidden in hf:
        raise SystemExit(f"HF train dry-run leaked a local path: {forbidden}")
PYCHECK

echo "verify ok"
