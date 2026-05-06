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
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

train_text = Path("train.py").read_text(encoding="utf-8")
tree = ast.parse(train_text)
functions = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {
        "crop_bounds",
        "jitter_bbox_xywh",
        "image_path",
        "image_path_for_record",
        "image_path_and_scale_for_record",
        "scale_xywh",
        "scale_xy",
        "box_xyxy_to_annotation_scale",
        "synloc_snapshot_patterns",
    }
]
module = ast.Module(body=functions, type_ignores=[])
ast.fix_missing_locations(module)
namespace = {"np": np, "Path": Path, "Image": Image, "Any": object}
exec(compile(module, "train.py:point_box_helpers", "exec"), namespace)
crop_bounds = namespace["crop_bounds"]
jitter_bbox_xywh = namespace["jitter_bbox_xywh"]
image_path = namespace["image_path"]
image_path_for_record = namespace["image_path_for_record"]
image_path_and_scale_for_record = namespace["image_path_and_scale_for_record"]
scale_xywh = namespace["scale_xywh"]
scale_xy = namespace["scale_xy"]
box_xyxy_to_annotation_scale = namespace["box_xyxy_to_annotation_scale"]
synloc_snapshot_patterns = namespace["synloc_snapshot_patterns"]

if synloc_snapshot_patterns("fullhd", ["valid"]) != [
    "raw/fullhd/annotations.zip",
    "raw/fullhd/manifest.json",
    "raw/fullhd/val.zip",
]:
    raise SystemExit("validation-only SynLoc fetch must avoid train.zip")
if synloc_snapshot_patterns("fullhd", ["train", "valid", "val"]) != [
    "raw/fullhd/annotations.zip",
    "raw/fullhd/manifest.json",
    "raw/fullhd/train.zip",
    "raw/fullhd/val.zip",
]:
    raise SystemExit("train+valid SynLoc fetch must deduplicate val.zip aliases")
if synloc_snapshot_patterns("fullhd", ["challenge"]) != [
    "raw/fullhd/annotations.zip",
    "raw/fullhd/manifest.json",
    "raw/fullhd/challenge.zip",
]:
    raise SystemExit("challenge SynLoc fetch must use the challenge split archive")
if 'load_synloc_data(version, [f"raw/{version}/*.zip", f"raw/{version}/manifest.json"])' in train_text:
    raise SystemExit("run modes must use split-specific SynLoc snapshot patterns instead of raw/*.zip")

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

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "resized").mkdir()
    (root / "fullhd" / "nested").mkdir(parents=True)
    wrong = root / "resized" / "frame001.jpg"
    right = root / "fullhd" / "nested" / "frame001.jpg"
    only_resized = root / "resized" / "frame002.jpg"
    Image.new("RGB", (1920, 1080)).save(wrong)
    Image.new("RGB", (3840, 2160)).save(right)
    Image.new("RGB", (1920, 1080)).save(only_resized)
    image = {"file_name": "frames/frame001.jpg", "width": 3840, "height": 2160}
    resolved = image_path_for_record(root, image)
    if resolved != right:
        raise SystemExit(f"dimension-aware image lookup picked {resolved}, expected {right}")
    try:
        image_path(root, "frames/frame001.jpg", expected_width=4096, expected_height=2160)
    except RuntimeError as exc:
        if "matched annotation size 4096x2160" not in str(exc):
            raise
    else:
        raise SystemExit("dimension-aware image lookup should fail when no candidate matches annotation size")
    path, scale_x, scale_y, annotation_size, actual_size = image_path_and_scale_for_record(
        root,
        {"file_name": "frames/frame002.jpg", "width": 3840, "height": 2160},
        coordinate_scale_mode="actual_image",
    )
    if path != only_resized or (scale_x, scale_y) != (0.5, 0.5):
        raise SystemExit(f"actual-image coordinate scale picked {(path, scale_x, scale_y)}, expected resized half-scale")
    if annotation_size != (3840, 2160) or actual_size != (1920, 1080):
        raise SystemExit(f"actual-image coordinate scale reported wrong sizes: {annotation_size}, {actual_size}")
    if scale_xywh((100.0, 200.0, 50.0, 80.0), scale_x, scale_y) != (50.0, 100.0, 25.0, 40.0):
        raise SystemExit("scale_xywh did not map annotation boxes into actual-image coordinates")
    if scale_xy((100.0, 200.0), scale_x, scale_y) != (50.0, 100.0):
        raise SystemExit("scale_xy did not map annotation points into actual-image coordinates")
    if box_xyxy_to_annotation_scale((50.0, 100.0, 75.0, 140.0), scale_x, scale_y) != (100.0, 200.0, 150.0, 280.0):
        raise SystemExit("box_xyxy_to_annotation_scale did not map detector boxes back to annotation coordinates")

if 'os.getenv("RFDETR_MODEL_CLASS", "RFDETRLarge")' not in train_text:
    raise SystemExit("RF-DETR SoccerNet lane must default to RFDETRLarge; the checkpoint is not base-width")
if "model_class(pretrain_weights=str(checkpoint_path))" not in train_text:
    raise SystemExit("RF-DETR SoccerNet lane should load checkpoints through the RF-DETR public constructor")
if "model.model.model.load_state_dict(state)" in train_text:
    raise SystemExit("RF-DETR SoccerNet lane should not manually strict-load unknown architecture state")
if "api.upload_folder(**kwargs, create_pr=True)" not in train_text:
    raise SystemExit("train.py must retry result upload as a Hub PR when direct commit is blocked")
if 'os.getenv("HF_STRICT_UPLOAD", "0")' not in train_text:
    raise SystemExit("train.py must let scored experiments survive artifact upload failures by default")
if 'def emit_autonomy_result(summary: dict[str, Any]) -> None:' not in train_text:
    raise SystemExit("train.py must emit AUTONOMY_RESULT through a shared helper")
if "emit_autonomy_result(summary)\n    upload_result(summary[\"run_id\"], upload_root)" not in train_text:
    raise SystemExit("train.py must print AUTONOMY_RESULT before artifact upload")
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
if ! rg -q 'Hugging Face Jobs connector/app' scripts/codex_research_tick.sh; then
  echo "Codex tick prompt must route around local HF CLI job.write failures through the HF Jobs connector" >&2
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
if ! rg -q 'HF_ENV_FILE' scripts/run_hf_train.sh; then
  echo "HF train helper must load repo-local .env before checking HF_TOKEN" >&2
  exit 1
fi
if ! rg -q 'whoami-v2' scripts/run_hf_train.sh || ! rg -q 'job.write' scripts/run_hf_train.sh; then
  echo "HF train helper must inspect fine-grained HF_TOKEN job.write permission before submission" >&2
  exit 1
fi
if ! rg -q 'HF_WHOAMI_JSON_FILE' scripts/run_hf_train.sh; then
  echo "HF train helper must expose a test fixture path for token-scope preflight coverage" >&2
  exit 1
fi
if ! rg -q 'set -a' scripts/run_hf_train.sh; then
  echo "HF train helper must export .env variables for hf jobs secret lookup" >&2
  exit 1
fi
if ! rg -q -- '--preflight' scripts/run_hf_train.sh; then
  echo "HF train helper must provide a non-submitting credential preflight" >&2
  exit 1
fi
if ! rg -q 'PREFLIGHT=1' scripts/run_hf_train.sh; then
  echo "HF train preflight must reuse the submission credential checks" >&2
  exit 1
fi

scripts/codex_research_tick.sh --allow-dirty --dry-run >/tmp/auto-research-2-codex-tick.txt
scripts/codex_research_loop.sh --allow-dirty --dry-run --iterations 2 >/tmp/auto-research-2-codex-loop.txt 2>&1
scripts/run_hf_train.sh --dry-run >/tmp/auto-research-2-hf-train.txt
cat >/tmp/auto-research-2-hf-env-test.env <<'EOF'
HF_TOKEN=dummy-secret-for-verify
HF_FLAVOR=cpu-basic
TRAIN_MODE=transformer_baseline
HF_DATASET_REPO=example/synloc-data
HF_MODEL_REPO=example/synloc-models
EOF
env -u HF_TOKEN -u HF_FLAVOR -u TRAIN_MODE -u HF_DATASET_REPO -u HF_MODEL_REPO \
  HF_ENV_FILE=/tmp/auto-research-2-hf-env-test.env \
  scripts/run_hf_train.sh --dry-run >/tmp/auto-research-2-hf-env-train.txt
cat >/tmp/auto-research-2-no-job-write.json <<'EOF'
{
  "name": "dmontgomery40",
  "auth": {
    "accessToken": {
      "role": "fineGrained",
      "fineGrained": {
        "global": ["discussion.write"],
        "scoped": [
          {"entity": {"name": "dmontgomery40"}, "permissions": ["repo.content.read", "repo.write"]}
        ]
      }
    }
  }
}
EOF
if env -u HF_TOKEN -u HF_FLAVOR -u TRAIN_MODE -u HF_DATASET_REPO -u HF_MODEL_REPO \
  HF_ENV_FILE=/tmp/auto-research-2-hf-env-test.env \
  HF_WHOAMI_JSON_FILE=/tmp/auto-research-2-no-job-write.json \
  scripts/run_hf_train.sh --preflight >/tmp/auto-research-2-no-job-write.out 2>/tmp/auto-research-2-no-job-write.err; then
  echo "HF train preflight must fail fine-grained tokens that lack job.write" >&2
  exit 1
fi
if ! rg -q 'lacks job.write' /tmp/auto-research-2-no-job-write.err; then
  echo "HF train preflight must explain missing job.write clearly" >&2
  exit 1
fi

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
program = Path("program.md").read_text(encoding="utf-8")
if "missing `job.write`" not in program or "Hugging Face Jobs connector/app" not in program:
    raise SystemExit("program.md must preserve the HF Jobs connector fallback for local CLI job.write failures")
hf_env = Path("/tmp/auto-research-2-hf-env-train.txt").read_text(encoding="utf-8")
for needle in ["--flavor cpu-basic", "--env TRAIN_MODE=transformer_baseline", "--env HF_DATASET_REPO=example/synloc-data", "--env HF_MODEL_REPO=example/synloc-models"]:
    if needle not in hf_env: raise SystemExit(f"HF train .env dry-run did not load {needle}")
if "dummy-secret-for-verify" in hf_env:
    raise SystemExit("HF train dry-run leaked HF_TOKEN from .env")
job_write_err = Path("/tmp/auto-research-2-no-job-write.err").read_text(encoding="utf-8")
if "Hugging Face Jobs connector/app" not in job_write_err:
    raise SystemExit("HF train job.write failure must point to the connector/app fallback")
PYCHECK

echo "verify ok"
