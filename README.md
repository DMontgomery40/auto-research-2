# auto-research-2

Markdown-first autonomous research harness for the 2026 Spiideo SoccerNet
SynLoc challenge.

Goal: beat the best tracked SynLoc score by June 30, 2026.

Metric: official SSKit `mAP-LocSim`, higher is better.

## Shape

This repo is intentionally small:

- `program.md` - main operating loop for local Codex.
- `train.py` - one editable experiment/training payload.
- `scripts/download_synloc.py` - official data download mechanics.
- `cloud/synloc_cache.py` - fixed cloud data-cache helper.
- `scripts/evaluate_synloc.py` - official SSKit metric wrapper.
- `scripts/run_hf_train.sh` - fixed Hugging Face Jobs launcher for `train.py`.
- `CURRENT.md` - live state and next action.
- `LEDGER.md` - concise result log.
- `IDEAS.md` - experiment backlog.
- `COUNCIL.md` and `scripts/ask_council.py` - sibling council queue.

Ignored local/heavy directories include `refs/`, `data/`, `runs/`, `outputs/`,
`models/`, and `worktrees/`.

## Runtime

Local Codex is the researcher: `gpt-5.5`, `low`, this repo, markdown state,
small code edits, commits, and next-experiment choice.

Hugging Face Jobs are CUDA execution substrate only: data setup, training,
inference, official evaluation, and artifact upload.

Run one local research pass:

```bash
scripts/codex_research_tick.sh
```

Run repeated local research passes every 300 seconds:

```bash
scripts/codex_research_loop.sh
```

Submit the central training/eval payload:

```bash
TRAIN_MODE=baseline scripts/run_hf_train.sh
```

## Current Facts

- Active direction: track/pose/keypoint or direct ground-point prediction.
- Best structural signal: SSKit projected GT keypoint scored `0.9809895759`.
- Generic detector and YOLO fine-tune paths are not the frontier.
- `first-yolo-train` is discarded: `mAP-LocSim=3.572767401302389e-06`,
  `recall_50=0.0`, worse than pose smoke `0.000825082508250825`.
- Compute rule: use the cheapest option that actually works, always.

Read `program.md` first for the loop contract.
