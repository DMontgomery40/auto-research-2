# AGENTS.md

This repo is an autonomous research harness. Keep it closer to Karpathy's
`autoresearch` than to a platform.

## Prime Directive

Beat the best tracked score for the 2026 Spiideo SoccerNet SynLoc challenge by
June 30, 2026.

SynLoc is single-frame athlete detection plus world-coordinate pitch
localization. Optimize official SSKit `mAP-LocSim`; higher is better. Do not
submit to Codabench unless the owner explicitly asks.

## Simplicity Rule

Prefer markdown instructions plus tiny scripts over services, databases,
dashboards, schedulers, or custom frameworks. If a few clear sentences in
`program.md` solve the problem, do that before writing code.

## Repository Shape

- `program.md` is the main instruction file and loop contract.
- `train.py` is the one editable experiment/training payload.
- Fixed helpers exist only for setup, Hugging Face execution, council queueing,
  and official metric mechanics.
- `CURRENT.md` is the live state.
- `LEDGER.md` is the experiment and score log.
- `IDEAS.md` is the experiment backlog.
- `COUNCIL.md` documents the sibling council queue.
- `refs/`, `data/`, `runs/`, `outputs/`, `models/`, and `worktrees/` are
  local-only and ignored.

## Runtime Model

Local is orchestration and reasoning only. This machine has no parity with CUDA
training and inference.

Research happens in local Codex with `gpt-5.5` and `low` reasoning. Local
Codex reads `program.md`, chooses one experiment, edits the smallest needed
surface, records the result, and repeats.

For Codex, `scripts/codex_research_loop.sh` is the actual loop. It starts one
bounded `scripts/codex_research_tick.sh` process every 300 seconds until
interrupted. A single tick proves only one iteration; it does not prove the loop.

Hugging Face Jobs are CUDA execution substrate only. They may download data,
train, infer, evaluate, and upload artifacts, but they must not choose the next
experiment.

Do not validate model quality locally. Meaningful baseline, training,
inference, threshold selection, prediction generation, and keep/discard
decisions must use cloud CUDA plus official SSKit `mAP-LocSim`.

## Loop Contract

Use this shape:

1. setup-once
2. baseline-once
3. loop-forever

Inside the loop:

1. Read `program.md`, `CURRENT.md`, `LEDGER.md`, and `IDEAS.md`.
2. Pick one experiment with a clear expected score movement.
3. Run it in an isolated branch/worktree under `worktrees/<tag>/` when needed.
4. Edit `train.py` unless a fixed helper is actually broken.
5. Evaluate with the official SSKit metric path.
6. Record score, commit/job, keep/discard, and notes in `LEDGER.md`.
7. Update `CURRENT.md`.

## Current Research Bias

The active direction is track/pose/keypoint or direct ground-point prediction.
The SSKit oracle proved that exact GT scores `1.0` and SSKit-projected GT
keypoints score `0.9809895759`. The failed `first-yolo-train` run is a discard:
`mAP-LocSim=3.572767401302389e-06`, `recall_50=0.0`, worse than pose smoke
`0.000825082508250825`.

Do not run another generic detector fine-tune trying to rescue that path.

Zero or near-zero official scores from soccer/football-pretrained models on
SoccerNet data are plumbing warnings first, not model verdicts. Audit runtime,
class ids, preprocessing, prediction format, projection, and evaluator ingestion
before calling the model weak.

## SoccerMaster Rule

SoccerMaster is a serious soccer-specific lead, but copied-adapter scores are
not valid SoccerMaster verdicts unless the official runtime/config/postprocess
parity has been proven. Zero athlete output or worse-than-baseline copied
adapter scores are runtime/config/decode warnings first.

Official role mapping: `ball=0`, `goalkeeper=1`, `other=2`, `player=3`,
`referee=4`, `None=5`.

## Compute Rule

Use the cheapest option that actually works, always. Prefer CPU for setup
checks and `t4-small` for tiny CUDA probes. Escalate only when the cheaper
option fails, times out, OOMs, or is plainly too slow for the bounded
experiment. Do not add money ledgers or approval gates; the owner handles that
outside this repo.

## Council

The sibling council lives at `../challenge-council/`. Use it sparingly when
stuck, after a meaningful baseline, or every 2-3 days during serious autonomous
runs.

Queue requests with:

```bash
python3 scripts/ask_council.py --title "SynLoc strategy"
```

Do not ask it to mine leaked submissions or post-deadline winner writeups.

## Verification

Before ending a mutating turn, run:

```bash
scripts/verify.sh
```

Also run any narrower smoke command for files you changed.
