# auto-research-2

Autonomous research harness for the 2026 Spiideo SoccerNet SynLoc challenge.

The goal is simple: beat the best tracked SynLoc score by June 30, 2026, without turning the repo into a platform. This project borrows the spirit of Karpathy's `autoresearch`: one clear loop, markdown as the control plane, small scripts only where they remove friction.

## Target

SynLoc asks for single-frame world-coordinate athlete detection and localization. A submission is a zip containing:

- `results.json` with one detection record per predicted player.
- `metadata.json` with the validation-selected `score_threshold`.

The primary metric is `mAP-LocSim`. Higher is better.

## Files

- `program.md` - the operating instructions for the autonomous agent.
- `CURRENT.md` - live state and next action.
- `LEDGER.md` - experiment history and score tracking.
- `IDEAS.md` - backlog of candidate experiments.
- `BUDGET.md` - weekly compute budget ledger.
- `COUNCIL.md` - how to use the sibling challenge council.
- `scripts/ask_council.py` - drop a markdown request into the council automation queue.
- `scripts/download_synloc.py` - official SoccerNet download wrapper for cloud job payloads.
- `scripts/evaluate_synloc.py` - official SSKit metric wrapper for cloud job payloads.
- `scripts/autonomy_tick.py` - one bounded autonomous controller tick.
- `scripts/verify.sh` - narrow repo sanity check.
- `train.py` - YOLO-family baseline and fine-tune script; baseline mode must pass before training mode runs.
- `.github/workflows/autonomy.yml` - scheduled heartbeat every two hours.

Local upstream clones live in ignored `refs/`:

- `refs/karpathy-autoresearch`
- `refs/sskit`

They are references, not vendored source.

## Runtime Model

This laptop is the control plane. It is not a validation target.

Local ML runs are not comparable to the real environment because local execution is MLX/Mac-shaped while challenge training and inference must run on cloud CUDA GPUs. Do not use local model scores to make keep/discard decisions.

Use local commands only for:

- repo sanity checks,
- packaging checks,
- queueing council requests,
- preparing small job scripts,
- reading logs and ledgers.

Use Hugging Face Jobs or another cloud CUDA GPU runtime for:

- dataset download,
- baseline inference,
- training,
- validation evaluation,
- threshold selection,
- challenge-style prediction generation.

Dataset storage rule: the owner has signed the SoccerNet NDA and has an official SoccerNet password, so SynLoc data is cleared for this project. Use private Hugging Face storage as the working cloud cache:

- Dataset/cache repo: `dmontgomery40/auto-research-2-synloc-data`
- Model/checkpoint repo: `dmontgomery40/auto-research-2-synloc-models`

Keep these repos private. Checkpoints, logs, metrics, and predictions can live there or in job artifacts.

## First Run

1. Read `AGENTS.md`, `program.md`, `CURRENT.md`, `LEDGER.md`, `IDEAS.md`, and `BUDGET.md`.
2. Verify the control plane without printing secrets:

   ```bash
   scripts/verify.sh
   ```

3. Run dataset download and baseline on a cloud CUDA GPU job.
4. Persist datasets, predictions, checkpoints, and metrics to Hugging Face storage or job artifacts, not this repo.
5. Only then start creating isolated experiment worktrees under `worktrees/`.

## Autonomy

The repo has a persistent heartbeat:

- GitHub Actions runs `scripts/autonomy_tick.py` every two hours and on manual dispatch.
- Comments or changes on autonomy-labeled GitHub issues wake the same controller immediately.
- The tick submits or checks exactly one Hugging Face Job.
- State lives in `autonomy/state.json`.
- Events live in `autonomy/events.jsonl`.
- If a secret, budget, or cloud job blocks progress, the controller opens a GitHub issue instead of going silent.
- If the owner fixes the blocker and replies on that issue, the issue event wakes the heartbeat without waiting for the next two-hour schedule.

Initial phases:

1. `cloud_smoke_pending` - verify HF Jobs, GPU visibility, imports, and private repo write access.
2. `dataset_cache_valid_pending` - cache the fullhd validation split in the private HF dataset repo.
3. `baseline_probe_pending` - run a small YOLO baseline probe on cloud CUDA.
4. `baseline_full_pending` - run the validation baseline on cloud CUDA.
5. `soccermaster_wiring_probe_pending` - current SoccerMaster state after the role-label bug was found: rerun the tiny T4 probe with official SoccerMaster role labels.
6. `soccermaster_synloc_conversion_probe_pending` - convert corrected SoccerMaster player/goalkeeper/referee outputs into SynLoc `results.json`, run official `mAP-LocSim` on a 64-image validation slice, and only then decide the first `train.py`/fine-tune experiment.
7. `pretrained_yolo_baseline_pending` - run `train.py` with `TRAIN_MODE=baseline` to evaluate pretrained football YOLO26 and Soccana weights before training.
8. `train_dataset_cache_pending` - cache `train,valid` after the pretrained baseline passes.
9. `first_train_experiment_pending` - run `train.py` with `TRAIN_MODE=finetune` for the first small SynLoc fine-tune.

## Ground Rules

- Keep the control plane markdown-first.
- Keep experiments isolated.
- Commit only source, scripts, and concise docs.
- Never commit credentials, datasets, checkpoints, generated predictions, or local upstream clones.
- Spend at most `$25/week` on compute unless a GitHub issue asks the owner for more and the owner approves.
