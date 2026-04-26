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
- `scripts/verify.sh` - narrow repo sanity check.

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

Dataset storage rule: prefer official-source download inside the cloud job. Mirror the dataset into a private Hugging Face dataset only if the SoccerNet/Spiideo terms allow it. Checkpoints, logs, metrics, and predictions can live in private HF repos or job artifacts when allowed.

## First Run

1. Read `program.md`.
2. Verify the control plane without printing secrets:

   ```bash
   scripts/verify.sh
   ```

3. Run dataset download and baseline on a cloud CUDA GPU job.
4. Persist datasets, predictions, checkpoints, and metrics to Hugging Face storage or job artifacts, not this repo.
5. Only then start creating isolated experiment worktrees under `worktrees/`.

## Ground Rules

- Keep the control plane markdown-first.
- Keep experiments isolated.
- Commit only source, scripts, and concise docs.
- Never commit credentials, datasets, checkpoints, generated predictions, or local upstream clones.
- Spend at most `$25/week` on compute unless a GitHub issue asks the owner for more and the owner approves.
