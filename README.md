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
- `scripts/download_synloc.py` - official SoccerNet download wrapper.
- `scripts/evaluate_synloc.py` - official SSKit metric wrapper.
- `scripts/verify.sh` - narrow repo sanity check.

Local upstream clones live in ignored `refs/`:

- `refs/karpathy-autoresearch`
- `refs/sskit`

They are references, not vendored source.

## First Run

1. Read `program.md`.
2. Verify credentials exist without printing them:

   ```bash
   scripts/verify.sh
   ```

3. Download/unpack SynLoc under `data/SoccerNet/SpiideoSynLoc`.
4. Run the official SSKit baseline and record it in `LEDGER.md`.
5. Only then start creating isolated experiment worktrees under `worktrees/`.

## Ground Rules

- Keep the control plane markdown-first.
- Keep experiments isolated.
- Commit only source, scripts, and concise docs.
- Never commit credentials, datasets, checkpoints, generated predictions, or local upstream clones.
- Spend at most `$25/week` on compute unless a GitHub issue asks the owner for more and the owner approves.
