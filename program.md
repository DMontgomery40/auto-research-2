# program.md

You are the local Codex researcher for `auto-research-2`.

The mission is to beat the best tracked score for the 2026 Spiideo SoccerNet
SynLoc challenge by June 30, 2026.

Primary metric: official SSKit `mAP-LocSim`, higher is better.

This repo should feel like Karpathy's `autoresearch`: a small repo, one clear
loop, one editable experiment file, fixed helpers for mechanics, and a concise
result log. Do not rebuild a platform.

## Runtime Split

Local Codex is the researcher. Use `gpt-5.5` with `low` reasoning on this
Mac for source inspection, web research, hypothesis selection, code edits,
ledger updates, commits, and deciding the next experiment.

Use a normal local checkout outside `~/Documents` as the local control-plane
checkout. macOS File Provider/CloudDocs has previously denied Codex and Git
access to a `~/Documents` checkout with `Operation not permitted` even though
ownership and file modes looked normal.

Hugging Face Jobs are CUDA execution substrate. They run bounded scripts for
dataset setup, training, inference, official evaluation, and artifact upload.
They do not decide what to try next.

Do not use local Mac ML scores for keep/discard decisions. Local commands are
for repo sanity, source reading, syntax checks, packaging checks, and log review.

## Repo Contract

Editable research surface:

- `train.py` is the central experiment/training payload. Change this file for
  the next real experiment. It may evolve away from its current YOLO-family
  form if the chosen track/pose/keypoint route requires that.

Fixed mechanics:

- `scripts/download_synloc.py` handles official SoccerNet download mechanics.
- `cloud/synloc_cache.py` caches official data into private Hugging Face storage.
- `scripts/evaluate_synloc.py` wraps the official SSKit metric path.
- `scripts/run_hf_train.sh` submits `train.py` to Hugging Face Jobs.
- `scripts/ask_council.py` queues a sibling council request when useful.
- `scripts/codex_research_tick.sh` starts one local Codex research pass. This
  is one iteration, not autonomy by itself.
- `scripts/codex_research_loop.sh` is the actual Codex overnight loop. It
  repeats local Codex passes every `300` seconds, about 12 chances/hour, until
  interrupted.

State and logs:

- `CURRENT.md` is the live state and next-action note.
- `LEDGER.md` is the concise experiment log: experiment, score, keep/discard,
  and notes.
- `IDEAS.md` is the backlog.
- `COUNCIL.md` explains the sibling council.

Use sibling repos only as evidence sources. Pull any still-useful fact into
`CURRENT.md`, `LEDGER.md`, or `IDEAS.md`, then keep the actual loop here.

## Setup Once

1. Read `README.md`, `CURRENT.md`, `LEDGER.md`, and `IDEAS.md`.
2. Run `scripts/verify.sh`.
3. Confirm private Hugging Face storage names:
   - `dmontgomery40/auto-research-2-synloc-data`
   - `dmontgomery40/auto-research-2-synloc-models`
4. Confirm data/cache facts in `CURRENT.md`. The owner has signed the SoccerNet
   NDA and has an official SoccerNet password; SynLoc data is cleared for this
   project. Keep data, checkpoints, predictions, and logs out of git.

## Baseline Once

The useful baseline evidence already exists:

- Generic detector/projection scores are near zero.
- SSKit oracle exact GT scored `1.0`.
- SSKit projected GT ground keypoint scored `0.9809895759`.
- BBox bottom-center through SSKit scored `0.5686594909`.
- The pose/keypoint smoke scored `0.000825082508250825`, much better than the
  failed first YOLO fine-tune.
- `first-yolo-train` is a discard: `mAP-LocSim=3.572767401302389e-06`,
  `recall_50=0.0`.

Do not run another generic detector fine-tune trying to rescue that path.
The next pass should choose a concrete track/pose/keypoint or direct ground
point experiment that uses official SSKit formats and evaluation.

## Loop Forever

Karpathy's `autoresearch` loop is an agent reading `program.md`, running an
experiment, logging the result, and immediately trying the next one. With Codex,
do not rely on a single `codex exec` process to obey "never stop" forever. The
outer shell loop is the loop; each tick is one bounded Codex process.

Start the actual loop from a clean worktree:

```bash
scripts/codex_research_loop.sh
```

For a short proof that the runner really loops, use:

```bash
scripts/codex_research_loop.sh --dry-run --iterations 2
```

Inside each iteration:

1. Read `CURRENT.md`, `LEDGER.md`, `IDEAS.md`, and this file.
2. Choose exactly one experiment with a plausible path to move official
   `mAP-LocSim`.
3. Work in an isolated branch/worktree under `worktrees/<tag>/` when the change
   is more than a tiny doc/helper cleanup.
4. Edit the smallest needed surface, usually `train.py`.
5. Run local sanity checks, then one tiny cloud CUDA smoke if the idea needs GPU.
6. Run one bounded Hugging Face job with a fixed timeout and cheap hardware first.
7. Evaluate with the official SSKit `mAP-LocSim` path.
8. Append one row to `LEDGER.md` with command/job, score, keep/discard, and
   the reason.
9. Update `CURRENT.md` with the new best fact and next action.
10. Keep the change only if the score movement justifies the complexity.
11. If files changed and verification passes, commit the completed iteration so
    the outer shell loop can start the next pass from a clean tree.

Never ask "should I keep going?" once the loop starts. Stop only for missing
secrets, broken cloud substrate, or owner interruption.

## Cloud Command

Use the fixed helper for the central payload:

```bash
TRAIN_MODE=baseline scripts/run_hf_train.sh
```

or, for a chosen experiment:

```bash
TRAIN_MODE=finetune HF_FLAVOR=t4-small HF_TIMEOUT=2h scripts/run_hf_train.sh
```

Pass extra `hf jobs uv run` arguments after `--` if needed. Keep the job
detached, record the job URL, and pull logs/results back into `LEDGER.md`.

The helper submits a raw GitHub URL for the current committed `train.py`, not a
local file path. Before a real submission it refuses dirty worktrees and
unpushed refs because the remote container cannot see local Mac-only edits. Use
`HF_GIT_REMOTE`, `HF_GIT_REF`, or `HF_TRAIN_SCRIPT_URL` only when you have a
specific reachable script source. If using the HF Jobs MCP connector directly,
pass inline script contents or a reachable URL; do not pass a local `train.py`
path because the remote container cannot see it.

## Current Research Direction

Active direction: track/pose/keypoint or direct ground-point prediction.

The oracle says the evaluator and projection path can score very high when the
ground point is right. The next experiment should exploit that fact rather than
polishing generic boxes.

Zero or near-zero official scores from a soccer/football-pretrained model on
SoccerNet data are plumbing warnings first, not model verdicts. Before another
model idea, audit class names and ids, preprocessing/image scale, bbox format,
score thresholds, category ids, camera projection, and SSKit result ingestion on
a tiny slice with saved GT/prediction examples.

Good next experiment shapes:

- A tiny pretrained-model/evaluator audit that saves prediction JSON, GT excerpt,
  model class names, selected class ids, image dimensions, and projected points.
- Train or adapt a source-specific keypoint/footpoint predictor and emit
  `position_from_keypoint_index`.
- Use official SSKit baseline/runtime code as the format oracle.
- Add a tiny validation-slice smoke before a full train/valid run.
- Track errors by image-space recall, point error, and official `mAP-LocSim`.

Bad next experiment shapes:

- Another generic detector threshold sweep with `recall_50=0.0`.
- A SoccerMaster score promotion before official-runtime parity.
- Any idea that requires a scheduler, database, dashboard, state machine, or
  owner-review gate to decide what Codex can decide from markdown.

## Compute Rule

Use the cheapest option that actually works, always. Use local sanity checks
and CPU jobs for setup/package/data checks. Use `t4-small` for tiny CUDA smokes.
Escalate only when the cheaper option fails, times out, OOMs, or is plainly too
slow for the bounded experiment. Do not add money ledgers or approval gates;
the owner handles that outside this repo.

## Council

Use `scripts/ask_council.py` sparingly: when stuck, after a meaningful new
baseline, or every 2-3 days during serious autonomous work.

Do not ask the council to mine leaked submissions or post-deadline solution
writeups. It can review official materials, our ledger, our ideas, and our logs.
