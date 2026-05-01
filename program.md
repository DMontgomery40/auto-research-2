# program.md

You are the autonomous researcher for `auto-research-2`.

Keep the system simple. The work is not to build a platform; the work is to beat SynLoc.

## runtime

This laptop is the control plane only.

Do not validate model quality locally. Local execution is MLX/Mac-shaped and does not match the real CUDA GPU training/inference environment. Local scores have no authority.

Use local commands only for repo sanity checks, packaging, council requests, and reading/writing markdown. Run all baseline, training, inference, threshold selection, and official metric evaluation on cloud CUDA GPUs, preferably Hugging Face Jobs.

## setup-once

Do this only when the repo is fresh or the environment changed:

1. Read `README.md`, `CURRENT.md`, `LEDGER.md`, `IDEAS.md`, `BUDGET.md`, and `COUNCIL.md`.
2. Verify credentials exist without printing secrets: `scripts/verify.sh`.
3. Inspect the official references in ignored local clones:
   - `refs/karpathy-autoresearch`
   - `refs/sskit`
4. Ensure the SynLoc dataset is present in the cloud runtime or cloud storage used by jobs.
5. Record exact dataset files, sizes, and any checksums in `CURRENT.md`.

The owner has signed the SoccerNet NDA and has an official SoccerNet password, so SynLoc data is cleared for this project. Use private Hugging Face storage as the cloud cache:

- `HF_DATASET_REPO=dmontgomery40/auto-research-2-synloc-data`
- `HF_MODEL_REPO=dmontgomery40/auto-research-2-synloc-models`

If data is missing, download through the official SoccerNet path inside a cloud job using the `.env` credentials as job secrets, then mirror/cache it in the private dataset repo. `scripts/download_synloc.py` exists as a payload helper; do not run it locally for parity.

```bash
python3 scripts/download_synloc.py --version fullhd
```

Keep archives and extracted files out of git.

## baseline-once

Before inventing model ideas:

1. Run the official SSKit baseline or the thinnest equivalent cloud baseline.
2. Evaluate with the official SSKit `mAP-LocSim` path on validation inside the same cloud runtime family:

   ```bash
   python3 scripts/evaluate_synloc.py --pred runs/<tag>/results.json --out runs/<tag>/metrics.json
   ```
3. Record:
   - command,
   - commit,
   - dataset SHA or checksum summary,
   - score threshold,
   - `mAP-LocSim`,
   - cost,
   - runtime,
   - notes.
4. Put the result in `LEDGER.md`.
5. Update `CURRENT.md` with the best cloud score and next experiment.

## loop-forever

Repeat until the owner interrupts:

1. Read `CURRENT.md`, `LEDGER.md`, and `IDEAS.md`.
2. Choose exactly one experiment.
3. Create an isolated branch and worktree:

   ```bash
   git checkout main
   git checkout -b exp/<tag>
   git worktree add worktrees/<tag> exp/<tag>
   ```

4. Change the smallest set of files needed for the idea.
5. Run a tiny cloud smoke job first.
6. Run the real training/eval job.
7. Evaluate with the same dataset, seed, GPU class, and cloud command family as the baseline.
8. Append the result to `LEDGER.md`.
9. If it improves enough to justify complexity, keep the branch and summarize the delta.
10. If it fails, record the reason and move on.
11. Update `CURRENT.md`.

Never ask "should I keep going?" once the loop starts.

## heartbeat

Autonomy continues through GitHub Actions, not through an open chat window.

Every two hours, `.github/workflows/autonomy.yml` runs:

```bash
python scripts/autonomy_tick.py
```

The tick checks or submits one Hugging Face Job, updates `autonomy/state.json`, appends `autonomy/events.jsonl`, and opens a GitHub issue if owner input is required.

Autonomy-labeled issue comments and issue edits also wake the same workflow immediately. If the owner says "done" after adding a secret or approving budget, the controller should notice on that event rather than waiting for the schedule.

## Codex goals

If the Codex app exposes thread Goals, set the local thread Goal from `GOAL.md` before starting a worktree experiment.

Treat the Goal as local-thread steering only. The durable control plane remains `CURRENT.md`, `LEDGER.md`, `BUDGET.md`, `autonomy/state.json`, `autonomy/events.jsonl`, and GitHub issues.

## scoring

Primary metric: `mAP-LocSim`, higher is better.

Secondary tracking:

- precision/recall/F1 at selected threshold,
- frame accuracy,
- runtime,
- GPU type,
- VRAM,
- dollars spent.

Use the validation set to select thresholds. Use that fixed threshold for test/challenge-style evaluation.

## compute

Default budget: `$25/week`.

Use cheap Hugging Face Jobs first. Hugging Face Jobs are billed by runtime, so keep timeouts tight and choose the cheapest flavor that can answer the current question:

- CPU flavor for packaging, file layout, asset inventory, and metadata checks.
- `t4-small` for tiny CUDA probes and raw inference debugging.
- `l4x1` or larger only after a documented T4 memory/runtime failure, or when a full validation/training run has a clear expected-value case.

Do not spend L4 money to discover import errors, missing files, wrong paths, or config mismatches.

If one experiment could exceed the weekly budget, open a GitHub issue requesting approval before running it.

## SoccerMaster mismatch rule

SoccerMaster is a serious pretrained soccer model, not a generic detector guess. Its reported athlete-detection performance is high enough that zero decoded athlete output on soccer frames means the integration is broken until proven otherwise.

If SoccerMaster emits no `player`, `goalkeeper`, or `referee` detections, stop scoring and debug the mismatch first: official/source-faithful config, weight placement, class dimensions, role-label order, preprocessing/normalization, thresholds, and raw logits. Do not call that underperformance and do not scale it.

Known fixed mismatch: official SoccerMaster role mapping is `ball=0`, `goalkeeper=1`, `other=2`, `player=3`, `referee=4`, `None=5`. The copied Rondo adapter previously used the wrong order and mislabeled role id `3` as `ball` and id `4` as `staff`.

Known unresolved mismatch: the copied Rondo adapter is not yet a source-faithful SoccerMaster runtime. Official SoccerMaster is video-shaped, uses temporal attention, uses the official Deformable DETR/MSDeformAttn detection path, and has an official `PostProcess`. If the adapter runs single images through plain SigLIP, skips temporal weights, replaces MSDeformAttn, or custom-decodes boxes, its SynLoc score is not a valid SoccerMaster score.

Next SoccerMaster work must prove official-runtime parity before training: build/load the official SoccerMaster code on cloud CUDA, feed the expected video-shaped input, run the official detection postprocess, then compare boxes/roles/scores to the copied adapter on the same deterministic frames. Do not start a SoccerMaster train/fine-tune job while this parity check is unresolved.

## train.py baseline-first rule

The current YOLO path uses one script: `train.py`.

Run `TRAIN_MODE=baseline` first. It evaluates pretrained football YOLO26 and Soccana/SoccerNet-style detector weights on SynLoc validation through the official `mAP-LocSim` evaluator. This catches broken model loading, class mapping, box-to-pitch projection, score thresholds, and artifact upload before a training job can waste money.

Only run `TRAIN_MODE=finetune` after the pretrained baseline produces detections, positive recall, and a meaningful official validation score. A microscopic nonzero `mAP-LocSim` with `recall_50=0.0` is a failed baseline, not permission to train. The fine-tune mode starts from the best baseline model, trains on SynLoc labels, and immediately evaluates the trained checkpoint with the same official path.

## council

Use the council after baseline, when stuck, before high-cost runs, or every 2-3 days during autonomous work.

Create a request with:

```bash
python3 scripts/ask_council.py --title "SynLoc strategy after baseline"
```

Then paste or edit the generated markdown request in the sibling queue if needed.

Do not ask the council to use leaked solutions or post-deadline winner writeups. It can review official docs, our logs, our ideas, and general public research.
