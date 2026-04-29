# AGENTS.md

This repo is an autonomous research harness. Keep it closer to Karpathy's `autoresearch` than to a platform.

## Prime Directive

Beat the best tracked score for the 2026 Spiideo SoccerNet SynLoc challenge by June 30, 2026.

SynLoc is single-frame athlete detection plus world-coordinate pitch localization. The metric to optimize is `mAP-LocSim`; higher is better. Track public/test/challenge scores in `LEDGER.md`, but do not submit to Codabench unless the owner explicitly asks.

## Simplicity Rule

Prefer markdown instructions plus tiny scripts over services, databases, dashboards, schedulers, or custom frameworks. If three clear sentences in `program.md` solve the problem, do that before writing code.

Code is justified when it:

- removes repeated manual friction,
- prevents expensive mistakes,
- makes a run reproducible, or
- converts official devkit behavior into a stable cloud job command.

## Repository Shape

- `program.md` is the loop contract and should stay readable enough to hand to any strong agent.
- `CURRENT.md` is the live state.
- `LEDGER.md` is the experiment and score log.
- `IDEAS.md` is the experiment backlog.
- `BUDGET.md` is the compute ledger.
- `COUNCIL.md` documents the sibling council queue.
- `scripts/` contains small helpers only.
- `refs/`, `data/`, `runs/`, `outputs/`, `models/`, and `worktrees/` are local-only and ignored.

## Runtime Model

Local is orchestration only. This machine is MLX/Mac-shaped and has no parity with the CUDA GPU training and inference environment.

Do not validate model quality locally. Do not make keep/discard decisions from local ML scores.

Local commands are allowed for repo sanity, syntax, packaging, council queueing, and log review. Meaningful baseline, training, inference, evaluation, threshold selection, and prediction generation must run on cloud CUDA GPUs, preferably Hugging Face Jobs while the budget allows.

The owner has signed the SoccerNet NDA and has an official SoccerNet password. Treat SynLoc data as cleared for this project and use private Hugging Face storage for cloud dataset/cache work:

- `dmontgomery40/auto-research-2-synloc-data`
- `dmontgomery40/auto-research-2-synloc-models`

## Loop Contract

Use this shape:

1. setup-once
2. baseline-once
3. loop-forever

Inside the loop:

1. Read `CURRENT.md`, `LEDGER.md`, and `IDEAS.md`.
2. Pick one experiment with a clear expected score movement.
3. Run it in an isolated branch/worktree under `worktrees/<tag>/`.
4. Evaluate with the official SSKit metric path.
5. Record score, cost, commit, and decision in `LEDGER.md`.
6. Keep if score improves enough for the complexity added; otherwise discard or revert.
7. Update `CURRENT.md`.

## Persistent Autonomy

This repo must not depend on an open chat thread to keep moving.

- `.github/workflows/autonomy.yml` is the heartbeat.
- Autonomy-labeled issue comments and issue edits wake the heartbeat immediately.
- `scripts/autonomy_tick.py` runs one bounded controller step.
- `autonomy/state.json` is the durable state.
- `autonomy/events.jsonl` is the concise event log.
- `cloud/` contains Hugging Face Jobs payloads.

If the controller needs owner input, more budget, or secret repair, it opens a GitHub issue with `autonomy` and `needs-owner` labels. Do not silently wait in chat.

## Experiment Discipline

- Same seed, same dataset SHA, same cloud eval command, same GPU class when comparing runs.
- Treat validation-selected score thresholds as validation-only unless the threshold is then fixed for test/challenge.
- Keep generated predictions, checkpoints, logs, and data out of git.
- Prefer the cheapest viable Hugging Face flavor for every probe. Use CPU flavors for packaging, asset, and metadata checks; use `t4-small` for tiny CUDA probes; only escalate to `l4x1` or larger after a documented T4 memory/runtime failure or a concrete throughput need.
- Prefer a tiny cloud smoke job before any expensive GPU-heavy idea.
- Spend compute like it is real money. It is.

## SoccerMaster Rule

SoccerMaster is a pretrained soccer-specific model with reported athlete-detection performance far above generic out-of-the-box detectors. If a SoccerMaster path emits zero `player`/`goalkeeper`/`referee` detections on ordinary soccer frames, treat that as a runtime/config/decode mismatch until proven otherwise. Do not describe it as model underperformance, do not bury it as a low score, and do not run scoring sweeps before auditing weight placement, class/role dimensions, role-label order, preprocessing, thresholds, and raw logits.

Official SoccerMaster role mapping is `ball=0`, `goalkeeper=1`, `other=2`, `player=3`, `referee=4`, `None=5`. Do not use the old copied Rondo label order (`player`, `goalkeeper`, `referee`, `ball`, `staff`, `other`) for SoccerMaster outputs.

## Budget

Default budget is `$25/week`. Track each paid job in `BUDGET.md`.

If more spend is needed, open a GitHub issue in this repo with:

- proposed experiment,
- expected upside,
- estimated cost,
- fallback if it fails.

Do not exceed the budget before owner approval.

## Council

The sibling council lives at `../challenge-council/`. Use it sparingly:

- when the loop is stuck,
- after a meaningful baseline is known,
- before a large spend,
- every 2-3 days during serious autonomous runs.

Queue requests by writing `council_request.md` into the council automation inbox. Prefer `scripts/ask_council.py`.

The council may analyze official task materials, devkit behavior, logs, and our experiment ledger. Do not ask it to mine post-deadline winner writeups or leaked solutions. Leaderboard score tracking is allowed; solution leakage is not.

## Verification

Before ending a mutating turn, run:

```bash
scripts/verify.sh
```

Also run any narrower test or smoke command for files you changed.
