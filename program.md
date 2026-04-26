# program.md

You are the autonomous researcher for `auto-research-2`.

Keep the system simple. The work is not to build a platform; the work is to beat SynLoc.

## setup-once

Do this only when the repo is fresh or the environment changed:

1. Read `README.md`, `CURRENT.md`, `LEDGER.md`, `IDEAS.md`, `BUDGET.md`, and `COUNCIL.md`.
2. Verify credentials exist without printing secrets: `scripts/verify.sh`.
3. Inspect the official references in ignored local clones:
   - `refs/karpathy-autoresearch`
   - `refs/sskit`
4. Ensure the SynLoc dataset is present at `data/SoccerNet/SpiideoSynLoc`.
5. Record exact dataset files, sizes, and any checksums in `CURRENT.md`.

If data is missing, download through the official SoccerNet path using the local `.env` credentials. Keep archives and extracted files out of git.

## baseline-once

Before inventing model ideas:

1. Run the official SSKit baseline or the thinnest equivalent local baseline.
2. Evaluate with the official SSKit `mAP-LocSim` path on validation.
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
5. Update `CURRENT.md` with the best local score and next experiment.

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
5. Run a smoke check first.
6. Run the real training/eval job.
7. Evaluate with the same dataset, seed, and command family as the baseline.
8. Append the result to `LEDGER.md`.
9. If it improves enough to justify complexity, keep the branch and summarize the delta.
10. If it fails, record the reason and move on.
11. Update `CURRENT.md`.

Never ask "should I keep going?" once the loop starts.

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

Use cheap local CPU/GPU checks first. Use Hugging Face Jobs or other rented GPU only when the local smoke path is clean and the expected signal is worth the spend.

If one experiment could exceed the weekly budget, open a GitHub issue requesting approval before running it.

## council

Use the council after baseline, when stuck, before high-cost runs, or every 2-3 days during autonomous work.

Create a request with:

```bash
python3 scripts/ask_council.py --title "SynLoc strategy after baseline"
```

Then paste or edit the generated markdown request in the sibling queue if needed.

Do not ask the council to use leaked solutions or post-deadline winner writeups. It can review official docs, our logs, our ideas, and general public research.
