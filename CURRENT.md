# Current State

Updated: 2026-04-26

## Mission

Beat the best tracked score for the 2026 Spiideo SoccerNet SynLoc challenge by June 30, 2026.

## Known Facts

- Official task: single-frame world-coordinate athlete detection and localization.
- Official metric: `mAP-LocSim`, higher is better.
- Official devkit: `https://github.com/Spiideo/sskit`.
- Local SSKit reference clone: `refs/sskit` at `9e28ad1`.
- Local Karpathy autoresearch reference clone: `refs/karpathy-autoresearch` at `228791f`.
- Local credentials file exists: `.env`.
- `.env` has `HF_TOKEN` and `SOCCERNET_PASSWORD`.
- GitHub CLI is already authenticated as `DMontgomery40`.
- Data download helper exists: `scripts/download_synloc.py`.
- Official metric wrapper exists: `scripts/evaluate_synloc.py`.
- Repo volume has about 13 GiB free as of 2026-04-26; this is likely not enough for the full SynLoc dataset.

## Unknowns

- Current public/test/challenge leaderboard top score.
- Dataset presence/checksum.
- Baseline local validation score.
- Codabench credentials/session status.
- Where to store the dataset if local disk stays tight.

## Next Action

1. Free disk space or choose an external dataset root for `data/SoccerNet/SpiideoSynLoc`.
2. Establish the official baseline score locally.
3. Start the first isolated experiment branch only after baseline is recorded.
