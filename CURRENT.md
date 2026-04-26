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

## Unknowns

- Current public/test/challenge leaderboard top score.
- Dataset presence/checksum.
- Baseline local validation score.
- Codabench credentials/session status.

## Next Action

1. Confirm or download `data/SoccerNet/SpiideoSynLoc`.
2. Establish the official baseline score locally.
3. Start the first isolated experiment branch only after baseline is recorded.
