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
- Data download helper exists: `scripts/download_synloc.py`, but it is intended for cloud job payloads.
- Official metric wrapper exists: `scripts/evaluate_synloc.py`, but meaningful metric runs belong on cloud CUDA GPUs.
- Repo volume has about 137 GiB free as of 2026-04-26 after owner cleanup.
- Local model validation is explicitly out of scope because local MLX/Mac execution has no parity with cloud CUDA training and inference.
- Owner has signed the SoccerNet NDA and has an official SoccerNet password; SynLoc data is cleared for this project.
- Private HF dataset/cache repo exists: `dmontgomery40/auto-research-2-synloc-data`.
- Private HF model/checkpoint repo exists: `dmontgomery40/auto-research-2-synloc-models`.
- Persistent heartbeat exists: `.github/workflows/autonomy.yml`.
- Controller state exists: `autonomy/state.json`.

## Unknowns

- Current public/test/challenge leaderboard top score.
- Cloud dataset presence/checksum.
- Baseline cloud validation score.
- Codabench credentials/session status.
- Exact Hugging Face storage layout for predictions, checkpoints, metrics, and logs.

## Next Action

1. Push and enable the GitHub Actions heartbeat.
2. Let the controller submit/check the first tiny Hugging Face Jobs smoke run.
3. Establish the official baseline score on cloud CUDA.
4. Start the first isolated experiment branch only after baseline is recorded.
