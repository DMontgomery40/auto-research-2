# Current State

Updated: 2026-04-27

## Mission

Beat the best tracked score for the 2026 Spiideo SoccerNet SynLoc challenge by June 30, 2026.

## Known Facts

- Official task: single-frame world-coordinate athlete detection and localization.
- Official metric: `mAP-LocSim`, higher is better.
- Official devkit: `https://github.com/Spiideo/sskit`.
- Local SSKit reference clone: `refs/sskit` at `9e28ad1`.
- Local Karpathy autoresearch reference clone: `refs/karpathy-autoresearch` at `228791f`.
- Local credentials file exists: `.env`.
- `.env` has `HF_TOKEN`, `SOCCERNET_USERNAME`, `SOCCERNET_PASSWORD`, and the separate Spiideo sign-in password alias.
- GitHub CLI is already authenticated as `DMontgomery40`.
- Data download helper exists: `scripts/download_synloc.py`, but it is intended for cloud job payloads.
- Official metric wrapper exists: `scripts/evaluate_synloc.py`, but meaningful metric runs belong on cloud CUDA GPUs.
- Repo volume has about 137 GiB free as of 2026-04-26 after owner cleanup.
- Local model validation is explicitly out of scope because local MLX/Mac execution has no parity with cloud CUDA training and inference.
- Owner has signed the SoccerNet NDA and has an official SoccerNet password; SynLoc data is cleared for this project.
- Dataset download also requires the Spiideo Research username/email as `SOCCERNET_USERNAME`.
- The Spiideo sign-in password is separate from the SoccerNet data password. Use `SOCCERNET_SIGNIN_PASSWORD`; `SOCCERNET_PASSWORD_2` is accepted as an alias.
- Private HF dataset/cache repo exists: `dmontgomery40/auto-research-2-synloc-data`.
- Private HF model/checkpoint repo exists: `dmontgomery40/auto-research-2-synloc-models`.
- Persistent heartbeat exists: `.github/workflows/autonomy.yml`.
- Autonomy-labeled issue comments/edits now wake the heartbeat immediately.
- Controller state exists: `autonomy/state.json`.
- Cloud smoke succeeded on HF Jobs T4 with SoccerNet and SSKit imports.
- Cloud dataset cache is present for `fullhd` validation:
  - `annotations.zip` sha256 `848c15ee5ad00494e636d5ca776f57aa10fd2533ad5bc5088c9702782ebabb87`
  - `val.zip` sha256 `a0d19585df77e18253f9d35d4c0b1c07d36af8e53797bc04fc8aa901d7c3d68e`
- Baseline probe job `69ee962cd2c8bd8662bd0432` failed before execution because the PEP 723 dependency used a bare `git+https://...` URL. It is now patched to `sskit @ git+https://github.com/Spiideo/sskit.git`.
- Baseline probe job `69efa0ccd70108f37ace0980` got past PEP 723 parsing, then failed on `xtcocotools` versus NumPy 2 ABI compatibility. The baseline job now pins `numpy<2`.
- Baseline probe job `69efa24bd70108f37ace098f` got past the NumPy issue, then failed because `ultralytics` imported desktop OpenCV and required `libGL.so.1`. The baseline now uses TorchVision Faster R-CNN instead.
- Baseline probe job `69efa359d2c8bd8662bd113e` got past TorchVision imports, then failed because GitHub SSKit imports `scipy` without declaring it. The baseline job now includes `scipy`.

## Unknowns

- Current public/test/challenge leaderboard top score.
- Baseline cloud validation score.
- Codabench credentials/session status.
- Exact Hugging Face storage layout for predictions, checkpoints, metrics, and logs.

## Next Action

1. Retry `baseline_probe_pending` with TorchVision Faster R-CNN plus explicit `scipy`.
2. If the probe succeeds, run `baseline_full_pending` on cloud CUDA.
3. Record the baseline `mAP-LocSim`, cost, and artifacts in `LEDGER.md`.
4. Ask council for experiment ideas after baseline is real, then start the first isolated experiment branch.
