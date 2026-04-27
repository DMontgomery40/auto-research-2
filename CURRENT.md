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
- Baseline probe job `69efa455d2c8bd8662bd115b` completed on CUDA for 64 validation images with TorchVision Faster R-CNN: `mAP-LocSim=0.00012376237623762376`, 2,831 detections, run artifact `baseline-torchvision-2026-04-27T18-02-12.913357Z`.
- SoccerMaster is available as a soccer-specific lead. Its paper reports strong athlete detection and role benchmarks, including `92.3` AP@50, `50.5` mAP, and `99.2` role accuracy on the paper's pretraining-task evaluation. The sibling zero-score run should be treated as a runtime/config/decode failure until proven otherwise.
- Sibling repo `/Users/davidmontgomery/v2d-research` tested a copied SoccerMaster GSR adapter on a bounded 64-frame SynLoc slice (`synloc-20260426-1308`): all rows scored `mAP-LocSim=0.0`, with no decoded `player` detections. This is not a verdict against SoccerMaster; it means this repo should first audit weight placement, role mapping, class dimensions, normalization, and raw logits before using projected scores.
- Full baseline job `69efa541d70108f37ace099f` completed on HF Jobs `l4x1` for all 6,777 `fullhd valid` images with TorchVision Faster R-CNN: `mAP-LocSim=0.00003561507229859677`, 288,766 detections, selected score threshold `0.49995696544647217`, run artifact `baseline-torchvision-2026-04-27T18-17-53.729401Z`.
- The full baseline is a scoring-pipeline sanity check, not a serious model direction: it uses COCO `person` boxes, projects each bbox bottom-center point into pitch coordinates, and scores almost zero.
- Council requests now include `COUNCIL_DOSSIER.md`, autonomy state/events, budget, and baseline source so the council can give high-context criticism before the next expensive run.

## Unknowns

- Current public/test/challenge leaderboard top score.
- Codabench credentials/session status.
- Exact Hugging Face storage layout for predictions, checkpoints, metrics, and logs.

## Next Action

1. Run `soccermaster_wiring_probe_pending` on cloud CUDA.
2. If raw SoccerMaster outputs include player/goalkeeper/referee detections, build a tiny SynLoc conversion/eval probe.
3. If raw outputs are still ball/staff-heavy or empty, debug config against the official SoccerMaster paper/repo before spending on scoring sweeps.
