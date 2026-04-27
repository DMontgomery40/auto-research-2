# Experiment Ledger

Primary metric: `mAP-LocSim` (higher is better).

| Date | Tag | Commit | Dataset | Cloud Runtime | Eval command | Score | Threshold | Cost | Runtime | Decision | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| 2026-04-26 | setup | pending | unknown | none | pending | 0.000 | 0.000 | $0.00 | 0m | keep | Repo initialized; baseline not run yet. |
| 2026-04-26 | cloud-smoke | 19f67af | none | HF Jobs `t4-small` | `cloud/synloc_smoke.py` | 0.000 | 0.000 | $0.75 est | n/a | keep | Two packaging/import failures, then job `69ee7a1ed2c8bd8662bd0302` verified GPU, SoccerNet, SSKit, and private HF write. |
| 2026-04-26 | dataset-cache-valid | 19f67af | fullhd valid | HF Jobs `cpu-upgrade` | `cloud/synloc_cache.py` | 0.000 | 0.000 | $2.00 est | n/a | keep | First run lacked username; job `69ee959bd70108f37ace0415` cached `annotations.zip` and `val.zip` to private HF dataset repo. |
| 2026-04-27 | baseline-probe | 19f67af | fullhd valid | HF Jobs `t4-small` | `cloud/synloc_baseline_yolo.py` | 0.000 | 0.000 | $1.50 est | 0m | retry | Job `69ee962cd2c8bd8662bd0432` failed before execution on bare PEP 723 `git+https://...` dependency; dependency patched and phase reset. |
| 2026-04-27 | baseline-probe | 9a57490 | fullhd valid | HF Jobs `t4-small` | `cloud/synloc_baseline_yolo.py` | 0.000 | 0.000 | $1.50 est | 51s | retry | Job `69efa0ccd70108f37ace0980` failed on `xtcocotools` NumPy ABI mismatch; pinned `numpy<2` and phase reset. |
| 2026-04-27 | baseline-probe | 1c98108 | fullhd valid | HF Jobs `t4-small` | `cloud/synloc_baseline_yolo.py` | 0.000 | 0.000 | $1.50 est | 39s | retry | Job `69efa24bd70108f37ace098f` failed on `libGL.so.1` via `ultralytics`/OpenCV; switched baseline detector to TorchVision Faster R-CNN. |
| 2026-04-27 | baseline-probe | e01f8d0 | fullhd valid | HF Jobs `t4-small` | `cloud/synloc_baseline_yolo.py` | 0.000 | 0.000 | $1.50 est | 41s | retry | Job `69efa359d2c8bd8662bd113e` failed because GitHub SSKit imports undeclared `scipy`; added explicit `scipy`. |
| 2026-04-27 | baseline-probe | 3681f37 | fullhd valid 64 images | HF Jobs `t4-small` | `cloud/synloc_baseline_yolo.py` | 0.0001237624 | 0.4998428226 | $1.50 est | 79s | keep | Job `69efa455d2c8bd8662bd115b` completed on CUDA with 2,831 detections; artifact run `baseline-torchvision-2026-04-27T18-02-12.913357Z`. |

## Leaderboard Tracking

| Date Checked | Board | Rank 1 Score | Our Best | Source | Notes |
|---|---|---:|---:|---|---|
| 2026-04-26 | Codabench test/challenge | unknown | 0.000 | pending | Codabench page identified; score not yet scraped or authenticated. |
