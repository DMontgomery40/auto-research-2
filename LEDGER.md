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
| 2026-04-27 | baseline-full | 7aa2a67 | fullhd valid 6,777 images | HF Jobs `l4x1` | `cloud/synloc_baseline_yolo.py` | 0.0000356151 | 0.4999569654 | $6.00 est | 14m | keep | Job `69efa541d70108f37ace099f` completed on CUDA with 288,766 detections; this proves the official metric path but is only a generic COCO Faster R-CNN bottom-center projection baseline. |
| 2026-04-27 | soccermaster-wiring-probe | 56f6632 | 4 SynLoc images | HF Jobs `l4x1` | `cloud/soccermaster_wiring_probe.py` | 0.000 | 0.000 | $2.00 est | n/a | retry | Job `69efb685d70108f37ace0a18` failed before inference because the copied adapter looked for `rondo_payload/models/soccermaster/backbone.pt`; the HF asset repo actually stores weights under `models/soccermaster/`. |
| 2026-04-29 | soccermaster-wiring-probe | 9c0020e | 4 SynLoc images | HF Jobs `l4x1` | `cloud/soccermaster_wiring_probe.py` | 0.000 | 0.000 | $2.00 est | 3m | relabel | Job `69f229c4d70108f37ace174a` loaded weights and ran CUDA inference. It reported `ball=1120`, `staff=80`, `athlete_like_at_conf_0_05=0`, but diagnosis found the copied adapter role-label order was wrong. Official SoccerMaster maps role id `3` to `player` and id `4` to `referee`, so this likely means `player=1120`, `referee=80`. Rerun on `t4-small` with official role labels before conversion/eval. |
| 2026-04-29 | soccermaster-wiring-probe-corrected | pending | 4 SynLoc images | HF Jobs `t4-small` | `cloud/soccermaster_wiring_probe.py` | 0.000 | 0.000 | $0.50 est | n/a | keep | Job `69f23419d2c8bd8662bd31f2` confirmed the label bug: official labels yield `raw_role_total={player:1196, referee:4}` and `role_total_at_conf_0_05={player:731, referee:4}`. SoccerMaster is emitting athletes; the next question is projection/eval quality. |
| 2026-04-29 | soccermaster-synloc-eval-probe | pending | fullhd valid 64 images | HF Jobs `t4-small` | `cloud/soccermaster_synloc_eval_probe.py` | pending | pending | $1.00 est | running | running | Job `69f23612d2c8bd8662bd3210` converts corrected SoccerMaster detections into SynLoc `results.json`, sweeps thresholds and `athlete` versus `person_plus_other` role sets, evaluates official `mAP-LocSim`, and uploads artifacts. |

## External Prior

- 2026-04-26 sibling repo `/Users/davidmontgomery/v2d-research` ran copied SoccerMaster GSR adapter probe `synloc-20260426-1308` on 64 deterministic SynLoc validation frames. Score was `mAP-LocSim=0.0` across 54 confidence/role/pitch-bound rows; role decode produced no `player` detections. This is not counted as an `auto-research-2` score, and it should be read as an integration/config failure signal, not a verdict against SoccerMaster. The paper's benchmark numbers make the zero-output path suspicious enough to debug first.

## Leaderboard Tracking

| Date Checked | Board | Rank 1 Score | Our Best | Source | Notes |
|---|---|---:|---:|---|---|
| 2026-04-27 | Codabench test/challenge | unknown | 0.0000356151 | local HF validation | Codabench page identified; platform score not yet scraped or authenticated. |
