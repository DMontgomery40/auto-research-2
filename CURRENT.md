# Current State

Updated: 2026-05-02

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
- SoccerMaster wiring probe job `69f229c4d70108f37ace174a` completed after the asset-path repair. It loaded the weights and ran CUDA inference on 4 images, but the copied Rondo adapter used the wrong role-label order. Official SoccerMaster defines `ball=0`, `goalkeeper=1`, `other=2`, `player=3`, `referee=4`, `None=5`; the copied adapter labeled id `3` as `ball` and id `4` as `staff`. Reinterpreting the last raw counts means the probe likely emitted `player=1120` and `referee=80`, not zero athlete output.
- Corrected-role SoccerMaster wiring probe job `69f23419d2c8bd8662bd31f2` completed on HF Jobs `t4-small`. It confirmed raw athlete output: `player=1196`, `referee=4`; at confidence `0.05`: `player=731`, `referee=4`; verdict `raw-athlete-output-present`.
- SoccerMaster-to-SynLoc conversion/eval probe job `69f23612d2c8bd8662bd3210` completed on HF Jobs `t4-small`. Best 64-image result: `mAP-LocSim=0.000007373902767781469`, role set `athlete`, threshold `0.01`, 18,268 detections, run artifact `soccermaster-synloc-probe-2026-04-29T16-51-53-869121Z`.
- That score is worse than the 64-image TorchVision baseline (`0.00012376237623762376`). Treat this as evidence that the copied adapter/runtime is still not source-faithful, not as evidence that SoccerMaster is bad.
- The unresolved leak is bigger than role-label plumbing: official SoccerMaster is a video/spatiotemporal model with temporal attention, an official Deformable DETR/MSDeformAttn detection path, and official `PostProcess`. The copied Rondo adapter drops temporal weights, runs single images through plain SigLIP, approximates MSDeformAttn with `grid_sample`, and custom-decodes boxes/NMS.
- `train.py` now exists and must baseline-evaluate the active pretrained football YOLO26 path before any training. Soccana is retired from active defaults and remains only as historical evidence.
- Pretrained YOLO baseline job `69f24012d2c8bd8662bd3267` completed on HF Jobs `t4-small`. It is now historical: YOLO26l scored `mAP-LocSim=0.000046702783485895764` with 2,338 detections; the now-retired Soccana row scored `mAP-LocSim=0.000057407296779217454` with 2,610 detections. Training remains blocked because official SynLoc recall was still `0.0`.
- SSKit dev-kit oracle retry job `69f2462ed70108f37ace17b5` completed on HF Jobs `cpu-upgrade` for 64 validation images and 1,004 annotations. Exact GT `position_on_pitch` scored `mAP-LocSim=1.0`; GT ground keypoints projected by SSKit scored `0.9809895759040843`; GT bbox bottom-center through SSKit keypoint projection and `BBoxLocSimCOCOeval` both scored `0.5686594909116471` with `precision_50=0.9269269269269269` and `recall_50=0.92`.
- Current controller phase is `train_dataset_cache_pending`; no active HF job exists, and issue #10 contains the owner approval for a `$25` budget reload before the next paid `dataset-cache-train-valid` job.
- GitHub issue #7 is historical/live context, but no longer the intended stall point.
- The oracle proves the official data, camera calibration, evaluator, and SSKit projection path are not globally broken. The near-zero pretrained detector scores are on the prediction side: wrong/poor boxes for SynLoc, class or role filtering, source-domain mismatch, postprocess/decode, or using non-SynLoc pretrained detectors as if they were official SynLoc baselines.
- Do not start training from a near-zero detector path. The next work must be dev-kit-first: use the official SSKit result formats, `position_from_keypoint_index`, `BBoxLocSimCOCOeval`, FOOTPASS/official challenge assets, and official baseline/runtime code before any fine-tune.
- Codex Goals are enabled in the local Codex config and are first-class for local worktree sessions. Set the local worktree thread Goal from `GOAL.md` or issue #7. Goals are local-thread steering only; keep markdown, `autonomy/state.json`, and GitHub issues as the durable control plane.
- Research/decision-making is now explicitly local Codex work, not Hugging Face work. Use `scripts/codex_research_tick.sh` to launch one local `gpt-5.5`/`xhigh` research tick; HF Jobs remain CUDA-only execution payloads.
- AK-style cadence now has an executable local loop: `scripts/codex_research_loop.sh` runs one local research chance every `300` seconds and dispatches the GitHub Actions heartbeat only when the pushed state is actionable.
- Future SoccerMaster wiring probes should use the cheapest viable HF CUDA flavor, currently `t4-small`, with tight timeouts. Escalate to `l4x1` only after a documented T4 memory/runtime failure or a clear full-run reason.
- Council requests now include `COUNCIL_DOSSIER.md`, autonomy state/events, budget, and baseline source so the council can give high-context criticism before the next expensive run.

## Unknowns

- Current public/test/challenge leaderboard top score.
- Codabench credentials/session status.
- Exact Hugging Face storage layout for predictions, checkpoints, metrics, and logs.

## Next Action

1. Wake the heartbeat so it can cache `train,valid` under the refreshed `$50` budget cap.
2. After the cache job completes, record artifacts and only then advance toward a real source-specific pose/keypoint train/valid experiment.
3. In parallel, run `scripts/codex_research_loop.sh --iterations 12` locally for a bounded one-hour no-spend research pass. The local Codex pass may improve the next train/valid pose experiment plan, inspect sources, patch code, and update issues, but it must not spend beyond the refreshed cap without owner approval.

<!-- autonomy-snapshot:start -->
## Autonomy Snapshot

- Updated: 2026-05-04T20:44:56.168045Z
- Phase: `train_result_review`
- Active job: `none`
- Spend estimate: `$27.50 / $50.00`
- Blocker: none
- Last result: `first-yolo-train` `69f82ec59d85bec4d76f1d3d` score `n/a` threshold `n/a`
<!-- autonomy-snapshot:end -->
