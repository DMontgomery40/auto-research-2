# Current State

Updated: 2026-05-07

## Mission

Beat the best tracked score for the 2026 Spiideo SoccerNet SynLoc challenge by
June 30, 2026.

Primary metric: official SSKit `mAP-LocSim`, higher is better.

## Runtime

- Local Codex `gpt-5.5` / `low` is the researcher.
- Hugging Face Jobs are CUDA execution substrate only.
- Private HF dataset/cache repo: `dmontgomery40/auto-research-2-synloc-data`.
- Private HF model/checkpoint repo: `dmontgomery40/auto-research-2-synloc-models`.
- SynLoc data is cleared for this project; the owner has signed the SoccerNet NDA.
- Keep data, checkpoints, predictions, logs, and local references out of git.

## Verified Facts

- Official task: single-frame athlete detection plus world-coordinate pitch
  localization.
- SSKit oracle exact GT `position_on_pitch`: `mAP-LocSim=1.0`.
- SSKit-projected GT ground keypoint: `mAP-LocSim=0.9809895759040843`.
- GT bbox bottom-center through SSKit: `mAP-LocSim=0.5686594909116471`.
- Generic TorchVision full validation baseline:
  `mAP-LocSim=0.00003561507229859677`.
- Pretrained football YOLO26 baseline (`mobadam/football-player-detection`,
  football/sports YOLO26, not verified as SoccerNet/SynLoc-pretrained):
  `mAP-LocSim=0.000046702783485895764`, `recall_50=0.0`.
- Pose/keypoint smoke: `mAP-LocSim=0.000825082508250825`.
- `first-yolo-train` is a discard:
  `mAP-LocSim=3.572767401302389e-06`, `recall_50=0.0`, worse than pose smoke.
- `keypoint-yolo11n-smoke` is a discard-result:
  `mAP-LocSim=5.743033700121752e-07`, `recall_50=0.0`,
  `gt_recall_px_50=0.05179282868525897`, worse than pose smoke.
- `keypoint-actual-image-scale-smoke` is a keep-mechanics / discard-result:
  job `69fbd6c3aff1cd33e8f2ebb6` added and scored
  `SYNLOC_COORD_SCALE_MODE=actual_image` in the YOLO keypoint lane. Official
  `mAP-LocSim=7.645552200007645e-05`, above the old strict-scale
  YOLO11n-pose keypoint smoke but still below pose smoke
  `0.000825082508250825`; diagnostics stayed weak with
  `gt_recall_iou_0_5=0.0038022813688212928`,
  `gt_recall_px_50=0.03802281368821293`, and mean best GT-to-pred point error
  about `499.6` px. Keep the keypoint scale adapter and verification guard,
  but do not rerun this exact YOLO11n-pose setup as the next scoring direction.
  Artifact upload still failed after `AUTONOMY_RESULT` with HF model-repo LFS
  `403`.
- `tmoklc-football-candidate-audit` is a keep-signal: job
  `69fbdb30317220dbbd1a56a4` evaluated
  `tmoklc/football-player-detection` on 16 validation frames through
  `SYNLOC_COORD_SCALE_MODE=actual_image`. The model labels loaded as
  `0=ball`, `1=goalkeeper`, `2=player`, `3=referee`; 429 detections for 301 GT
  boxes scored official `mAP-LocSim=0.007351653532700209`, above pose smoke
  and prior direct-point synthetic smokes. Candidate diagnostics finally look
  useful for a real source: `gt_recall_iou_0_5=0.840531561461794`,
  `gt_recall_iou_0_3=0.9269102990033222`,
  `mean_best_iou_gt_to_det=0.6984613095305611`, and
  `det_precision_iou_0_5=0.6713286713286714`. Treat this as the next real
  candidate source to pair with the direct point regressor, not as a reason to
  resume broad public-detector hunting. Artifact upload still failed after
  `AUTONOMY_RESULT` with HF model-repo PR/write `403`.
- `point-regressor-tmoklc-yolo-candidates` is a discard-result but a useful
  bridge-audit warning: job `69fbdd96aff1cd33e8f2ec00` trained the direct point
  regressor on 64 train images for 2 epochs and evaluated it on 32 validation
  images with `tmoklc/football-player-detection` real YOLO candidates through
  `SYNLOC_COORD_SCALE_MODE=actual_image`. Official `mAP-LocSim=0.0`,
  `precision_50=0.0`, and `recall_50=0.0`; candidate diagnostics collapsed to
  `gt_recall_iou_0_5=0.0019011406844106464`,
  `gt_recall_iou_0_3=0.011406844106463879`, and
  `mean_best_iou_gt_to_det=0.0074524783749109385`, despite the earlier 16-frame
  tmoklc detector-only audit looking strong. This point bridge used
  `POINT_DETECTOR_IMGSZ=960` by default and top-25 candidates, while the good
  detector audit used `YOLO_IMGSZ=1280`; rerun the bridge with detector settings
  matched to the audit before discarding tmoklc as a source. Artifact upload
  failed after `AUTONOMY_RESULT` with HF model-repo LFS `403`.
- `tmoklc-matched-bridge-libgl-blocker` added no score: job
  `69fbe081aff1cd33e8f2ec29` attempted that matched tmoklc bridge with
  `POINT_DETECTOR_IMGSZ=1280` and `POINT_MAX_DETECTIONS_PER_IMAGE=50`.
  Repo-local preflight correctly stopped on missing HF `job.write`, so the
  Hugging Face Jobs connector launched committed raw `train.py` from ref
  `6d5881258224855734aab8923dd0f2f23b568d1a`. The job failed before training
  or official SSKit evaluation while importing Ultralytics/OpenCV:
  `ImportError: libGL.so.1: cannot open shared object file`. This is a
  packaging blocker, not a tmoklc/model verdict. Fix the UV/OpenCV dependency
  surface so point-regressor jobs import headless `cv2` without system GL, then
  rerun the same matched bridge command.
- `opencv-headless-rfdetr-lazy-install` is a keep-mechanics fix for that
  blocker. The default UV payload now pins `opencv-python-headless>=4.10,<5`
  for YOLO/Ultralytics jobs and no longer installs `rfdetr==1.2.1` in every
  run, avoiding a transitive GUI OpenCV dependency before point-regressor code
  can execute. RF-DETR support remains available behind lazy installation only
  when `TRAIN_MODE=rfdetr_baseline` is selected, and `scripts/verify.sh` guards
  the dependency split. Rerun the exact matched tmoklc bridge command next.
- `tmoklc-matched-bridge-scored` is a discard-result and resolves the prior
  `libGL.so.1` blocker: job `69fbe2c0aff1cd33e8f2ec42` reran the matched
  tmoklc direct point-regressor bridge with `POINT_DETECTOR_IMGSZ=1280` and
  `POINT_MAX_DETECTIONS_PER_IMAGE=50` from committed ref
  `96ccb7eae8da508e1ccda25ca395a3bd2735fcfc`. It reached official SSKit
  evaluation, but official `mAP-LocSim=0.0`, `precision_50=0.0`,
  `recall_50=0.0`; candidate diagnostics stayed collapsed with
  `gt_recall_iou_0_5=0.0019011406844106464`,
  `gt_recall_iou_0_3=0.011406844106463879`,
  `mean_best_iou_gt_to_det=0.007551797826536668`,
  `gt_recall_px_50=0.04182509505703422`, and mean GT-to-pred point error about
  `457.3` px from 617 predictions for 526 GT boxes. Do not rerun this exact
  matched bridge unchanged. The 16-frame detector-only tmoklc audit remains a
  useful anomaly/source signal; next work should audit why the detector-only
  slice looked strong while the bridge slice collapses, or move to a stronger
  official-runtime candidate source. Artifact upload still failed after
  `AUTONOMY_RESULT` with HF model-repo LFS `403`.
- `tmoklc-detector-class-slice-audit` is a keep-signal and narrows the bridge
  bug: job `69fbe74baff1cd33e8f2ec7c` added/scored
  `TRAIN_MODE=detector_class_audit` from committed ref
  `9e75f47d5e015a284f9e0a933e7048d636aee8be`, comparing the same
  `tmoklc/football-player-detection` detector across `all=0,1,2,3`,
  `athletes=1,2,3`, and `player=2` on 16 and 32 validation frames with
  `SYNLOC_COORD_SCALE_MODE=actual_image`, `YOLO_IMGSZ=1280`, and
  `YOLO_CONF=0.01`. The detector stayed strong on the exact 32-frame slice:
  athlete-only scored official `mAP-LocSim=0.006958124383866958`,
  `gt_recall_iou_0_5=0.8384030418250951`, 739 detections for 526 GT boxes,
  and `det_precision_iou_0_5=0.6752368064952639`. Player-only still had
  `gt_recall_iou_0_5=0.6901140684410646` and
  `mAP-LocSim=0.006993505802193122`. So the point-regressor bridge collapse is
  not explained by validation slice size, class ids, image scale, or detector
  recall; it is likely a bridge ingestion/training/eval-path bug. A first
  connector launch `69fbe651aff1cd33e8f2ec6f` failed before repo code because
  the raw GitHub URL used a mistyped SHA and returned `404: Not Found`.
  Artifact upload again failed after `AUTONOMY_RESULT` with HF model-repo
  `403`.
- `tmoklc-bridge-split-leak-audit` is a blocker-fix update, not a model
  verdict: job `69fbea40aff1cd33e8f2eca0` reran the matched tmoklc
  detector-to-point bridge from committed ref
  `e419c94b066d17b41082a958ebb803a89415c150` with extra pre-point detector
  diagnostics. It again reached official SSKit evaluation and printed
  `mAP-LocSim=0.0`, but the new diagnostics showed the collapse was already
  present in `raw_detector_boxes_before_point`, before point crops or top-k:
  `gt_recall_iou_0_5=0.0019011406844106464` with 617 detector boxes for 526 GT
  boxes. Audit examples exposed the likely cause: point-regressor jobs download
  both train and val archives, and duplicate basenames such as `000000.jpg`
  could resolve to the train image while using validation annotations. `train.py`
  now passes split hints through image lookup for detector, keypoint, and point
  paths, and `scripts/verify.sh` guards same-basename train/val selection.
  Rerun the same matched tmoklc bridge from the split-aware commit next; do not
  treat the pre-fix score as a detector/model result. Artifact upload still
  failed after `AUTONOMY_RESULT` with HF model-repo LFS `403`.
- `tmoklc-split-aware-bridge-scored` is a keep-signal and resolves the
  duplicate-basename split-leak blocker: job `69fbee83317220dbbd1a56f6` reran
  the same matched tmoklc detector-to-point bridge from split-aware committed
  ref `991b6d32a62fe49b5ce70a89cc524a4dc8dd8982` through the Hugging Face Jobs
  connector after repo-local preflight correctly stopped on missing local
  `job.write`. Official `mAP-LocSim=0.008787128712871288`, above the prior
  tmoklc detector-only audit `0.007351653532700209` and far above pose smoke.
  The raw detector path recovered on the exact 32-frame bridge slice:
  `raw_detector_boxes_before_point.gt_recall_iou_0_5=0.8384030418250951`,
  `gt_recall_iou_0_3=0.9106463878326996`,
  `det_precision_iou_0_5=0.6752368064952639`, and 739 detections for 526 GT
  boxes. Point diagnostics are now useful rather than collapsed:
  `gt_recall_px_50=0.9315589353612167` with mean best GT-to-pred point error
  about `38.27` px. Keep the split-aware lookup and tmoklc point bridge as the
  current best real-candidate direction. Artifact upload still failed after
  `AUTONOMY_RESULT` with HF model-repo LFS `403`; the printed job log is the
  durable score.
- `tmoklc-128-train-bridge-scored` is a new best keep-signal: job
  `69fbf206317220dbbd1a5719` kept the split-aware tmoklc detector-to-point
  bridge fixed, increased only the point-regressor train slice from 64 to 128
  images, and held validation at 32 images with `POINT_DETECTOR_IMGSZ=1280`,
  `POINT_MAX_DETECTIONS_PER_IMAGE=50`, `POINT_EPOCHS=2`, and
  `POINT_BATCH=16`. Official `mAP-LocSim=0.009405940594059406`, above the prior
  best `0.008787128712871288`. Detector/candidate diagnostics stayed healthy:
  `raw_detector_boxes_before_point.gt_recall_iou_0_5=0.8384030418250951`,
  `gt_recall_iou_0_3=0.9106463878326996`,
  `det_precision_iou_0_5=0.6752368064952639`, and 739 detector boxes for 526
  GT boxes. Point diagnostics were essentially similar to the 64-image run:
  `gt_recall_px_50=0.9315589353612167`, mean best GT-to-pred point error about
  `38.24` px, and 739 predictions. Keep the 128-image bridge as the current
  best real-candidate score; next loop should change one different lever, such
  as candidate filtering or point-head/loss, rather than merely rerunning the
  same 128-image smoke. Artifact upload still failed after `AUTONOMY_RESULT`
  with HF model-repo LFS `403`; the printed job log is the durable score.
- `tmoklc-128-train-4epoch-bridge` is a discard-result: job
  `69fbf50aaff1cd33e8f2ed35` held the current best 128-image split-aware
  tmoklc bridge fixed and changed only `POINT_EPOCHS=2` to `POINT_EPOCHS=4`
  from committed ref `ba7f14a9c86dd0d460d76bb742fabd1fbefb88af`. It reached
  official SSKit evaluation and exactly tied, but did not improve, the current
  best official `mAP-LocSim=0.009405940594059406`. Detector diagnostics stayed
  healthy at `raw_detector_boxes_before_point.gt_recall_iou_0_5=0.8384030418250951`,
  `gt_recall_iou_0_3=0.9106463878326996`, `det_precision_iou_0_5=0.6752368064952639`,
  and 739 detections for 526 GT boxes. Point losses continued down, but mean
  best GT-to-pred point error was slightly worse at about `38.96` px, with
  `gt_recall_px_50=0.9315589353612167`. Do not spend the next loop on more
  epochs alone; change point-head/loss, image size/crop context, or candidate
  filtering. Artifact upload still failed after `AUTONOMY_RESULT` with HF
  model-repo LFS `403`; the printed job log is the durable score.
- `keypoint-topk25-smoke` is a plumbing warning, not a model verdict:
  top-25-per-frame filtering reduced candidate noise but official SSKit still
  printed `mAP-LocSim=0.000`, `precision_50=0.000`, `recall_50=0.000`, and
  `frame_accuracy=0.000`; the same job then failed artifact upload with HF
  model-repo LFS `403` because the connector token had read but not write
  permission.
- `keypoint-audit-samples` added no new score yet: `train.py TRAIN_MODE=keypoint`
  now saves model class names, raw class counts, GT rows, top predicted
  keypoints, and nearest-GT pixel distances in the validation audit artifacts,
  but the local HF CLI could not submit the smoke because no Hugging Face token
  or login was present in this shell.
- `keypoint-score-mode-audit` added no new score yet: the keypoint lane now has
  `YOLO_KEYPOINT_SCORE_MODE=combined|box|keypoint` so the next tiny cloud smoke
  can separate bad point placement from bad candidate/COCO score ranking.
- `keypoint-score-mode-matrix` is a discard-result: job
  `69f938fd98a8d679adfb9346` trained one epoch on 64 train / 32 valid images
  and compared `combined`, `box`, and `keypoint` ranking from the same
  checkpoint. Best official score was `combined` with
  `mAP-LocSim=7.407224512974988e-06`, `recall_50=0.0`,
  `gt_recall_px_50=0.017110266159695818`, and
  `gt_recall_iou_0_5=0.0019011406844106464`, far below the pose smoke.
- `bbox-bottom-center-keypoint-target` added no new score yet: `train.py
  TRAIN_MODE=keypoint` now supports
  `SYNLOC_KEYPOINT_TARGET=annotation|bbox_bottom_center`, so the same official
  `position_from_keypoint_index=0` path can test whether a model learns the
  easier bbox bottom-center proxy that scored `mAP-LocSim=0.5686594909` with
  GT boxes. A tiny `t4-small` smoke was attempted with 64 train / 32 valid /
  1 epoch, but no HF job was created because this shell is not logged into
  Hugging Face and `HF_TOKEN` is unset.
- `bbox-bottom-center-keypoint-smoke` is a discard-result: job
  `69f93dc79d85bec4d76f29ed` trained one epoch on 64 train / 32 valid images
  with `SYNLOC_KEYPOINT_TARGET=bbox_bottom_center`. Official
  `mAP-LocSim=7.316002536214212e-06`, `precision_50=0.0024630541871921183`,
  `recall_50=0.0`, `gt_recall_px_50=0.017110266159695818`, and
  `gt_recall_iou_0_5=0.0019011406844106464`, still far below the pose smoke.
- `direct-point-regressor-oracle-candidates` added no new score yet:
  `train.py TRAIN_MODE=point_regressor` now trains a tiny crop-based direct
  point regressor and evaluates through official SSKit
  `position_from_keypoint_index=0`, using GT boxes as oracle validation
  candidates to isolate point quality from detector recall. This is not a
  challenge-submittable setup. A tiny `t4-small` smoke command was prepared for
  64 train / 32 valid / 2 epochs, but no HF job was created because this shell
  is not logged into Hugging Face.
- `direct-point-regressor-oracle-smoke` is a keep-signal but not a valid
  submission path: job `69fa5b70f2f4addb7839bfba` trained on 64 train / 32
  valid images for 2 epochs using GT boxes as oracle candidates and scored
  official `mAP-LocSim=0.0027874657923080423`, beating pose smoke
  `0.000825082508250825`. Candidate boxes were oracle-perfect
  (`gt_recall_iou_0_5=1.0`) and point diagnostics improved over YOLO11n-pose
  (`gt_recall_px_50=0.39272030651340994`), so direct point quality is worth a
  next experiment only if paired with real candidates. The job also confirmed a
  persistent artifact blocker: model-repo LFS upload still fails with HF `403`
  because the injected token has read but not write permission.
- `point-regressor-yolo-candidates` is a discard-result: job
  `69fa5f40f2f4addb7839bfd9` trained the same tiny direct point regressor and
  evaluated it on real `mobadam/football-player-detection` YOLO26 candidate
  boxes with top-25 per image. Official `mAP-LocSim=0.0`,
  `precision_50=0.0`, `recall_50=0.0`; diagnostics show the candidate source
  failed before point quality mattered: `gt_recall_iou_0_5=0.0`,
  `gt_recall_iou_0_3=0.0019011406844106464`,
  `gt_recall_px_50=0.0019011406844106464`, 508 predictions for 526 GT boxes.
  Artifact upload again failed with HF model-repo LFS `403`.
- `yolo11n-coco-person-candidate-audit` is a discard-result: job
  `69fa6402f2f4addb7839bffe` evaluated public Ultralytics `yolo11n.pt` COCO
  person detections on 64 validation frames through official SSKit bottom-center
  projection. Official `mAP-LocSim=4.0550130098334066e-05`,
  `precision_50=0.006825938566552901`, `recall_50=0.0`,
  `gt_recall_iou_0_5=0.00099601593625498`, and 4,014 person detections for
  1,004 GT boxes. This is slightly better than the full generic TorchVision
  baseline but far below the pose smoke and does not solve candidate recall.
  Two earlier connector launches exposed HF Jobs packaging hazards
  (`xtcocotools` build isolation and NumPy ABI); the working recipe used
  Python 3.10, `numpy<2`, and the published `xtcocotools` wheel. Artifact
  upload still failed with HF model-repo `403`.
- `football-yolo26-imgsz1600-candidate-audit` is a discard-result: job
  `69fa6701f2f4addb7839c00a` evaluated the football YOLO26 detector at larger
  `YOLO_IMGSZ=1600` and lower `YOLO_CONF=0.001` on 64 validation frames. It
  produced 2,125 detections for 1,004 GT boxes, but official
  `mAP-LocSim=4.877335024142809e-06`, `recall_50=0.0`,
  `gt_recall_iou_0_5=0.0`, and
  `mean_best_iou_gt_to_det=0.0012062984910266583`. Image scale and confidence
  threshold did not rescue this candidate source. Artifact upload still failed
  with HF model-repo `403`.
- `easychamp-martinjolif-source-candidate-audit` is a discard-result: job
  `69fa6a3ab745af80fb3734f9` evaluated two different soccer-specific public
  detector candidates on 64 validation frames. Best was
  `martinjolif-yolo11m` with official
  `mAP-LocSim=1.0820754206568197e-05`, `precision_50=0.00273224043715847`,
  `recall_50=0.0`, `gt_recall_iou_0_5=0.0`,
  `gt_recall_iou_0_3=0.00099601593625498`, and
  `mean_best_iou_gt_to_det=0.0013935321462024005`; `easychamp-yolov8` was
  similar at `mAP-LocSim=9.32798528843463e-06`. These public soccer detectors
  do not unblock direct point regression. First launch
  `69fa6a1db745af80fb3734f7` failed before scoring on the known Python 3.12
  `xtcocotools` build-isolation issue; Python 3.10 remains the working HF Jobs
  recipe. Artifact upload still failed with HF model-repo `403`.
- `rtdetr-r18-coco-person-candidate-audit` is a discard-result: job
  `69fa6d7ff2f4addb7839c046` evaluated a non-YOLO COCO RT-DETR person detector
  on 64 validation frames through official SSKit bottom-center projection.
  Official `mAP-LocSim=7.549363399931301e-06`, `precision_50=0.0013343499809378575`,
  `recall_50=0.0`, `gt_recall_iou_0_5=0.0069721115537848604`, and
  `mean_best_iou_gt_to_det=0.029056858644601863`. This had slightly better
  candidate recall than prior public-detector audits but remains far below the
  pose smoke and does not unblock direct point regression. `train.py` now keeps
  a reusable `TRAIN_MODE=transformer_baseline` lane for future non-YOLO
  detector audits. Artifact upload still failed with HF model-repo `403`.
- `uisikdag-yolov8-football-candidate-audit` is a discard-result: corrected
  job `69fa71b6b745af80fb373541` evaluated
  `uisikdag/yolo-v8-football-players-detection` on 64 validation frames with
  athlete classes `1,2,3` after initial job `69fa7058f2f4addb7839c058`
  accidentally selected `ball=0`. The corrected official score was
  `mAP-LocSim=7.084787190704759e-06`, `precision_50=0.0017889087656529517`,
  `recall_50=0.0`, `gt_recall_iou_0_5=0.0`, and
  `mean_best_iou_gt_to_det=0.003439228441225626` from 2,265 detections for
  1,004 GT boxes. This older public football YOLO source does not unblock
  direct point regression. Artifact upload still failed with HF model-repo
  `403`.
- `rfdetr-soccernet-large-parity-smoke` is a discard-result but resolves the
  prior shape blocker: job `69fa7cefb745af80fb3735c0` loaded
  `julianzu9612/RFDETR-Soccernet` with `RFDETRLarge`, proving the previous
  `192/384` versus `128/256` tensor mismatch was caused by using
  `RFDETRBase`. The 16-frame official SSKit smoke still scored
  `mAP-LocSim=0.0`, `precision_50=0.0`, `recall_50=0.0`,
  `gt_recall_iou_0_5=0.0`, and `mean_best_iou_gt_to_det=0.0012578130198192673`
  from 156 detections for 301 GT boxes. Keep the `RFDETRLarge` mechanics, but
  do not treat this SoccerNet-Tracking checkpoint as the next candidate source
  without a new preprocessing/coordinate-parity reason. Artifact upload still
  failed with HF model-repo `403`.
- `soccana-yolo11-soccernet-candidate-audit` is a discard-result: job
  `69fa8013b745af80fb3735e4` evaluated `Adit-jain/soccana`, whose loaded
  labels were `0=Player`, `1=Ball`, `2=Referee`, on 64 validation frames at
  `YOLO_IMGSZ=1280` and `YOLO_CONF=0.01`. Official
  `mAP-LocSim=4.304778303917348e-05`, `precision_50=0.004347826086956522`,
  `recall_50=0.0`, `gt_recall_iou_0_5=0.0`, and
  `mean_best_iou_gt_to_det=0.0018615731134729054` from 1,584 detections for
  1,004 GT boxes. This SoccerNet-labeled public YOLO11 detector does not
  unblock direct point regression. First wrapper job
  `69fa7ffef2f4addb7839c083` failed before scoring because the wrapper omitted
  `train.py` dependencies; artifact upload still failed with HF model-repo
  `403`.
- `point-regressor-jittered-candidates` is a discard-result but found a more
  important blocker: job `69fabd24b745af80fb3738a3` trained the same tiny
  direct point regressor on 64 train / 32 valid images, then evaluated it with
  deterministically jittered GT boxes
  (`POINT_JITTER_CENTER_FRAC=0.10`, `POINT_JITTER_SCALE_FRAC=0.15`). Official
  `mAP-LocSim=0.0006912422136296282`, below pose smoke
  `0.000825082508250825` and far below GT-box oracle point regression
  `0.0027874657923080423`. Diagnostics were
  `gt_recall_iou_0_5=0.3563218390804598`,
  `gt_recall_px_50=0.39272030651340994`, and 522 predictions for 522 GT
  boxes. The saved audit examples exposed an image/annotation scale blocker:
  several GT boxes on image `0` have x coordinates like `2442`, `2737`, and
  `2928`, while the opened image width forced jittered candidate boxes to clamp
  at `1919`, creating zero-IoU candidates from GT-derived boxes. Before more
  model-source searches, audit whether `image_path()` is selecting resized
  images for fullhd annotations, whether annotation dimensions disagree with
  the actual image files, and whether all detector/keypoint/point lanes need a
  coordinate-scale adapter. Artifact upload again failed with HF model-repo LFS
  `403`; the job log `AUTONOMY_RESULT` recap is the durable result source.
- `image-path-dimension-guard` added no new score yet but closes the concrete
  file-selection blocker exposed by the jittered-GT audit. `train.py` now
  resolves SynLoc image paths with the COCO image record when available and
  requires the actual PIL image size to match annotation `width`/`height`.
  If only resized same-basename files are found for fullhd annotations, the
  run now fails with the candidate sizes instead of silently clamping boxes to
  the wrong image width. Local `python3 -m py_compile train.py` and
  `scripts/verify.sh` passed, including a duplicate-basename resized/fullhd
  regression check.
- `image-cache-scale-blocker-cloud` added no new score, but proved the cloud
  data path is currently mismatched: job `69fac036f2f4addb7839c18b` reran the
  jittered-GT point-regressor smoke through the dimension guard and failed
  before training/evaluation because annotation image `000000.jpg` expected
  `3840x2160`, while the available train and validation image candidates were
  both `1920x1080`. This confirms the prior zero-IoU jittered candidates were
  contaminated by image/annotation scale mismatch. First connector job
  `69fabfe7f2f4addb7839c185` failed before running on Python 3.12 from the
  known `xtcocotools`/`numpy` build-isolation issue; Python 3.10 remains the
  working HF Jobs recipe.
- `actual-image-scale-jittered-point-smoke` is a keep-signal but not a valid
  submission path: job `69fac2a1b745af80fb3738f9` used
  `SYNLOC_COORD_SCALE_MODE=actual_image` to train/crop on the actual
  `1920x1080` cached images while emitting predictions back in `3840x2160`
  annotation coordinates for SSKit. Official
  `mAP-LocSim=0.0032123790168145185`, above the prior jittered-GT smoke
  `0.0006912422136296282`, pose smoke `0.000825082508250825`, and oracle-box
  direct point smoke `0.0027874657923080423`. Candidate diagnostics recovered:
  `gt_recall_iou_0_5=0.9578544061302682`,
  `mean_best_iou_gt_to_det=0.6846319161542074`,
  `gt_recall_px_50=0.9980842911877394`, and
  `mean_best_px_gt_to_pred=12.602958457397472`. This proves the scale adapter
  is the correct immediate mechanics fix for cached fullHD annotations paired
  with half-size images, but the score is not challenge-submittable because the
  candidates are jittered GT boxes. Artifact upload still failed with HF
  model-repo LFS `403`; the printed `AUTONOMY_RESULT` log is the durable
  result. First connector launch `69fac28cf2f4addb7839c1a5` failed before
  running because the raw GitHub URL had a bad SHA and returned `404: Not
  Found`.
- `actual-image-scale-yolo26-point-smoke` is a discard-result: job
  `69fac472b745af80fb37390f` reran the direct point regressor with
  `SYNLOC_COORD_SCALE_MODE=actual_image` and real default
  `mobadam/football-player-detection` YOLO26 candidates. Official
  `mAP-LocSim=7.984669434685405e-06`, `precision_50=0.002688172043010753`,
  `recall_50=0.0`, and 508 predictions for 526 GT boxes. The coordinate
  adapter improved the pre-adapter YOLO bridge diagnostics but did not make the
  candidate source useful: `gt_recall_iou_0_5=0.0019011406844106464`,
  `gt_recall_iou_0_3=0.009505703422053232`,
  `mean_best_iou_gt_to_det=0.0072488041644957375`, and
  `gt_recall_px_50=0.039923954372623575`. Artifact upload still failed with HF
  model-repo LFS `403`; the printed `AUTONOMY_RESULT` log is the durable
  result.
- `detector-baseline-scale-adapter` added no new score yet but closes a
  follow-on coordinate blocker in the candidate-audit lanes. `train.py` now
  supports `SYNLOC_COORD_SCALE_MODE=actual_image` for YOLO, transformer, and
  RF-DETR detector baselines, mapping boxes from the actual cached image size
  back to annotation coordinates before image-space diagnostics, bottom-center
  projection, and official SSKit evaluation. Local `python3 -m py_compile
  train.py` and `scripts/verify.sh` passed, including detector-box backscale
  coverage. No HF job was launched from the dirty mechanics pass.
- `keypoint-actual-image-scale-smoke` extends the same coordinate-scale repair
  into the YOLO keypoint lane. `train.py` now trains keypoint labels in actual
  cached-image coordinates and maps predicted keypoint boxes/points back to
  annotation coordinates before official SSKit evaluation. The cloud score
  improved over the old strict-scale keypoint run but remained below pose smoke,
  so the adapter is kept as mechanics and the exact YOLO11n-pose setup is
  discarded as a scoring direction.
- `actual-image-hamza-football-candidate-audit` is a discard-result: job
  `69fac706b745af80fb373923` evaluated
  `HamzaAliKhan/football-players-detection` through the repaired
  `SYNLOC_COORD_SCALE_MODE=actual_image` YOLO detector-audit path on 64
  validation frames. Loaded labels were `0=ball`, `1=goalkeeper`, `2=player`,
  `3=referee`, and the run produced 2,290 detections for 1,004 GT boxes, but
  official `mAP-LocSim=1.3407689313908168e-05`, `recall_50=0.0`,
  `precision_50=0.002554278416347382`,
  `gt_recall_iou_0_5=0.00099601593625498`,
  `gt_recall_iou_0_3=0.012948207171314742`, and
  `mean_best_iou_gt_to_det=0.00950942234585973`. This public football detector
  does not unblock the direct point regressor. Artifact upload again failed
  with HF model-repo `403`; the printed `AUTONOMY_RESULT` log is the durable
  result. First connector launch `69fac6f1f2f4addb7839c1af` failed before
  running because the raw GitHub URL used a mistyped SHA and returned
  `404: Not Found`.
- `actual-image-rtdetr-coco-person-candidate-audit` is a discard-result: job
  `69fac8c4b745af80fb373937` reran `PekingU/rtdetr_r18vd` COCO person boxes
  through the repaired `SYNLOC_COORD_SCALE_MODE=actual_image`
  transformer-audit path on 64 validation frames. Official
  `mAP-LocSim=7.543611504007545e-06`, `precision_50=0.0013333333333333333`,
  `recall_50=0.0`, `gt_recall_iou_0_5=0.0069721115537848604`, and
  `mean_best_iou_gt_to_det=0.02903191841477167` from 10,000 detections for
  1,004 GT boxes. This is effectively unchanged from the pre-adapter RT-DETR
  audit, so the actual-image adapter does not rescue generic RT-DETR person
  candidates. Artifact upload again failed with HF model-repo `403`; the
  printed `AUTONOMY_RESULT` log is the durable score.
- `repo-local-hf-env-loader-bug` added no score: the repo-local helper had a
  false credential failure because `scripts/run_hf_train.sh` checked only the
  inherited shell environment and `hf auth whoami`; it did not load the
  repo-local `.env` that already contains `HF_TOKEN`. Sourcing `.env`
  authenticates as `dmontgomery40`, and prior connector-launched HF jobs were
  real. This was a helper/env-loading bug, not a Hugging Face access blocker.
- `hf-dry-run-reachability-guard` added no score but closes the bad-SHA HF
  packaging blocker family. `scripts/run_hf_train.sh --dry-run` now checks
  that the selected `HF_GIT_REF` exists on the configured remote before
  printing the raw GitHub `train.py` URL, so a copied dry-run command should
  not launch a `404: Not Found` job for a local-only commit.
- `hf-submit-preflight` added no score but gives the repo-local helper a
  non-submitting submission check. After the `.env` loader fix, preflight
  reads repo-local `HF_TOKEN`, verifies the selected git ref is pushed, and
  prints the HF Jobs command without exposing the token.
- `hf-job-write-permission-blocker` and
  `hf-job-write-permission-recheck` added no score but exposed the exact local
  submission boundary: `.env` loads and authenticates as `dmontgomery40`, but
  the fine-grained token lacks `job.write`, so the repo-local `hf` CLI cannot
  create Jobs. This is not an HF access blocker for connector-launched jobs.
- `actual-image-rfdetr-soccernet-large-candidate-audit` used the Hugging Face
  Jobs connector to run the selected RF-DETR Large audit anyway. Job
  `69fb0327b745af80fb373bd5` ran on `t4-small`, downloaded the SynLoc archives
  and `julianzu9612/RFDETR-Soccernet`, loaded `RFDETRLarge` on CUDA, and reached
  official SSKit eval. Score was `mAP-LocSim=0.0`, `precision_50=0.0`,
  `recall_50=0.0`, with 566 detections for 526 GT boxes,
  `gt_recall_iou_0_5=0.0019011406844106464`, and
  `mean_best_iou_gt_to_det=0.007630024549978594`. Discard this RF-DETR
  SoccerNet checkpoint as a current candidate source.
- `hf-jobwrite-and-upload-guard` added no score but prevents this failure family
  from repeating silently: helper preflight now detects fine-grained
  no-`job.write` tokens before actual submission, the loop contract routes that
  case to the Hugging Face Jobs connector/app when available, and `train.py`
  does not turn a scored experiment into a failed iteration just because
  artifact upload is blocked.
- `point-regressor-train-jitter-smoke` is a discard-result: job
  `69fbcd06317220dbbd1a5657` trained the direct point regressor with
  deterministic jittered GT crops
  (`POINT_TRAIN_JITTER_CENTER_FRAC=0.10`,
  `POINT_TRAIN_JITTER_SCALE_FRAC=0.15`) and evaluated the same actual-image
  jittered-GT candidate smoke. Official
  `mAP-LocSim=0.003013072317418499`, below the previous no-train-jitter
  actual-image jittered smoke `0.0032123790168145185`; diagnostics remained
  strong for the synthetic candidates (`gt_recall_iou_0_5=0.9578544061302682`,
  `gt_recall_px_50=0.9980842911877394`) with mean point error about `12.67`
  px. The temporary train-jitter code was reverted because the score did not
  justify widening `train.py`. Artifact upload again failed after
  `AUTONOMY_RESULT` with HF model-repo LFS `403`.
- `pericles-player-detector-candidate-audit` is a discard-result: job
  `69fbd098aff1cd33e8f2eb52` evaluated public
  `PericlesRodrigues01/player-detector` through the repaired
  `SYNLOC_COORD_SCALE_MODE=actual_image` YOLO detector-audit path on 64
  validation frames. The model labels matched its README
  (`0=Person`, `1=Ball`, `2=Equipment`) and it produced 7,726 detections for
  1,004 GT boxes, but official `mAP-LocSim=8.406501027461238e-05`,
  `precision_50=0.009433962264150943`, `recall_50=0.0`,
  `gt_recall_iou_0_5=0.00298804780876494`,
  `gt_recall_iou_0_3=0.013944223107569721`, and
  `mean_best_iou_gt_to_det=0.023235810243179132`. This is below pose smoke and
  does not unblock the direct point regressor. Artifact upload again failed
  after `AUTONOMY_RESULT` with HF model-repo `403`; the printed log is the
  durable score.
- `split-specific-synloc-fetch` added no score but fixes a loop-time mechanics
  waste exposed by the Pericles audit: validation-only baseline, transformer,
  and RF-DETR jobs now fetch only annotations, manifest, and the requested split
  archive instead of `raw/fullhd/*.zip`; train/keypoint/point-regressor jobs
  still fetch train plus valid. `scripts/verify.sh` covers this fetch-pattern
  contract so future validation-only jobs do not silently download train.zip.
- `keremberke-yolov5m-runtime-blocker` added no score: connector job
  `69fbd3a2317220dbbd1a5671` tried the unscored legacy
  `keremberke/yolov5m-football` detector through
  `SYNLOC_COORD_SCALE_MODE=actual_image` on 16 validation frames. The repo
  helper preflight correctly stopped on missing local `job.write`, the
  connector launched the same committed raw GitHub `train.py`, and the job
  fetched only annotations, manifest, and `val.zip`. It failed before official
  SSKit evaluation when the current Ultralytics runtime attempted to load the
  old YOLOv5 checkpoint and raised
  `TypeError: BaseModel.fuse() got an unexpected keyword argument 'verbose'`.
  Treat this as an integration blocker, not a detector verdict. Do not add a
  YOLOv5-specific runtime path without a stronger source-specific reason than
  another generic public football detector.
- Compute rule: use the cheapest option that actually works, always.

## Interpretation

Active direction remains track/pose/keypoint or direct ground-point prediction.

The official data, camera calibration, evaluator, and SSKit projection path are
not globally broken. The failure is on the prediction side.

The latest cloud smoke proved part of the prediction side failure was a
concrete coordinate-scale problem, not just model weakness. GT annotations in
the audit can extend past x=2900, while the cloud cache currently serves
`1920x1080` images for COCO records declaring `3840x2160`.
`SYNLOC_COORD_SCALE_MODE=actual_image` now maps annotations into actual-image
coordinates for crop training/inference and maps emitted prediction boxes and
keypoints back to annotation coordinates for SSKit. Use strict mode by default
when matching full-size files exist; use `actual_image` for the current cached
fullHD jobs until the cache is rebuilt with true annotation-size images.

Zero or near-zero official scores from a soccer/football-pretrained model on
SoccerNet data should be treated as runtime, class mapping, coordinate,
preprocess, prediction-format, or evaluator plumbing failures until audited.
Do not treat them as model underperformance.

`train.py` now has a `TRAIN_MODE=keypoint` lane that trains one projected
ground keypoint and evaluates through official SSKit `position_from_keypoint_index=0`.
The lane now supports `SYNLOC_COORD_SCALE_MODE=actual_image`, so keypoint
training/inference no longer silently mixes 3840x2160 annotations with
1920x1080 cached image coordinates. The actual-image rerun improved the old
strict-scale YOLO11n-pose score but still produced too many noisy detections
and too little point recall to promote. Top-k filtering, the
`combined`/`box`/`keypoint` score-mode matrix, bbox-bottom-center target, and
coordinate-scale repair did not rescue official recall, so do not spend another
pass on this exact YOLO11n-pose setup through confidence ranking, target
switching, or scale plumbing alone. The saved audit examples still show
candidate points often hundreds of pixels from the nearest GT point, which
points to a point-quality/model-family problem more than an SSKit ingestion
problem.

`TRAIN_MODE=point_regressor` beat the pose smoke while given GT boxes, so the
direct crop-regressor point head is not dead. Pairing it with the existing
football YOLO26 candidate source through the repaired `actual_image` coordinate
adapter still scored far below the pose smoke because candidate recall remains
essentially absent. A larger-image/lower-threshold YOLO26 audit also did not
fix it. Public COCO `yolo11n` person detections, public COCO RT-DETR person
detections, and five public soccer/football YOLO detectors
(`easychamp-yolov8`, `martinjolif-yolo11m`, `uisikdag-yolov8-football`,
`Adit-jain/soccana`, `PericlesRodrigues01/player-detector`) all failed as
useful candidate sources despite producing many boxes. The new exception is
`tmoklc/football-player-detection`: its 16-frame actual-image audit had strong
real candidate overlap (`gt_recall_iou_0_5=0.840531561461794`) and official
`mAP-LocSim=0.007351653532700209`. The next loop should pair this detector
with the direct point regressor before scaling or searching more public
detectors. The untried legacy `keremberke/yolov5m-football` source did not
reach scoring because its old YOLOv5 checkpoint is incompatible with the
current Ultralytics runtime, so a YOLOv5 loader is now a mechanics decision
rather than a free candidate audit. Training the same tiny point regressor on
deterministic jittered GT crops did not improve the synthetic jittered-candidate
score, so do not add jittered-crop training knobs back without a new point-head
architecture or candidate-noise hypothesis.

SoccerMaster remains possible only after official-runtime parity. Copied
adapter scores are not valid SoccerMaster verdicts. RF-DETR SoccerNet large
runtime parity is sufficient to score, and the post-adapter RF-DETR Large audit
still had near-zero image-space overlap on SynLoc validation frames, so do not
spend another pass on that checkpoint without a new preprocessing/runtime
parity hypothesis.

## Next Action

Next loop should build on the split-aware tmoklc detector-to-point bridge:
job `69fbee83317220dbbd1a56f6` restored raw detector recall on the bridge path
and set the current best real-candidate score at
`mAP-LocSim=0.008787128712871288`. Do not rerun the exact 64/32, 2-epoch
smoke unchanged. The next useful unit is a bounded point-head or candidate
quality experiment on this same source, such as a slightly larger train slice
or a small point-regressor architecture/loss change, with the same raw detector
diagnostics kept in the log. Keep judging with official SSKit `mAP-LocSim` and
stop if raw detector recall collapses again.

Do not rerun the keypoint actual-image YOLO11n-pose smoke just because the
scale adapter now exists; job `69fbd6c3aff1cd33e8f2ebb6` already showed the
mechanics improvement still lands below pose smoke. Do not rerun the
point-regressor train-jitter variant; it underperformed the existing
no-train-jitter actual-image jittered smoke and its code was reverted. Do not
rerun `PericlesRodrigues01/player-detector`; it produced many boxes but stayed
below pose smoke with near-zero image-space overlap. Do not spend the next pass
building legacy YOLOv5 support for `keremberke/yolov5m-football` unless there
is a concrete source-specific reason to believe that checkpoint is worth a new
runtime lane.
For job submission, use `scripts/run_hf_train.sh --preflight` before the local
CLI path; if it reports missing `job.write`, launch through the Hugging Face
Jobs connector/app with the same raw GitHub `train.py` URL and env, or record a
single cloud-submission blocker if the connector/app is unavailable in that
runtime. Do not rerun the default YOLO26 candidate bridge just because the
coordinate adapter exists; job `69fac472b745af80fb37390f` already gave a clean
discard. Artifact upload should now retry as a Hub PR and remain nonfatal after
`AUTONOMY_RESULT`; job logs are still the durable score source until model-repo
write/PR persistence is proven.
