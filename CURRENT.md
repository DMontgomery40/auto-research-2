# Current State

Updated: 2026-05-05

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
- Compute rule: use the cheapest option that actually works, always.

## Interpretation

Active direction remains track/pose/keypoint or direct ground-point prediction.

The official data, camera calibration, evaluator, and SSKit projection path are
not globally broken. The failure is on the prediction side.

Zero or near-zero official scores from a soccer/football-pretrained model on
SoccerNet data should be treated as runtime, class mapping, coordinate,
preprocess, prediction-format, or evaluator plumbing failures until audited.
Do not treat them as model underperformance.

`train.py` now has a `TRAIN_MODE=keypoint` lane that trains one projected
ground keypoint and evaluates through official SSKit `position_from_keypoint_index=0`.
The tiny YOLO11n-pose smokes proved the wiring but produced too many noisy
detections and too little point recall to promote. Top-k filtering, the
`combined`/`box`/`keypoint` score-mode matrix, and the bbox-bottom-center target
did not rescue official recall, so do not spend another pass on this exact
YOLO11n-pose setup through confidence ranking or target switching alone. The
saved audit examples show candidate points often hundreds of pixels from the
nearest GT point, which points to a point-quality/model-family problem more than
an SSKit ingestion problem.

`TRAIN_MODE=point_regressor` beat the pose smoke while given GT boxes, so the
direct crop-regressor point head is not dead. Pairing it with the existing
football YOLO26 candidate source scored zero because image-space candidate
recall was essentially absent, and a larger-image/lower-threshold YOLO26 audit
did not fix it. Public COCO `yolo11n` person detections, public COCO RT-DETR
person detections, and three public soccer/football YOLO detectors
(`easychamp-yolov8`, `martinjolif-yolo11m`, `uisikdag-yolov8-football`) all
failed as useful candidate sources despite producing many boxes. Do not spend
another pass pairing the point regressor with generic COCO person boxes, COCO
transformer detector boxes, or these public soccer/football YOLO detectors.
The next blocker is an official SSKit/SoccerNet-format candidate source,
SoccerMaster official-runtime parity, or a track/pose source with real
athlete-box recall on the same frames.

SoccerMaster remains a possible lead only after official-runtime parity. Copied
adapter scores are not valid SoccerMaster verdicts.

## Next Action

Choose one genuinely different candidate-generation experiment before
revisiting direct point regression at scale: official SSKit/SoccerNet-format
candidates, SoccerMaster official-runtime parity that proves real athlete boxes
on the same validation frames, or a track/pose source with saved box-recall
diagnostics. Do not spend the next pass on football YOLO26 image-size/confidence
tuning, generic COCO person detectors, COCO RT-DETR detector boxes, or the
public EasyChamp/MartijnJolif/Uisikdag soccer-football YOLO detectors. Keep
treating GT-box oracle point-regressor runs as diagnostics only. Also fix or
work around HF model-repo write permission before relying on uploaded
artifacts; job logs are currently the only durable result source for these
smokes.
