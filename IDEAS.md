# Ideas

Active direction: track/pose/keypoint or direct ground-point prediction.

## Next Best Experiments

- Audit the pretrained-model/evaluator path before adding another model idea:
  verify model class names, selected class ids, preprocessing/image scale, bbox
  format, score thresholds, camera projection, category ids, and SSKit ingestion
  on a tiny slice with saved GT/prediction examples.
- Use the saved keypoint matrix audit examples to pick a different point-quality
  signal or target: current candidate points are often hundreds of pixels from
  the nearest GT point, and confidence ranking did not fix official recall.
- Run the new tiny direct footpoint/ground-point payload:
  `TRAIN_MODE=point_regressor` keeps the official
  `position_from_keypoint_index=0` evaluator path but does not start from
  `yolo11n-pose.pt` COCO pose priors. It uses GT boxes as oracle validation
  candidates to isolate point quality, so treat positive results as direction
  signal rather than challenge-submittable scores.
- Find a better real candidate source before scaling the direct point regressor:
  the first YOLO26 detector-box bridge scored zero because candidate recall was
  essentially absent, the larger-image/lower-threshold YOLO26 audit did not
  fix it, and public COCO `yolo11n.pt` person boxes also stayed far below the
  pose smoke. The useful next unit is source-specific detector/track
  candidates, official SSKit/SoccerNet-format candidates, or SoccerMaster
  runtime parity that proves real athlete boxes on the same frames.
- Improve the new `train.py TRAIN_MODE=keypoint` lane: the first
  YOLO11n-pose footpoint smoke proved `position_from_keypoint_index` wiring but
  scored only `5.743033700121752e-07` with too many noisy detections. A
  follow-up top-25-per-frame filter scored zero, so the next pass needs a new
  point-quality signal, target, or model family rather than confidence-only
  pruning.
- Use SSKit official baseline/runtime code as the format oracle before adding
  model complexity.
- Train on a tiny train slice and evaluate a held-out validation slice with:
  `mAP-LocSim`, point error, image-space recall, and threshold diagnostics.
- Try a direct ground-point head if the official keypoint route is simpler than
  box-to-pitch projection.
- Add source-specific augmentations only after the basic keypoint/ground-point
  payload produces nonzero recall.

## Useful Facts

- SSKit exact GT scores `1.0`.
- SSKit-projected GT ground keypoint scores `0.9809895759040843`.
- BBox bottom-center through SSKit scores `0.5686594909116471`.
- Pose/keypoint smoke scored `0.000825082508250825`.
- `first-yolo-train` scored `3.572767401302389e-06` with `recall_50=0.0` and
  is discarded.
- `keypoint-yolo11n-smoke` scored `5.743033700121752e-07` with
  `recall_50=0.0`; keep the keypoint wiring, discard this exact config.
- `keypoint-topk25-smoke` printed `0.000` after SSKit evaluation with
  top-25-per-frame filtering; treat this as another plumbing warning and discard
  confidence-only pruning as a standalone rescue.
- `keypoint-audit-samples` added saved GT/prediction examples for the keypoint
  lane; the later matrix job uploaded `summary.json` and per-mode validation
  artifacts.
- `keypoint-score-mode-audit` keeps the same model/evaluator path but allows
  box-confidence, keypoint-confidence, and combined-score ranking to be compared
  with official `mAP-LocSim`.
- `keypoint-score-mode-matrix-run` scored only `7.407224512974988e-06` best
  official `mAP-LocSim` with `combined` ranking, `recall_50=0.0`,
  `gt_recall_px_50=0.017110266159695818`, and `gt_recall_iou_0_5=0.0019011406844106464`.
  Discard confidence-only ranking as the next lever for this exact YOLO11n-pose
  keypoint setup.
- `bbox-bottom-center-keypoint-target` added
  `SYNLOC_KEYPOINT_TARGET=annotation|bbox_bottom_center` to `train.py`; no HF
  score exists yet because local `hf` has no token/login in this shell.
- `bbox-bottom-center-keypoint-smoke` scored
  `7.316002536214212e-06` official `mAP-LocSim` with `recall_50=0.0`,
  `gt_recall_px_50=0.017110266159695818`, and
  `gt_recall_iou_0_5=0.0019011406844106464`. Discard target switching alone as
  a rescue for this exact YOLO11n-pose keypoint setup.
- `direct-point-regressor-oracle-candidates` added
  `TRAIN_MODE=point_regressor` to `train.py`; initial local `hf auth whoami`
  returned `Not logged in`, but the later connector job ran the smoke.
- `direct-point-regressor-oracle-smoke` scored
  `0.0027874657923080423` official `mAP-LocSim` on 64 train / 32 valid /
  2 epochs with GT-box oracle candidates, beating pose smoke
  `0.000825082508250825`. Candidate boxes were perfect by construction
  (`gt_recall_iou_0_5=1.0`) and `gt_recall_px_50=0.39272030651340994`, so this
  is a point-head signal and a candidate-generation blocker, not a valid
  submission path.
- `point-regressor-yolo-candidates` scored `0.0` official `mAP-LocSim` on the
  same 64 train / 32 valid / 2 epoch direct point regressor when evaluated with
  `mobadam/football-player-detection` YOLO26 boxes. Candidate diagnostics were
  the failure: `gt_recall_iou_0_5=0.0`, `gt_recall_iou_0_3=0.0019011406844106464`,
  and `gt_recall_px_50=0.0019011406844106464`, so discard this exact
  detector-to-point bridge.
- `yolo11n-coco-person-candidate-audit` scored
  `4.0550130098334066e-05` official `mAP-LocSim` on 64 validation frames with
  public Ultralytics `yolo11n.pt` COCO person detections, `recall_50=0.0`,
  `gt_recall_iou_0_5=0.00099601593625498`, and 4,014 detections for 1,004 GT
  boxes. Discard generic COCO person boxes as the next candidate source for the
  point regressor.
- `football-yolo26-imgsz1600-candidate-audit` scored
  `4.877335024142809e-06` official `mAP-LocSim` on 64 validation frames with
  `YOLO_IMGSZ=1600`, `YOLO_CONF=0.001`, `recall_50=0.0`,
  `gt_recall_iou_0_5=0.0`, and 2,125 detections for 1,004 GT boxes. Discard
  image-size/confidence tuning as a rescue for this football YOLO26 candidate
  source.
- HF model artifact upload still fails with LFS `403` read-only token in
  connector jobs; do not assume artifacts landed in
  `dmontgomery40/auto-research-2-synloc-models` unless write access is fixed or
  a different upload path is proven.

## Maybe Later

- Official-runtime SoccerMaster parity probe: official config, video-shaped
  input, official detection head, official postprocess, and direct comparison
  against the copied adapter on deterministic frames.
- Detector confidence calibration only after the representation has useful
  official recall.
- Make HF result persistence robust after the next meaningful score: fix the
  write-token path or persist tiny JSON summaries somewhere other than model
  repo LFS.
- Ensemble only if two independently useful models have complementary errors.

## Avoid

- Another generic detector threshold sweep with zero official recall.
- More YOLO fine-tune compute from the failed detector representation.
- Scaling the same one-epoch YOLO11n-pose keypoint smoke before fixing noisy
  candidate generation or point recall.
- More target-only tweaks on the same YOLO11n-pose keypoint lane; both the
  annotation and bbox-bottom-center targets stayed far below pose smoke.
- Pairing the direct point regressor with the same `mobadam/football-player-detection`
  YOLO26 candidate boxes; the real-candidate smoke already scored zero because
  image-space candidate recall was essentially absent.
- More image-size or confidence-threshold rescue attempts for
  `mobadam/football-player-detection` YOLO26; `imgsz=1600` and `conf=0.001`
  still had zero IoU-0.5 GT recall.
- Pairing the direct point regressor with generic COCO `yolo11n.pt` person
  candidate boxes; the candidate audit stayed far below the pose smoke and had
  near-zero image-space GT recall despite many detections.
- Treating zero or near-zero official scores from soccer/football-pretrained
  models as model underperformance before auditing runtime/plumbing.
- Another confidence-only candidate filter or score-mode job on the same
  YOLO11n-pose smoke; the matrix already stayed far below the pose smoke.
- Any scheduler, dashboard, database, JSON state machine, or review phase that
  tries to decide what local Codex should decide from markdown.
- Leaked submissions or post-deadline winner writeups.
