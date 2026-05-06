# Ideas

Active direction: track/pose/keypoint or direct ground-point prediction.

## Next Best Experiments

- Re-audit one real candidate source through the coordinate-scale adapter before
  trusting any earlier detector-source discard. The jittered-GT point-regressor
  smoke recovered when `SYNLOC_COORD_SCALE_MODE=actual_image` mapped 3840x2160
  annotation coordinates to the cached 1920x1080 images and back to annotation
  coordinates for SSKit. Next run:
  `TRAIN_MODE=point_regressor SYNLOC_COORD_SCALE_MODE=actual_image
  POINT_CANDIDATE_MODE=yolo TRAIN_MAX_IMAGES=64 VAL_MAX_IMAGES=32
  POINT_EPOCHS=2 POINT_BATCH=16`. Start with the existing default YOLO26 bridge
  or one previously best public soccer/SoccerNet-labeled source, and decide
  from candidate IoU diagnostics whether prior detector failures were mostly
  scale contamination or still true candidate-source failures.
- Later, decide whether to rebuild the private data cache with true
  annotation-size images or keep the explicit `actual_image` coordinate adapter
  for cached fullHD jobs. Strict image-size matching should remain the default
  guard when matching files exist.
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
  fix it, public COCO `yolo11n.pt` person boxes stayed far below the pose
  smoke, public COCO RT-DETR stayed far below the pose smoke despite slightly
  better IoU diagnostics, and public EasyChamp/MartijnJolif/Uisikdag
  soccer-football YOLO detectors also had zero IoU-0.5 GT recall on the same
  frames. `Adit-jain/soccana` is now in the same discard bucket despite its
  SoccerNet-labeled model card and loaded `Player/Ball/Referee` labels. The
  useful next unit is official SSKit/SoccerNet-format candidates, SoccerMaster
  runtime parity that proves real athlete boxes on the same frames, or a
  track/pose candidate source with saved recall diagnostics.
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
- `easychamp-martinjolif-source-candidate-audit` scored
  `1.0820754206568197e-05` best official `mAP-LocSim` on 64 validation frames
  with `martinjolif-yolo11m`; `easychamp-yolov8` scored
  `9.32798528843463e-06`. Both had `recall_50=0.0`,
  `gt_recall_iou_0_5=0.0`, and only `gt_recall_iou_0_3=0.00099601593625498`,
  so discard these public soccer YOLO detectors as the next candidate source
  for the point regressor.
- `rtdetr-r18-coco-person-candidate-audit` scored
  `7.549363399931301e-06` official `mAP-LocSim` on 64 validation frames with
  public `PekingU/rtdetr_r18vd` COCO person detections. It produced 10,006
  detections for 1,004 GT boxes and had better candidate diagnostics than prior
  public-detector audits (`gt_recall_iou_0_5=0.0069721115537848604`), but
  official `recall_50=0.0` and score stayed far below the pose smoke. Keep
  `TRAIN_MODE=transformer_baseline` as reusable mechanics; discard COCO RT-DETR
  person boxes as the next candidate source for the point regressor.
- `uisikdag-yolov8-football-candidate-audit` scored
  `7.084787190704759e-06` official `mAP-LocSim` on 64 validation frames with
  public `uisikdag/yolo-v8-football-players-detection` after correcting class
  ids to athlete labels `1,2,3`; an initial job selected `ball=0`, which was a
  plumbing mistake rather than a model verdict. The corrected run had
  `recall_50=0.0`, `gt_recall_iou_0_5=0.0`,
  `gt_recall_iou_0_3=0.00298804780876494`, and 2,265 detections for 1,004 GT
  boxes. Discard this public football YOLO source as the next candidate source
  for the point regressor.
- `rfdetr-soccernet-runtime-parity` added a reusable
  `TRAIN_MODE=rfdetr_baseline` lane for `julianzu9612/RFDETR-Soccernet`.
  Current `rfdetr` and pinned `rfdetr==1.2.1` both failed before evaluation
  when `RFDETRBase` was used because the checkpoint tensors were `192/384` wide
  while base builds `128/256`-wide tensors.
- `rfdetr-soccernet-large-parity-smoke` resolved that shape blocker by loading
  the checkpoint with `RFDETRLarge`, but the scored 16-frame smoke stayed
  officially useless: `mAP-LocSim=0.0`, `precision_50=0.0`, `recall_50=0.0`,
  `gt_recall_iou_0_5=0.0`, and
  `mean_best_iou_gt_to_det=0.0012578130198192673` from 156 detections for 301
  GT boxes. Keep the large-model loader mechanics; do not spend another
  RF-DETR SoccerNet scoring pass without a new preprocessing/coordinate-parity
  hypothesis.
- `soccana-yolo11-soccernet-candidate-audit` scored
  `4.304778303917348e-05` official `mAP-LocSim` on 64 validation frames with
  public `Adit-jain/soccana` YOLO11 detections. Loaded labels were `0=Player`,
  `1=Ball`, `2=Referee`; the audit selected `0,1,2,3`, produced 1,584
  detections for 1,004 GT boxes, and still had `recall_50=0.0`,
  `gt_recall_iou_0_5=0.0`, `gt_recall_iou_0_3=0.00199203187250996`, and
  `mean_best_iou_gt_to_det=0.0018615731134729054`. Discard this public
  SoccerNet-labeled YOLO detector as the next candidate source for the point
  regressor. The first wrapper job failed before scoring because wrapper
  dependencies omitted `numpy`; Python 3.10 plus the full `train.py` dependency
  header is the working connector recipe.
- `point-regressor-jittered-candidates` scored
  `0.0006912422136296282` official `mAP-LocSim` with deterministic GT-box
  jitter (`center_frac=0.10`, `scale_frac=0.15`) on the same 64 train / 32
  valid / 2 epoch direct point regressor. It is below the pose smoke and far
  below the GT-box oracle point regressor, but the more important finding is
  coordinate-scale evidence: audit examples show GT boxes extending beyond
  x=`2900` while opened images clamp candidates at x=`1919`. Treat image file
  choice / coordinate scale as the next blocker.
- `image-path-dimension-guard` added no score yet, but `train.py` now resolves
  image paths with expected COCO `width`/`height` when a full image record is
  available and fails loudly if same-basename candidates do not match. Local
  `scripts/verify.sh` covers duplicate-basename resized/fullhd selection.
- `image-cache-scale-blocker-cloud` added no score but proved the mismatch on
  HF Jobs: job `69fac036f2f4addb7839c18b` failed before training because
  `000000.jpg` expected annotation size `3840x2160`, while both discovered
  image candidates were `1920x1080`. Treat this as the next concrete blocker,
  not a model verdict. The earlier connector launch
  `69fabfe7f2f4addb7839c185` also reconfirmed that Python 3.12 hits the known
  `xtcocotools` build-isolation failure; keep using Python 3.10.
- `actual-image-scale-jittered-point-smoke` scored
  `0.0032123790168145185` official `mAP-LocSim` after adding
  `SYNLOC_COORD_SCALE_MODE=actual_image` for the direct point-regressor lane.
  It trains/crops on actual 1920x1080 cached images and emits predictions back
  in 3840x2160 annotation coordinates for SSKit. Candidate diagnostics recovered
  to `gt_recall_iou_0_5=0.9578544061302682`,
  `gt_recall_px_50=0.9980842911877394`, and
  `mean_best_px_gt_to_pred=12.602958457397472`. This beats the prior jittered
  smoke, pose smoke, and oracle-box direct point smoke, but it is not
  challenge-submittable because candidate boxes are jittered GT. Artifact
  upload still fails with HF model-repo LFS `403`.
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
- Pairing the direct point regressor with public COCO RT-DETR person candidate
  boxes; the candidate audit stayed far below the pose smoke despite 10,006
  detections and slightly better IoU diagnostics.
- Pairing the direct point regressor with public EasyChamp or MartijnJolif
  soccer YOLO detectors; both produced many boxes but had zero IoU-0.5 GT
  recall on the SynLoc validation slice.
- Pairing the direct point regressor with public Uisikdag football YOLOv8
  detections; the corrected athlete-class audit produced many boxes but still
  had zero IoU-0.5 GT recall on the SynLoc validation slice.
- Pairing the direct point regressor with public `Adit-jain/soccana`
  detections; the SoccerNet-labeled YOLO11 audit produced many boxes but still
  had zero IoU-0.5 GT recall on the SynLoc validation slice.
- Another public detector-source audit without `SYNLOC_COORD_SCALE_MODE=actual_image`
  while using the current cached fullHD images. Prior near-zero detector overlap
  may be contaminated by coordinate scale, so real-candidate reruns must use the
  repaired coordinate path or a rebuilt true-fullHD cache.
- Running another RF-DETR SoccerNet scoring job just because the architecture
  mismatch is fixed. The large-model smoke scored zero with near-zero image-space
  overlap, so it needs a new preprocessing/coordinate-parity hypothesis first.
- Treating zero or near-zero official scores from soccer/football-pretrained
  models as model underperformance before auditing runtime/plumbing.
- Another confidence-only candidate filter or score-mode job on the same
  YOLO11n-pose smoke; the matrix already stayed far below the pose smoke.
- Any scheduler, dashboard, database, JSON state machine, or review phase that
  tries to decide what local Codex should decide from markdown.
- Leaked submissions or post-deadline winner writeups.
