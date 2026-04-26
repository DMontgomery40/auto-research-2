# Ideas

Do not spend GPU money until baseline and evaluator are real.

## Baseline and Infrastructure

- Reproduce the SSKit YOLO baseline on a tiny sample.
- Build the thinnest local evaluation command around `sskit.coco.LocSimCOCOeval`.
- Produce a valid `results.json` plus `metadata.json` zip for local validation.
- Find a reliable way to read the Codabench leaderboard score without scraping private or leaked content.

## Model Directions

- Strong detector plus calibrated ground projection from player foot/bottom-center point.
- Keypoint/pelvis prediction route using `position_from_keypoint_index`.
- Synthetic-to-real augmentation sweep: blur, compression, exposure, field color, player scale, camera crop.
- Detector confidence calibration to improve threshold-selected F1 without harming mAP-LocSim.
- Ensemble only if two independent models produce complementary localization errors and cost is justified.

## Risky Ideas

- Full custom pose model training before baseline is stable.
- Large multi-GPU sweeps without cheap proxy signal.
- Any solution inspired by post-deadline winner writeups or leaked submissions.
