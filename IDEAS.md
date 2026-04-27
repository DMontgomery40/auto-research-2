# Ideas

Do not spend real GPU money until a tiny cloud smoke job proves the environment, credentials, SSKit install, and artifact persistence.

## Baseline and Infrastructure

- Reproduce the SSKit YOLO baseline on a tiny cloud CUDA sample.
- Build the thinnest cloud evaluation command around `sskit.coco.LocSimCOCOeval`.
- Produce a valid `results.json` plus `metadata.json` zip from cloud predictions.
- Find a reliable way to read the Codabench leaderboard score without scraping private or leaked content.

## Model Directions

- Strong detector plus calibrated ground projection from player foot/bottom-center point.
- Keypoint/pelvis prediction route using `position_from_keypoint_index`.
- Synthetic-to-real augmentation sweep: blur, compression, exposure, field color, player scale, camera crop.
- Detector confidence calibration to improve threshold-selected F1 without harming mAP-LocSim.
- Ensemble only if two independent models produce complementary localization errors and cost is justified.

## Soccer-Specific Prior

- SoccerMaster exists in the sibling workbench/research context and is an optional lead, not a mandate.
- First useful question: can SoccerMaster produce player/person-like detections or role outputs on SynLoc images that can be converted into `position_on_pitch` predictions and beat the TorchVision baseline?
- Sibling evidence from `/Users/davidmontgomery/v2d-research`: scratch run `synloc-20260426-1308` tested the copied SoccerMaster GSR adapter on 64 deterministic SynLoc validation frames with official camera projection and official LocSim eval. All 54 confidence/role/pitch-bound rows scored `mAP-LocSim=0.0`; role decode produced mostly `ball=18370`, `staff=271`, `goalkeeper=131`, and no `player` detections.
- Do not scale that copied GSR adapter as-is. Retest SoccerMaster only if the experiment changes the decode/head/postprocess, uses it for pitch/keypoint calibration, or otherwise has a specific reason it could produce athlete-like world positions.

## Risky Ideas

- Full custom pose model training before baseline is stable.
- Large multi-GPU sweeps without cheap proxy signal.
- Any solution inspired by post-deadline winner writeups or leaked submissions.
