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

- SoccerMaster exists in the sibling workbench/research context and is a serious soccer-specific lead, not a mandate.
- The SoccerMaster paper reports strong spatial benchmarks: Table 3 has `92.3` athlete-detection AP@50, `50.5` mAP, and `99.2` role accuracy for SoccerMaster with pipeline data, far beyond generic vision baselines. A zero SynLoc score is therefore a runtime/config/decode warning before it is a model-quality signal.
- First useful question: can the code-faithful SoccerMaster runtime load the expected backbone/head weights, decode the expected role labels, and produce player/goalkeeper/referee outputs on SynLoc images before any projection or metric conversion?
- Sibling evidence from `/Users/davidmontgomery/v2d-research`: scratch run `synloc-20260426-1308` tested the copied SoccerMaster GSR adapter on 64 deterministic SynLoc validation frames with official camera projection and official LocSim eval. All 54 confidence/role/pitch-bound rows scored `mAP-LocSim=0.0`; role decode produced mostly `ball=18370`, `staff=271`, `goalkeeper=131`, and no `player` detections.
- Auto-research-2 evidence from job `69f229c4d70108f37ace174a`: after fixing the asset path, the probe loaded SoccerMaster weights and ran CUDA on 4 images, but emitted `ball=1120`, `staff=80`, and `athlete_like_at_conf_0_05=0`.
- Treat the zero-athlete result as evidence that our adapter/config/postprocess path is wrong or incomplete. The next bounded test is not a score sweep; it is a source-faithful config audit against the official SoccerMaster repo: inspect raw detection logits, role logits, class dimensions, role mapping, thresholds, image normalization, and weight placement before deciding whether SoccerMaster can or cannot help SynLoc.
- Run future tiny SoccerMaster probes on HF Jobs `t4-small` with tight timeouts. Escalate to `l4x1` only after a recorded T4 memory/runtime failure.

## Risky Ideas

- Full custom pose model training before baseline is stable.
- Large multi-GPU sweeps without cheap proxy signal.
- Any solution inspired by post-deadline winner writeups or leaked submissions.
