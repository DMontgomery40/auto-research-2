# Research Priors For Council

This is a compact research map for the challenge council. Do not ingest the whole sibling `/docs` tree unless a specific recommendation needs deeper detail.

## Local Research Docs Worth Knowing

The sibling workbench has a durable research library at `/Users/davidmontgomery/football_pose_workbench/docs/architectures/`.

Use these as pointers, not bulk context:

- `README.md`: high-level map of SoccerMaster, GSR, detector families, SoccerNet tasks, and research-agent harnesses.
- `gsr-and-video-to-data-pipelines.md`: SoccerNet-GSR, FOOTPASS, SoccerNet-v3D, calibration modules, and modular pipeline traps. Do not import its prior-year result summaries into council requests.
- `soccer-foundation-models.md`: SoccerMaster, MatchVision/UniSoccer, SoccerChat, SoccerAgent, and why dense spatial soccer models differ from generic VLMs.
- `detectors-yolo-and-small-object-models.md`: YOLO11/YOLO26, RF-DETR/RT-DETR, SoccerDETR, Football-YOLO, Soccana, ball detection, jersey-number recognition, and detector-only traps.
- `monthly/2026-04.md`: live April 2026 notes, including FOOTPASS contract, RF-DETR soccer checkpoint scan, SoccerNet VQA 2026 details, and calibration watch items.
- `experiment-harness-map.md`: how research agents should compare modular video-to-data recipes, and why detector mAP alone is not enough for game-state tasks.

## Direct SynLoc Facts That Should Drive This Repo

Primary source: `Spiideo SoccerNet SynLoc: Single Frame World Coordinate Athlete Detection and Localization with Synthetic Data` and the official SSKit repo.

Critical facts:

- SynLoc labels include camera calibration, athlete image boxes, pelvis keypoints, pitch-location keypoints, and world-space ground truth.
- Dataset split reported in the paper: 42,504 train images, 6,777 validation images, 9,309 test images, 11,352 challenge images.
- Official evaluation is world-coordinate `mAP-LocSim`, not image-box IoU.
- The paper uses LocSim parameter `tau=1m`; the paper also states the practical LocSim 0.5 threshold corresponds to about 0.48m distance.
- Official baseline evidence says pose/keypoint localization is dramatically better than bottom-of-box projection.
- Paper table: YOLOX bbox at 960x960 reaches roughly 50.1 to 52.4 `mAP-LocSim`, while YOLOX pose at 960x960 reaches roughly 72.6 to 79.3 `mAP-LocSim`.
- Therefore: tuning generic COCO boxes is a dead end. The highest-value direction is task-specific supervised training that predicts the correct ground/pelvis point.

## SoccerNet 2026 Context

Official SoccerNet 2026 tasks observed:

- Spiideo SoccerNet SynLoc: single-frame world-coordinate athlete localization, scored with `mAP-LocSim`.
- Ball Action Anticipation: anticipate action timing/type in the next 5 seconds from preceding 30-second clips.
- VQA: close-ended multimodal soccer QA over text/image/video with 14 task types.
- Player-Centric Ball Action Spotting / FOOTPASS: identify what happened, when, and who performed the action.
- Novel View Synthesis: generate views from sparse soccer broadcast observations.
- FIFA Innovation Challenge - Skeletal Tracking Light: pose estimation from main broadcast camera.

Council implication:

- These tasks are converging on player identity, pitch geometry, temporal context, and structured state.
- SynLoc is the cleanest single-frame piece of that stack. It should be treated as supervised world-coordinate athlete localization, not generic object detection.

## Results Boundary

Do not use prior SoccerNet challenge result material.

If 2026 official results are not available yet, include no results material. The council may use official 2026 challenge pages, devkits, dataset papers, model papers, and our own experiment ledger.

## Current Breakthroughs / Model Leads

These are leads for the council to rank, not mandates:

- SoccerMaster: soccer-specific vision foundation model with supervised multi-task pretraining over spatial perception and semantic reasoning. The paper reports `92.3` athlete-detection AP@50, `50.5` mAP, and `99.2` role accuracy in Table 3. The repaired `auto-research-2` wiring probe initially looked like zero athlete roles, but diagnosis found the copied Rondo adapter used the wrong role-label order. Official SoccerMaster maps `ball=0`, `goalkeeper=1`, `other=2`, `player=3`, `referee=4`, `None=5`; the copied adapter mapped id `3` to `ball` and id `4` to `staff`. The corrected conversion/eval probe then scored worse than TorchVision, which should be read as a copied-runtime parity failure: the adapter is not yet the official video/temporal SoccerMaster runtime, does not use the official Deformable DETR/MSDeformAttn path, and does not use official postprocess.
- Pretrained YOLO baseline first: before any fine-tuning, `train.py` now evaluates current football YOLO26 and Soccana/SoccerNet-style detector weights on SynLoc validation through official `mAP-LocSim`. The first run produced thousands of detections but only `0.0000574073` best `mAP-LocSim` with `recall_50=0.0`, so the loop must debug model loading/class mapping/projection/eval plumbing before training.
- RF-DETR: current detector-transformer family using DINOv2 backbone, with maintained Python package and Apache-designated open models. Potentially useful for fine-tuning on SynLoc labels, but detector-only output still needs ground-point/pelvis conversion or direct point head.
- PnLCalib and field-line calibration methods: more relevant to video/GSR and real broadcast domain adaptation than to SynLoc's already-provided camera calibration, unless calibration robustness or synthetic-to-real transfer becomes the bottleneck.
- FOOTPASS and SoccerNet-v3D: adjacent structured-soccer tasks. They are not SynLoc targets, but they reinforce that player-centric state, calibration, and identity are where the field is moving.

## Council Questions To Answer

1. Given the paper baseline table, should the agent immediately implement YOLOX-pose/MMDetection/MMPose-style training rather than more generic detector probing?
2. What is the cheapest cloud test that proves the training/eval path can approach the official paper's `mAP-LocSim` scale?
3. Is direct point regression from image to pitch coordinates better than pose/pelvis keypoint prediction for this dataset?
4. Should RF-DETR be fine-tuned as a detector only, or is that still too indirect unless paired with a point head?
5. What is the smallest source-faithful SoccerMaster runtime parity probe that proves whether official code/config/postprocess produce sane boxes on SynLoc frames before any train/fine-tune job?
6. What experiment should consume the remaining weekly budget, and what exact result should kill it?

## Source Links

- SoccerNet 2026 challenge page: `https://www.soccer-net.org/challenges/2026`
- SynLoc paper: `https://www.scitepress.org/Papers/2025/131082/131082.pdf`
- SSKit devkit: `https://github.com/Spiideo/sskit`
- FOOTPASS paper: `https://arxiv.org/abs/2511.16183`
- FOOTPASS devkit: `https://github.com/JeremieOchin/FOOTPASS`
- SoccerMaster paper: `https://arxiv.org/abs/2512.11016`
- SoccerMaster GitHub: `https://github.com/haolinyang-hlyang/SoccerMaster`
- RF-DETR GitHub: `https://github.com/roboflow/rf-detr`
- PnLCalib GitHub: `https://github.com/mguti97/PnLCalib`
