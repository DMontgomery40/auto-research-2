# Research Priors For Council

This is a compact research map for the challenge council. Do not ingest the whole sibling `/docs` tree unless a specific recommendation needs deeper detail.

## Local Research Docs Worth Knowing

The sibling workbench has a durable research library at `/Users/davidmontgomery/football_pose_workbench/docs/architectures/`.

Use these as pointers, not bulk context:

- `README.md`: high-level map of SoccerMaster, GSR, detector families, SoccerNet tasks, and research-agent harnesses.
- `gsr-and-video-to-data-pipelines.md`: SoccerNet-GSR, Broadcast-to-Minimap, SoccerNet 2025 challenge direction, FOOTPASS, SoccerNet-v3D, calibration modules, and modular pipeline traps.
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

## What Recent SoccerNet Winners And Reports Suggest

Do not mine leaked 2026 solutions. It is acceptable to learn from official published prior-year challenge reports and winner papers.

Useful lessons:

- SoccerNet 2024 GSR winner `From Broadcast to Minimap` was a modular system, not a single generic detector. It combined fine-tuned YOLOv5m detection, SegFormer-based camera parameter estimation, DeepSORT tracking, ReID, orientation prediction, and jersey-number recognition.
- SoccerNet 2025 challenge report describes GSR submissions using YOLO-X athlete detection, Deep-EIoU plus OSNet ReID, and multi-frame keypoint models for pitch coordinates.
- The same report also mentions YOLOv12 with 27 field-line categories, and a later RF-DETR detector replacement.
- This pattern matters: winners are pipeline systems with task-specific supervision, calibration, tracking/identity, and structured postprocessing. Detector swaps alone are usually not enough.

## Current Breakthroughs / Model Leads

These are leads for the council to rank, not mandates:

- SoccerMaster: soccer-specific vision foundation model with supervised multi-task pretraining over spatial perception and semantic reasoning. The paper reports `92.3` athlete-detection AP@50, `50.5` mAP, and `99.2` role accuracy in Table 3. Our copied GSR adapter failed for SynLoc as wired, but that should be treated as a likely runtime/config/decode problem. First retest should be a raw wiring audit: expected weights, class dimensions, role-label order, normalization, thresholds, and player/goalkeeper/referee logits before projection.
- RF-DETR: current detector-transformer family using DINOv2 backbone, with maintained Python package and Apache-designated open models. Potentially useful for fine-tuning on SynLoc labels, but detector-only output still needs ground-point/pelvis conversion or direct point head.
- PnLCalib and field-line calibration methods: more relevant to video/GSR and real broadcast domain adaptation than to SynLoc's already-provided camera calibration, unless calibration robustness or synthetic-to-real transfer becomes the bottleneck.
- FOOTPASS and SoccerNet-v3D: adjacent structured-soccer tasks. They are not SynLoc targets, but they reinforce that player-centric state, calibration, and identity are where the field is moving.

## Council Questions To Answer

1. Given the paper baseline table, should the agent immediately implement YOLOX-pose/MMDetection/MMPose-style training rather than more generic detector probing?
2. What is the cheapest cloud test that proves the training/eval path can approach the official paper's `mAP-LocSim` scale?
3. Is direct point regression from image to pitch coordinates better than pose/pelvis keypoint prediction for this dataset?
4. Should RF-DETR be fine-tuned as a detector only, or is that still too indirect unless paired with a point head?
5. Why did the copied SoccerMaster GSR adapter emit no player detections despite the paper's athlete-detection benchmark, and what is the smallest source-faithful runtime/config probe to isolate that?
6. What experiment should consume the remaining weekly budget, and what exact result should kill it?

## Source Links

- SoccerNet 2026 challenge page: `https://www.soccer-net.org/challenges/2026`
- SynLoc paper: `https://www.scitepress.org/Papers/2025/131082/131082.pdf`
- SSKit devkit: `https://github.com/Spiideo/sskit`
- SoccerNet 2025 challenge results: `https://arxiv.org/abs/2508.19182`
- SoccerNet 2024 challenge results: `https://arxiv.org/abs/2409.10587`
- Broadcast-to-Minimap / GSR 2024 winner paper: `https://arxiv.org/abs/2504.06357`
- FOOTPASS paper: `https://arxiv.org/abs/2511.16183`
- FOOTPASS devkit: `https://github.com/JeremieOchin/FOOTPASS`
- SoccerMaster paper: `https://arxiv.org/abs/2512.11016`
- SoccerMaster GitHub: `https://github.com/haolinyang-hlyang/SoccerMaster`
- RF-DETR GitHub: `https://github.com/roboflow/rf-detr`
- PnLCalib GitHub: `https://github.com/mguti97/PnLCalib`
