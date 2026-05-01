# Council Dossier

This file is context for the external challenge council. It is intentionally richer and more opinionated than the main loop files.

## Council Role

Act as the outside research board for the autonomous SynLoc agent. Be blunt. If the current plan is dumb, say so and explain what evidence makes it dumb.

The council should give large, high-context hints and ranked strategic direction without taking over the autonomy repo. The agent still owns the loop, worktrees, budget checks, and keep/discard decisions.

## Challenge Target

- Challenge: 2026 Spiideo SoccerNet SynLoc.
- Official task: given one calibrated soccer-pitch image, predict all persons in world pitch coordinates.
- Primary score: official `mAP-LocSim`, higher is better.
- Devkit: `https://github.com/Spiideo/sskit`.
- Codabench page observed by operator: `https://codabench.org/competitions/10128/`.
- SoccerNet challenge index: `https://www.soccer-net.org/challenges/2026`.
- Submission is still available to play with until the platform closes, but the cash-prize portion may be over.

## Current Baseline Autopsy

The current full baseline is not a serious soccer model. It is only a scoring-pipeline sanity check:

- Model: TorchVision `fasterrcnn_resnet50_fpn_v2` with default COCO weights.
- Runtime so far: HF Jobs `t4-small`, `cpu-upgrade`, and `l4x1`. Future tiny SoccerMaster wiring probes should default to `t4-small`; `l4x1` needs a documented T4 memory/runtime failure or full-run reason.
- Validation split: `fullhd valid`.
- Images: 6,777.
- Detections emitted: 288,766.
- Score: `mAP-LocSim = 0.00003561507229859677`.
- Precision at LocSim 0.5 / selected threshold: `0.0035971223021582736`.
- Recall / F1 / frame accuracy: `0.0`.

The baseline keeps COCO class `person`, takes the bottom-center of each detected box, projects that image point to the ground plane via SSKit calibration, and evaluates the resulting `position_on_pitch`.

That is a lossy generic detector/projection guess. It proves the official metric path works, but it should not be treated as an architecture direction.

## Important Adjacent Context

The repo was started blank to avoid cross-project contamination, so not all useful soccer-specific context is present in this checkout.

The council should also read `RESEARCH_PRIORS.md`. It gives a compact map of the sibling workbench `/docs/architectures/` research library, current SoccerNet 2026 task direction, official SynLoc paper facts, and non-result technical priors. Do not ask the models to ingest the whole docs tree unless a narrow follow-up requires it.

There is a prior sibling SynLoc effort at `/Users/davidmontgomery/v2d-research` with useful evidence:

- Historical `soccana` soccer detector subset baseline on 128 validation frames scored `mAP-LocSim = 0.0005657708628005657`; Soccana is retired from active defaults and should not be proposed as the next run.
- Pitch-bounds filtering on that same 128-frame slice reached `mAP-LocSim = 0.0009900990099009901`, but did not generalize to 512 frames and should not be promoted blindly.
- A copied SoccerMaster GSR adapter was tried on 64 deterministic SynLoc validation frames.
- SoccerMaster attempt result: `mAP-LocSim = 0.0`; role decode produced mostly `ball=18370`, `staff=271`, `goalkeeper=131`, and no `player` detections.
- Interpretation: this is probably a runtime/config/decode failure in our copied adapter path, not a model-quality result. The SoccerMaster paper reports `92.3` athlete-detection AP@50, `50.5` mAP, and `99.2` role accuracy in Table 3, plus strong pitch-registration/camera-calibration transfer. A correct SoccerMaster runtime should not decode no players on ordinary soccer frames.
- The repaired `auto-research-2` wiring probe job `69f229c4d70108f37ace174a` loaded SoccerMaster assets and ran CUDA on 4 images. It appeared to produce `ball=1120`, `staff=80`, and `athlete_like_at_conf_0_05=0`, but diagnosis found the copied adapter used the wrong role-label order. Official SoccerMaster maps `ball=0`, `goalkeeper=1`, `other=2`, `player=3`, `referee=4`, `None=5`; the copied adapter mapped id `3` to `ball` and id `4` to `staff`. Reinterpreted, the last probe likely produced `player=1120` and `referee=80`.

Council should distinguish:

- "SoccerMaster adapter as previously wired failed or was misconfigured" from
- "soccer-specific models are useless."

Those are not the same statement.

## Likely High-Leverage Directions

The council should evaluate, rank, and criticize these directions:

- SynLoc-specific supervised detector/localizer using the provided synthetic labels.
- Predicting the annotated player ground point directly instead of deriving it from generic boxes.
- Pose/keypoint/pelvis route, if the annotation geometry and model output can be made sane.
- Better use of camera calibration, distortion, and world-coordinate loss.
- Training or fine-tuning a soccer/person detector on SynLoc images rather than using COCO person detection.
- Postprocessing only after detector/localizer recall is real; do not spend days threshold-tuning near-zero predictions.
- SoccerMaster wiring audit before any score sweep: verify official/source-faithful weights, class dimensions, role label order, image normalization/resizing, confidence thresholding, raw logits, and whether player/goalkeeper/referee outputs appear before projection.
- Using prior soccer assets only when a tiny deterministic slice shows raw athlete output and then metric lift.
- Strong diagnostics: GT matching, projection error histograms, per-camera/per-scene failure analysis, and visual overlays for a small set.

## Budget And Compute Reality

- Weekly budget is `$25/week` unless the owner approves more through a GitHub issue.
- Current estimated spend is `$20.25 / $25`.
- Do not recommend more L4 spend for import/path/config discovery. Recommend the cheapest kill test first, usually CPU for file/config checks or `t4-small` for tiny CUDA inference probes.
- Future GPU use must be evidence-gated.
- The council is allowed to recommend asking for more budget, but must say why the upside justifies it and what the kill criteria are.

## What The Council Should Produce

Do not produce generic "try better models" advice. Produce a ranked, ruthless plan.

Include:

- the top 5 mistakes the current agent is likely making,
- the top 10 next experiments ranked by expected value,
- cheap kill tests for each expensive idea,
- whether SoccerMaster or other soccer-specific assets deserve another bounded test and exactly what would have to change,
- what should be done in the next 6 hours, 24 hours, and 72 hours,
- what evidence would justify spending the remaining weekly budget,
- what evidence would justify asking the owner for more money.

## Bans

- Do not use leaked solutions, post-deadline solution writeups, or prior SoccerNet result material.
- If official 2026 results are not available yet, use no results material.
- Do not recommend leaderboard/result-page imitation.
- Do not endorse metric gaming without honest offline validation.
- Do not let the agent keep doing "promising vibe" experiments without kill criteria.
- Do not let the agent continue generic COCO detector tweaks if the evidence says it needs task-specific supervision.
