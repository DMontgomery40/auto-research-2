# Budget

Weekly compute budget: `$50/week`.

Owner approved a `$25` reload on 2026-05-03 after confirming Hugging Face billing showed available funds. Do not exceed the refreshed `$50` cap without another owner approval.

Week starts: Monday.

| Date | Tag | Provider | Hardware | Est. Cost | Actual Cost | Purpose | Result |
|---|---|---|---|---:|---:|---|---|
| 2026-04-26 | setup | local | CPU | $0.00 | $0.00 | Control-plane repo setup | Done |
| 2026-04-26 | cloud-smoke attempts | HF Jobs | T4 small | $0.75 | pending | Verify GPU, imports, and private HF write | Passed on third attempt |
| 2026-04-26 | dataset-cache-valid attempts | HF Jobs | CPU upgrade | $2.00 | pending | Cache fullhd validation split in private HF dataset repo | Passed on second attempt |
| 2026-04-27 | baseline-probe | HF Jobs | T4 small | $1.50 | pending | Small YOLO validation probe | Failed before execution; patched and retrying |
| 2026-04-27 | baseline-probe retry | HF Jobs | T4 small | $1.50 | pending | Small YOLO validation probe | Failed on NumPy ABI; pinned `numpy<2` and retrying |
| 2026-04-27 | baseline-probe retry | HF Jobs | T4 small | $1.50 | pending | Small validation probe | Failed on `libGL`; switched to TorchVision detector and retrying |
| 2026-04-27 | baseline-probe retry | HF Jobs | T4 small | $1.50 | pending | Small validation probe | Failed on missing `scipy`; added dependency and retrying |
| 2026-04-27 | baseline-probe retry | HF Jobs | T4 small | $1.50 | pending | Small validation probe | Completed; `mAP-LocSim=0.0001237624` on 64 images |
| 2026-04-27 | baseline-full | HF Jobs | L4 x1 | $6.00 | pending | Full validation baseline | Completed; `mAP-LocSim=0.0000356151` on 6,777 images |
| 2026-04-27 | soccermaster-wiring-probe | HF Jobs | L4 x1 | $2.00 | pending | Check SoccerMaster raw athlete output | Failed before inference on wrong asset path |
| 2026-04-29 | soccermaster-wiring-probe retry | HF Jobs | L4 x1 | $2.00 | pending | Check SoccerMaster raw athlete output after asset repair | Completed; zero athlete roles, config/runtime/decode mismatch |
| 2026-04-29 | soccermaster-wiring-probe corrected roles | HF Jobs | T4 small | $0.50 | pending | Confirm official SoccerMaster role-label decode | Completed; `player=1196`, `referee=4`, bug confirmed |
| 2026-04-29 | soccermaster-synloc-eval-probe | HF Jobs | T4 small | $1.00 | pending | Convert SoccerMaster detections to SynLoc and score 64 images | Completed; best `mAP-LocSim=0.0000073739` |
| 2026-04-29 | pretrained-yolo-baseline | HF Jobs | T4 small | $0.75 | pending | Historical baseline-eval of pretrained football YOLO26 plus the now-retired Soccana path through `train.py` before training | Completed; the now-retired Soccana row scored `mAP-LocSim=0.0000574073`; training blocked |
| 2026-04-29 | synloc-devkit-oracle | HF Jobs | CPU upgrade | $0.25 | pending | Run SSKit oracle checks before any more model training | Failed before execution on missing `torchvision`; dependency patched |
| 2026-04-29 | synloc-devkit-oracle retry | HF Jobs | CPU upgrade | $0.25 | pending | Run SSKit oracle checks before any more model training | Completed; exact GT `1.0`, projected GT keypoint `0.9809895759`, bbox bottom-center via SSKit `0.5686594909` |
| 2026-05-01 | football-yolo26-diagnostic | HF Jobs | t4-small | $0.75 | pending | Submitted by autonomy heartbeat | Completed; job `69f4facf9d85bec4d76efcb0` scored `mAP-LocSim=0.0000467028` with image-space GT recall still near zero. |
| 2026-05-02 | synloc-pose-smoke | HF Jobs | t4-small | $1.25 | pending | Submitted by autonomy heartbeat | Completed; job `69f655699d85bec4d76f0adb` scored `mAP-LocSim=0.0008250825`, non-promotable validation-slice smoke. |

Current estimated spend: `$25.00 / $50.00`.

Future tiny SoccerMaster probes should use HF Jobs `t4-small` with tight timeouts, not `l4x1`, unless T4 fails for memory/runtime reasons that are recorded in the ledger.

Recent autonomous spend: `football-yolo26-diagnostic` and `synloc-pose-smoke` completed the original `$25` budget. The owner approved a `$25` reload so the next cache job can proceed under a refreshed `$50` cap.

Next planned paid step: `dataset-cache-train-valid` on HF Jobs `cpu-upgrade`, estimated `$1.00`, to cache the train split beside validation before a real source-specific pose/keypoint experiment.

Current blocker: none. Issue #10 contains the owner approval for the `$25` reload.

Local Codex research ticks and the local `scripts/codex_research_loop.sh` cost `$0.00` in this compute ledger. They may inspect sources, patch code, update issues, and improve the next experiment plan, but they must not launch paid HF Jobs while the budget gate is closed.

## Spend Rule

If an experiment could push the week over `$25`, open a GitHub issue before running it.

Issue should include:

- experiment name,
- why it might beat current best,
- estimated cost,
- stop condition,
- fallback if it fails.
