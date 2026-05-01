# Codex Goal

<!-- codex-goal:start -->
Beat the 2026 Spiideo SoccerNet SynLoc challenge by June 30 using a tiny AK-style loop. Start from official/dev-kit paths, not hand-rolled detector scoring. Current verified baseline: SSKit oracle passed (exact GT mAP-LocSim 1.0, projected GT keypoint 0.9809895759, bbox bottom-center via SSKit 0.5686594909). Do not train from the near-zero detector path until a source-faithful official/dev-kit detector baseline produces meaningful recall and mAP. Soccana is retired from active defaults. Work in one isolated experiment worktree, record score/cost/decision in LEDGER.md, update CURRENT.md, and keep the repo markdown-first.
<!-- codex-goal:end -->

Use this text as the Codex app thread Goal for local worktree experiment sessions when the Goals feature is available. Keep `CURRENT.md`, `LEDGER.md`, `BUDGET.md`, `autonomy/state.json`, `autonomy/events.jsonl`, and GitHub issues current because those files are the durable control plane that GitHub Actions and HF Jobs can read.
