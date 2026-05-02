#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "autonomy" / "state.json"
EVENTS_PATH = ROOT / "autonomy" / "events.jsonl"
GOAL_PATH = ROOT / "GOAL.md"
CURRENT_PATH = ROOT / "CURRENT.md"
LEDGER_PATH = ROOT / "LEDGER.md"
BUDGET_PATH = ROOT / "BUDGET.md"
JOURNAL_PATH = ROOT / "JOURNAL.md"
COUNCIL_INBOX = Path("challenge-council") / "data" / "automation_queue" / "inbox"

TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"ERROR", "CANCELED", "DELETED"}
PRETRAINED_YOLO_BASELINE_MIN_MAP_LOCSIM = 0.01
DEVKIT_ORACLE_MIN_EXACT_MAP = 0.99
DEVKIT_ORACLE_MIN_KEYPOINT_MAP = 0.90
DEVKIT_ORACLE_MIN_BBOX_MAP = 0.10

JOB_SPECS: dict[str, dict[str, Any]] = {
    "cloud_smoke_pending": {
        "label": "cloud-smoke",
        "script": "cloud/synloc_smoke.py",
        "flavor": "t4-small",
        "timeout": "30m",
        "python": "3.10",
        "cost_estimate_usd": 0.25,
        "next_phase": "dataset_cache_valid_pending",
        "env": {},
    },
    "dataset_cache_valid_pending": {
        "label": "dataset-cache-valid",
        "script": "cloud/synloc_cache.py",
        "flavor": "cpu-upgrade",
        "timeout": "6h",
        "python": "3.10",
        "required_secrets": ["SOCCERNET_USERNAME", "SOCCERNET_PASSWORD"],
        "required_secret_groups": [["SOCCERNET_SIGNIN_PASSWORD", "SOCCERNET_PASSWORD_2", "SPIIDEO_PASSWORD", "SOCCERNET_PASSWORD"]],
        "cost_estimate_usd": 1.0,
        "next_phase": "baseline_probe_pending",
        "env": {"SYNLOC_SPLITS": "valid", "SYNLOC_VERSION": "fullhd"},
    },
    "baseline_probe_pending": {
        "label": "baseline-probe",
        "script": "cloud/synloc_baseline_yolo.py",
        "flavor": "t4-small",
        "timeout": "2h",
        "python": "3.10",
        "required_secrets": [],
        "required_secret_groups": [],
        "cost_estimate_usd": 1.5,
        "next_phase": "baseline_full_pending",
        "env": {"SYNLOC_SPLIT": "valid", "SYNLOC_VERSION": "fullhd", "BASELINE_MAX_IMAGES": "64"},
    },
    "baseline_full_pending": {
        "label": "baseline-full",
        "script": "cloud/synloc_baseline_yolo.py",
        "flavor": "l4x1",
        "timeout": "6h",
        "python": "3.10",
        "required_secrets": [],
        "required_secret_groups": [],
        "cost_estimate_usd": 6.0,
        "next_phase": "council_after_baseline_pending",
        "env": {"SYNLOC_SPLIT": "valid", "SYNLOC_VERSION": "fullhd", "BASELINE_MAX_IMAGES": "0"},
    },
    "soccermaster_wiring_probe_pending": {
        "label": "soccermaster-wiring-probe",
        "script": "cloud/soccermaster_wiring_probe.py",
        "flavor": "t4-small",
        "timeout": "1h",
        "python": "3.10",
        "required_secrets": [],
        "required_secret_groups": [],
        "cost_estimate_usd": 0.5,
        "next_phase": "soccermaster_synloc_conversion_probe_pending",
        "env": {
            "V2D_ASSET_REPO": "dmontgomery40/v2d-research-assets",
            "SOCCERMASTER_MAX_IMAGES": "4",
        },
    },
    "soccermaster_synloc_conversion_probe_pending": {
        "label": "soccermaster-synloc-eval-probe",
        "script": "cloud/soccermaster_synloc_eval_probe.py",
        "flavor": "t4-small",
        "timeout": "2h",
        "python": "3.10",
        "required_secrets": [],
        "required_secret_groups": [],
        "cost_estimate_usd": 1.0,
        "next_phase": "first_train_experiment_pending",
        "env": {
            "V2D_ASSET_REPO": "dmontgomery40/v2d-research-assets",
            "SYNLOC_SPLIT": "valid",
            "SYNLOC_VERSION": "fullhd",
            "SOCCERMASTER_EVAL_MAX_IMAGES": "64",
            "SOCCERMASTER_THRESHOLDS": "0.01,0.03,0.05,0.1,0.2,0.3",
        },
    },
    "pretrained_yolo_baseline_pending": {
        "label": "pretrained-yolo-baseline",
        "script": "train.py",
        "flavor": "t4-small",
        "timeout": "2h",
        "python": "3.10",
        "required_secrets": [],
        "required_secret_groups": [],
        "cost_estimate_usd": 0.75,
        "next_phase": "train_dataset_cache_pending",
        "env": {
            "TRAIN_MODE": "baseline",
            "SYNLOC_SPLIT": "valid",
            "SYNLOC_VERSION": "fullhd",
            "TRAIN_MAX_IMAGES": "128",
            "YOLO_IMGSZ": "960",
            "YOLO_CONF": "0.01",
            "YOLO_IOU": "0.7",
        },
    },
    "devkit_detector_diagnostic_pending": {
        "label": "football-yolo26-diagnostic",
        "script": "train.py",
        "flavor": "t4-small",
        "timeout": "2h",
        "python": "3.10",
        "required_secrets": [],
        "required_secret_groups": [],
        "cost_estimate_usd": 0.75,
        "next_phase": "devkit_detector_diagnostic_review",
        "env": {
            "TRAIN_MODE": "baseline",
            "SYNLOC_SPLIT": "valid",
            "SYNLOC_VERSION": "fullhd",
            "TRAIN_MAX_IMAGES": "128",
            "YOLO_IMGSZ": "960",
            "YOLO_CONF": "0.01",
            "YOLO_IOU": "0.7",
            "YOLO_BASELINES": "football-yolo26l|mobadam/football-player-detection|player_detector.pt|1,3",
        },
    },
    "synloc_pose_smoke_pending": {
        "label": "synloc-pose-smoke",
        "script": "cloud/synloc_pose_smoke.py",
        "flavor": "t4-small",
        "timeout": "2h",
        "python": "3.10",
        "required_secrets": [],
        "required_secret_groups": [],
        "cost_estimate_usd": 1.25,
        "next_phase": "pose_smoke_review",
        "env": {
            "SYNLOC_SPLIT": "valid",
            "SYNLOC_VERSION": "fullhd",
            "POSE_TRAIN_MAX_IMAGES": "64",
            "POSE_VAL_START": "64",
            "POSE_VAL_MAX_IMAGES": "64",
            "YOLO_IMGSZ": "640",
            "YOLO_EPOCHS": "1",
            "YOLO_BATCH": "4",
            "YOLO_CONF": "0.01",
        },
    },
    "devkit_oracle_pending": {
        "label": "synloc-devkit-oracle",
        "script": "cloud/synloc_devkit_oracle.py",
        "flavor": "cpu-upgrade",
        "timeout": "1h",
        "python": "3.10",
        "required_secrets": [],
        "required_secret_groups": [],
        "cost_estimate_usd": 0.25,
        "next_phase": "devkit_oracle_review",
        "env": {
            "SYNLOC_SPLIT": "valid",
            "SYNLOC_VERSION": "fullhd",
            "DEVKIT_ORACLE_MAX_IMAGES": "64",
        },
    },
    "train_dataset_cache_pending": {
        "label": "dataset-cache-train-valid",
        "script": "cloud/synloc_cache.py",
        "flavor": "cpu-upgrade",
        "timeout": "8h",
        "python": "3.10",
        "required_secrets": ["SOCCERNET_USERNAME", "SOCCERNET_PASSWORD"],
        "required_secret_groups": [["SOCCERNET_SIGNIN_PASSWORD", "SOCCERNET_PASSWORD_2", "SPIIDEO_PASSWORD", "SOCCERNET_PASSWORD"]],
        "cost_estimate_usd": 1.0,
        "next_phase": "first_train_experiment_pending",
        "env": {"SYNLOC_SPLITS": "train,valid", "SYNLOC_VERSION": "fullhd"},
    },
    "first_train_experiment_pending": {
        "label": "first-yolo-train",
        "script": "train.py",
        "flavor": "t4-small",
        "timeout": "3h",
        "python": "3.10",
        "required_secrets": [],
        "required_secret_groups": [],
        "cost_estimate_usd": 1.5,
        "next_phase": "train_result_review",
        "env": {
            "TRAIN_MODE": "finetune",
            "SYNLOC_VERSION": "fullhd",
            "TRAIN_MAX_IMAGES": "2048",
            "VAL_MAX_IMAGES": "512",
            "YOLO_IMGSZ": "960",
            "YOLO_EPOCHS": "3",
            "YOLO_BATCH": "4",
            "YOLO_CONF": "0.01",
            "YOLO_IOU": "0.7",
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def env_any(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    STATE_PATH.write_text(json.dumps(json_safe(state), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_event(event: str, **payload: Any) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe({"ts": utc_now(), "event": event, **payload}), sort_keys=True, allow_nan=False) + "\n")


def append_journal(message: str) -> None:
    if not JOURNAL_PATH.exists():
        JOURNAL_PATH.write_text("# Journal\n\n", encoding="utf-8")
    with JOURNAL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"- {utc_now()} - {message}\n")


def markdown_cell(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("\n", " ").replace("|", "\\|")


def score_from_result(result: dict[str, Any]) -> tuple[float | None, float | None]:
    best = result.get("best", {}) if isinstance(result.get("best"), dict) else {}
    metrics = best.get("metrics", {}) if isinstance(best.get("metrics"), dict) else {}
    if not metrics and isinstance(result.get("metrics"), dict):
        metrics = result["metrics"]
    score = metrics.get("map_locsim")
    threshold = metrics.get("score_threshold") or best.get("threshold")
    return (
        float(score) if score is not None else None,
        float(threshold) if threshold is not None else None,
    )


def result_dataset(result: dict[str, Any]) -> str:
    split = result.get("split") or result.get("best", {}).get("split") if isinstance(result.get("best"), dict) else result.get("split")
    version = result.get("version") or result.get("best", {}).get("version") if isinstance(result.get("best"), dict) else result.get("version")
    max_images = result.get("max_images") or result.get("num_images") or result.get("best", {}).get("max_images") if isinstance(result.get("best"), dict) else result.get("max_images")
    bits = [str(item) for item in (version, split) if item]
    if max_images:
        bits.append(f"{max_images} images")
    return " ".join(bits) or "unknown"


def append_budget_submission(active: dict[str, Any], spec: dict[str, Any]) -> None:
    text = BUDGET_PATH.read_text(encoding="utf-8") if BUDGET_PATH.exists() else "# Budget\n\n"
    if active.get("id") and active["id"] in text:
        return
    row = (
        f"| {utc_now()[:10]} | {markdown_cell(active.get('label'))} | HF Jobs | {markdown_cell(spec.get('flavor'))} | "
        f"${float(active.get('cost_estimate_usd') or 0):.2f} | pending | Submitted by autonomy heartbeat | "
        f"Job `{markdown_cell(active.get('id'))}` running; {markdown_cell(active.get('url'))} |\n"
    )
    BUDGET_PATH.write_text(text.rstrip() + "\n" + row, encoding="utf-8")


def append_ledger_completion(active: dict[str, Any], result: dict[str, Any]) -> None:
    text = LEDGER_PATH.read_text(encoding="utf-8") if LEDGER_PATH.exists() else "# Experiment Ledger\n\n"
    if active.get("id") and active["id"] in text:
        return
    code, commit = run(["git", "rev-parse", "--short", "HEAD"])
    commit = commit.strip() if code == 0 else "unknown"
    score, threshold = score_from_result(result)
    best = result.get("best", {}) if isinstance(result.get("best"), dict) else {}
    diagnostics = best.get("diagnostics", {}) if isinstance(best.get("diagnostics"), dict) else result.get("diagnostics", {})
    diagnostic_bits = []
    if isinstance(diagnostics, dict):
        for key in ("gt_recall_iou_0_3", "gt_recall_iou_0_5", "mean_best_iou_gt_to_det"):
            if key in diagnostics:
                diagnostic_bits.append(f"{key}={diagnostics[key]}")
    notes = f"Job `{active.get('id')}`; {active.get('url')}"
    if diagnostic_bits:
        notes += "; image-space " + ", ".join(diagnostic_bits)
    row = (
        f"| {utc_now()[:10]} | {markdown_cell(active.get('label'))} | {markdown_cell(commit)} | "
        f"{markdown_cell(result_dataset(result))} | HF Jobs `{markdown_cell(active.get('flavor'))}` | "
        f"`{markdown_cell(active.get('label'))}` | {score if score is not None else 0:.10f} | "
        f"{threshold if threshold is not None else 0:.6f} | ${float(active.get('cost_estimate_usd') or 0):.2f} est | n/a | review | "
        f"{markdown_cell(notes)} |\n"
    )
    LEDGER_PATH.write_text(text.rstrip() + "\n" + row, encoding="utf-8")


def update_current_snapshot(state: dict[str, Any]) -> None:
    if not CURRENT_PATH.exists():
        return
    start = "<!-- autonomy-snapshot:start -->"
    end = "<!-- autonomy-snapshot:end -->"
    active = state.get("active_job") or {}
    blocker = state.get("blocker") or {}
    last = state.get("history", [])[-1] if state.get("history") else {}
    last_result = last.get("result", {}) if isinstance(last.get("result"), dict) else {}
    score, threshold = score_from_result(last_result)
    lines = [
        start,
        "## Autonomy Snapshot",
        "",
        f"- Updated: {utc_now()}",
        f"- Phase: `{state.get('phase')}`",
        f"- Active job: `{active.get('id', 'none')}` {active.get('url', '')}".rstrip(),
        f"- Spend estimate: `${float(state.get('spent_estimate_usd', 0.0)):.2f} / ${float(state.get('weekly_budget_usd', 25.0)):.2f}`",
        f"- Blocker: {blocker.get('title', 'none')}",
        f"- Last result: `{last.get('label', 'none')}` `{last.get('id', 'none')}` score `{score if score is not None else 'n/a'}` threshold `{threshold if threshold is not None else 'n/a'}`",
        end,
    ]
    text = CURRENT_PATH.read_text(encoding="utf-8")
    section = "\n".join(lines) + "\n"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + section + after.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + section
    CURRENT_PATH.write_text(text, encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr)[-8000:]


def post_issue(token: str, repo: str, payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=encoded,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def create_issue(title: str, body: str) -> None:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("GH_PAT")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        append_event("issue_not_created", title=title, reason="missing GitHub token or repo")
        return
    payload = {"title": title, "body": body, "labels": ["autonomy", "needs-owner"]}
    try:
        response = post_issue(token, repo, payload)
        append_event("issue_created", title=title, response=response.decode("utf-8")[:1000])
    except Exception as exc:
        try:
            response = post_issue(token, repo, {"title": title, "body": body})
            append_event("issue_created_without_labels", title=title, response=response.decode("utf-8")[:1000])
        except Exception as retry_exc:
            append_event("issue_create_failed", title=title, error=repr(exc), retry_error=repr(retry_exc))


def council_queue_candidates() -> list[Path]:
    paths: list[Path] = []
    if os.getenv("CHALLENGE_COUNCIL_INBOX"):
        paths.append(Path(os.environ["CHALLENGE_COUNCIL_INBOX"]).expanduser())
    paths.append(ROOT.parent / COUNCIL_INBOX)
    if len(ROOT.parents) > 1:
        paths.append(ROOT.parents[1] / COUNCIL_INBOX)
    paths.append(Path.home() / COUNCIL_INBOX)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def existing_council_queue() -> Path | None:
    for candidate in council_queue_candidates():
        if candidate.exists():
            return candidate
    return None


def latest_result(state: dict[str, Any], label: str) -> dict[str, Any] | None:
    for item in reversed(state.get("history", [])):
        if item.get("label") == label and isinstance(item.get("result"), dict):
            return item["result"]
    return None


def case_map(result: dict[str, Any], name: str) -> float:
    for case in result.get("cases", []):
        if case.get("name") == name:
            metrics = case.get("metrics", {}) if isinstance(case.get("metrics"), dict) else {}
            return float(metrics.get("map_locsim") or 0.0)
    return 0.0


def codex_goal_text() -> str:
    fallback = (
        "Beat the 2026 Spiideo SoccerNet SynLoc challenge by June 30 using a tiny AK-style loop. "
        "Start from official/dev-kit paths, not hand-rolled detector scoring. Work in one isolated experiment worktree, "
        "record score/cost/decision in LEDGER.md, update CURRENT.md, and keep the repo markdown-first."
    )
    if not GOAL_PATH.exists():
        return fallback
    raw_lines = GOAL_PATH.read_text(encoding="utf-8").splitlines()
    if "<!-- codex-goal:start -->" in raw_lines and "<!-- codex-goal:end -->" in raw_lines:
        start = raw_lines.index("<!-- codex-goal:start -->") + 1
        end = raw_lines.index("<!-- codex-goal:end -->")
        raw_lines = raw_lines[start:end]
    lines = [line.strip() for line in raw_lines if line.strip() and not line.startswith("#")]
    return " ".join(lines) or fallback


def handle_devkit_oracle_review(state: dict[str, Any], *, dry_run: bool) -> None:
    result = latest_result(state, "synloc-devkit-oracle")
    if not result:
        state["phase"] = "blocked_devkit_oracle_missing_result"
        append_event("devkit_oracle_review_missing_result")
        create_issue(
            "Autonomy blocked: devkit oracle result missing",
            "The controller reached `devkit_oracle_review`, but no `synloc-devkit-oracle` result exists in `autonomy/state.json` history.",
        )
        return

    exact = case_map(result, "exact_position_on_pitch")
    keypoint = case_map(result, "gt_projected_ground_keypoint")
    bbox = max(
        case_map(result, "gt_bbox_bottom_center_keypoint"),
        case_map(result, "gt_bbox_bottom_center_devkit_eval"),
    )
    passed = (
        result.get("ok", False)
        and exact >= DEVKIT_ORACLE_MIN_EXACT_MAP
        and keypoint >= DEVKIT_ORACLE_MIN_KEYPOINT_MAP
        and bbox >= DEVKIT_ORACLE_MIN_BBOX_MAP
    )
    append_event(
        "devkit_oracle_review",
        passed=passed,
        exact_map=exact,
        keypoint_map=keypoint,
        bbox_map=bbox,
    )
    if dry_run:
        return
    if not passed:
        state["phase"] = "blocked_devkit_oracle_failed"
        state["blocker"] = {
            "title": "SSKit devkit oracle failed review gates",
            "created_at": utc_now(),
            "exact_map": exact,
            "keypoint_map": keypoint,
            "bbox_map": bbox,
        }
        create_issue(
            "Autonomy blocked: SSKit devkit oracle failed review gates",
            (
                "The dev-kit oracle did not clear the review gates, so model training remains blocked.\n\n"
                f"- exact_position_on_pitch: `{exact}`\n"
                f"- gt_projected_ground_keypoint: `{keypoint}`\n"
                f"- bbox bottom-center via SSKit: `{bbox}`\n\n"
                "Fix the official data/camera/evaluator path before any detector or training experiment."
            ),
        )
        return

    state["phase"] = "devkit_detector_diagnostic_pending"
    state.pop("blocker", None)
    append_journal(
        "SSKit oracle passed; auto-resuming into football-yolo26-diagnostic instead of parking at blocked_next_worktree_needed."
    )
    create_issue(
        "Autonomy continuing: dev-kit detector diagnostic queued",
        (
            "The SSKit oracle cleared the global data/camera/evaluator plumbing. The heartbeat is continuing with a cheap "
            "`football-yolo26-diagnostic` run instead of silently spinning on a local-worktree handoff.\n\n"
            "Current verified oracle scores:\n\n"
            f"- exact GT `position_on_pitch`: `{exact}`\n"
            f"- GT ground keypoint projected by SSKit: `{keypoint}`\n"
            f"- GT bbox bottom-center via SSKit: `{bbox}`\n\n"
            "Next action: run one cloud CUDA diagnostic that keeps training off, uses the active football YOLO26 path only, and records "
            "image-space IoU recall beside official `mAP-LocSim`. Soccana is retired from active defaults.\n\n"
            "Suggested Codex Goal:\n\n"
            f"```text\n{codex_goal_text()}\n```\n"
        ),
    )


def resume_from_worktree_block(state: dict[str, Any]) -> None:
    blocker = state.pop("blocker", {})
    state["phase"] = "devkit_detector_diagnostic_pending"
    append_event(
        "auto_resume_from_worktree_block",
        from_phase="blocked_next_worktree_needed",
        to_phase=state["phase"],
        previous_blocker=blocker,
    )
    append_journal(
        "Auto-resumed from blocked_next_worktree_needed into football-yolo26-diagnostic; Soccana remains retired from active defaults."
    )


def handle_devkit_detector_diagnostic_review(state: dict[str, Any]) -> None:
    result = latest_result(state, "football-yolo26-diagnostic")
    if not result:
        state["phase"] = "blocked_detector_diagnostic_missing_result"
        append_event("detector_diagnostic_review_missing_result")
        create_issue(
            "Autonomy blocked: detector diagnostic result missing",
            "The controller reached `devkit_detector_diagnostic_review`, but no `football-yolo26-diagnostic` result exists in `autonomy/state.json` history.",
        )
        append_journal("Detector diagnostic review blocked because the job result is missing from state history.")
        return

    best = result.get("best", {}) if isinstance(result.get("best"), dict) else {}
    metrics = best.get("metrics", {}) if isinstance(best.get("metrics"), dict) else {}
    diagnostics = best.get("diagnostics", {}) if isinstance(best.get("diagnostics"), dict) else {}
    map_locsim = float(metrics.get("map_locsim") or 0.0)
    recall_50 = float(metrics.get("recall_50") or 0.0)
    gt_recall_iou_03 = float(diagnostics.get("gt_recall_iou_0_3") or 0.0)
    gt_recall_iou_05 = float(diagnostics.get("gt_recall_iou_0_5") or 0.0)
    append_event(
        "detector_diagnostic_review",
        map_locsim=map_locsim,
        recall_50=recall_50,
        gt_recall_iou_0_3=gt_recall_iou_03,
        gt_recall_iou_0_5=gt_recall_iou_05,
    )
    state.pop("blocker", None)
    state["phase"] = "synloc_pose_smoke_pending"
    append_journal(
        "Detector diagnostic completed; low image-space recall auto-discarded the detector-only path and queued SynLoc pose smoke."
    )
    create_issue(
        "Autonomy continuing: football YOLO26 discarded, pose smoke queued",
        (
            "The cheap diagnostic completed and the controller made the next decision without waiting for the owner.\n\n"
            f"- official `mAP-LocSim`: `{map_locsim}`\n"
            f"- official `recall_50`: `{recall_50}`\n"
            f"- image-space GT recall @ IoU 0.3: `{gt_recall_iou_03}`\n"
            f"- image-space GT recall @ IoU 0.5: `{gt_recall_iou_05}`\n\n"
            "Decision: discard detector-only YOLO26 as the current frontier and run `synloc-pose-smoke` next. "
            "That smoke trains a tiny YOLO pose/keypoint model on one cached validation slice and evaluates a later validation slice with "
            "`position_from_keypoint_index=1`. It is not promotable as a leaderboard score; it is only a cheap proof that the pose/keypoint "
            "training/eval route can move toward the `.98` SSKit keypoint oracle. Soccana remains retired."
        ),
    )


def resume_detector_diagnostic_block(state: dict[str, Any]) -> None:
    blocker = state.pop("blocker", {})
    state["phase"] = "synloc_pose_smoke_pending"
    append_event(
        "auto_resume_from_detector_diagnostic_block",
        from_phase="blocked_detector_diagnostic_review",
        to_phase=state["phase"],
        previous_blocker=blocker,
    )
    append_journal(
        "Auto-resumed from blocked_detector_diagnostic_review into synloc-pose-smoke; the detector-only path is discarded."
    )


def handle_pose_smoke_review(state: dict[str, Any]) -> None:
    result = latest_result(state, "synloc-pose-smoke")
    if not result:
        state["phase"] = "blocked_pose_smoke_missing_result"
        append_event("pose_smoke_review_missing_result")
        create_issue(
            "Autonomy blocked: pose smoke result missing",
            "The controller reached `pose_smoke_review`, but no `synloc-pose-smoke` result exists in `autonomy/state.json` history.",
        )
        append_journal("Pose smoke review blocked because the job result is missing from state history.")
        return

    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    map_locsim = float(metrics.get("map_locsim") or 0.0)
    recall_50 = float(metrics.get("recall_50") or 0.0)
    state.pop("blocker", None)
    state["phase"] = "train_dataset_cache_pending"
    append_event("pose_smoke_review", map_locsim=map_locsim, recall_50=recall_50)
    append_journal(
        "Pose smoke completed; queued train/valid dataset cache for a real source-specific pose/keypoint experiment, subject to budget gate."
    )
    create_issue(
        "Autonomy continuing: pose smoke reviewed, real train data next",
        (
            "The SynLoc pose smoke completed. It is not a promotable score because it trained on a validation slice, but it is the first "
            "controller step aimed at the keypoint route that matched the SSKit oracle.\n\n"
            f"- smoke `mAP-LocSim`: `{map_locsim}`\n"
            f"- smoke `recall_50`: `{recall_50}`\n"
            f"- predictions: `{result.get('num_predictions', 'unknown')}`\n\n"
            "Next autonomous phase is `train_dataset_cache_pending` so a real train/valid pose experiment can run. "
            "If the weekly cap blocks that, the controller should open the budget issue instead of silently looping."
        ),
    )


def baseline_summary(state: dict[str, Any]) -> str:
    result = latest_result(state, "baseline-full") or latest_result(state, "baseline-probe") or {}
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    return (
        f"run_id={result.get('run_id', 'unknown')}; "
        f"mAP-LocSim={metrics.get('map_locsim', 'unknown')}; "
        f"threshold={metrics.get('score_threshold', 'unknown')}; "
        f"num_images={result.get('num_images', 'unknown')}; "
        f"num_detections={result.get('num_detections', 'unknown')}; "
        f"model={result.get('model', 'unknown')}"
    )


def council_question(state: dict[str, Any]) -> str:
    return (
        "We have a TorchVision Faster R-CNN SynLoc baseline. "
        f"Baseline summary: {baseline_summary(state)}.\n\n"
        "Please recommend the next three highest expected-value experiments to improve official mAP-LocSim within the $25/week budget. "
        "For each, include the smallest CUDA smoke/probe, expected upside, failure signal, and whether to keep/discard.\n\n"
        "SoccerMaster is a serious soccer-specific lead, not a mandate. "
        "The paper reports 92.3 athlete-detection AP@50, 50.5 mAP, and 99.2 role accuracy, so the sibling zero-score run should be treated as a likely runtime/config/decode failure. "
        "Please recommend how to audit weight placement, role mapping, class dimensions, normalization, thresholds, and raw logits before interpreting SoccerMaster scores."
    )


def handle_council_after_baseline(state: dict[str, Any]) -> None:
    queue = existing_council_queue()
    title = "SynLoc strategy after baseline"
    question = council_question(state)
    if not queue:
        state["phase"] = "blocked_council_request"
        append_event(
            "council_queue_missing",
            candidates=[str(path) for path in council_queue_candidates()],
        )
        create_issue(
            "Autonomy blocked: council queue unavailable",
            (
                "The full baseline phase completed, but the controller cannot see the challenge council inbox from this runtime.\n\n"
                "Queue this request locally with:\n\n"
                "```bash\n"
                f"python3 scripts/ask_council.py --title {json.dumps(title)} --question {json.dumps(question)}\n"
                "```\n\n"
                "After the council report is available, resume the next experiment phase."
            ),
        )
        return

    code, output = run(
        [
            sys.executable,
            "scripts/ask_council.py",
            "--title",
            title,
            "--question",
            question,
            "--queue",
            str(queue),
        ]
    )
    if code != 0:
        state["phase"] = "blocked_council_request"
        append_event("council_request_failed", output=output)
        create_issue(
            "Autonomy blocked: council request failed",
            f"The controller tried to queue the post-baseline council request but failed.\n\n```text\n{output[-4000:]}\n```",
        )
        return

    request_path = output.strip().splitlines()[-1]
    state["phase"] = "awaiting_council_report"
    state["council_request"] = {"title": title, "path": request_path, "queued_at": utc_now()}
    append_event("council_request_queued", path=request_path)


def handle_council_report_wait(state: dict[str, Any]) -> None:
    request = state.get("council_request") or {}
    request_path = Path(str(request.get("path", ""))).expanduser()
    if not request_path.name:
        state["phase"] = "blocked_council_request"
        append_event("council_request_missing_from_state")
        return

    request_id = request_path.name
    queue = existing_council_queue()
    queue_root = queue.parent if queue else request_path.parent.parent
    done_report = queue_root / "done" / request_id / "final_report.md"
    failed_status = queue_root / "failed" / request_id / "status.json"
    if done_report.exists():
        report = done_report.read_text(encoding="utf-8", errors="replace")
        state["phase"] = "first_experiment_pending"
        state["council_report"] = {"path": str(done_report), "received_at": utc_now()}
        append_event("council_report_ready", path=str(done_report))
        create_issue(
            "Council report ready: choose first SynLoc experiment",
            (
                f"Council report: `{done_report}`\n\n"
                "Open a small isolated experiment branch/worktree from the recommendations and keep/discard by official mAP-LocSim.\n\n"
                "Report excerpt:\n\n"
                f"```text\n{report[:4000]}\n```"
            ),
        )
        return
    if failed_status.exists():
        state["phase"] = "blocked_council_request"
        append_event("council_report_failed", path=str(failed_status))
        create_issue(
            "Autonomy blocked: council request failed",
            f"Council request `{request_id}` failed. See `{failed_status}`.",
        )
        return
    append_event("council_report_pending", request_id=request_id)


def block_once(state: dict[str, Any], *, phase: str, title: str, body: str, missing: list[str]) -> None:
    blocker = state.get("blocker") or {}
    state["phase"] = "blocked_missing_secret"
    state["blocked_phase"] = phase
    state["blocked_missing"] = missing
    state["blocker"] = {"title": title, "created_at": blocker.get("created_at") or utc_now()}
    append_event("blocked_missing_secret", phase=phase, missing=missing)
    if blocker.get("title") != title:
        create_issue(title, body)


def job_status(job: Any) -> str:
    stage = getattr(getattr(job, "status", None), "stage", None)
    return getattr(stage, "value", str(stage))


def job_to_dict(job: Any) -> dict[str, Any]:
    return {
        "id": getattr(job, "id", ""),
        "url": getattr(job, "url", ""),
        "status": job_status(job),
        "flavor": str(getattr(job, "flavor", "")),
    }


def parse_autonomy_result(logs: str) -> dict[str, Any] | None:
    marker = "AUTONOMY_RESULT "
    for line in reversed(logs.splitlines()):
        if marker in line:
            raw = line.split(marker, 1)[1].strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"ok": False, "parse_error": raw[:1000]}
    return None


def next_phase_for_result(active: dict[str, Any], result: dict[str, Any]) -> str:
    next_phase = active.get("next_phase", active.get("phase", "blocked"))
    if not result.get("ok", False):
        return "blocked"
    if active.get("label") == "pretrained-yolo-baseline":
        best = result.get("best", {}) if isinstance(result.get("best"), dict) else {}
        metrics = best.get("metrics", {}) if isinstance(best.get("metrics"), dict) else {}
        detections = int(best.get("num_detections") or 0)
        score = float(metrics.get("map_locsim") or 0.0)
        recall = float(metrics.get("recall_50") or 0.0)
        if detections <= 0 or score < PRETRAINED_YOLO_BASELINE_MIN_MAP_LOCSIM or recall <= 0.0:
            return "blocked_pretrained_yolo_baseline_eval"
        return next_phase
    if active.get("label") != "soccermaster-wiring-probe":
        return next_phase
    athlete_count = int(result.get("athlete_like_at_conf_0_05") or 0)
    if athlete_count <= 0:
        return "soccermaster_config_mismatch_review"
    return "soccermaster_synloc_conversion_probe_pending"


def inspect_active_job(api: HfApi, state: dict[str, Any]) -> bool:
    active = state.get("active_job")
    if not active:
        return False
    job = api.inspect_job(job_id=active["id"], token=os.environ["HF_TOKEN"])
    status = job_status(job)
    active["status"] = status
    active["url"] = getattr(job, "url", active.get("url", ""))
    append_event("job_inspected", job=active)
    if status in TERMINAL_OK:
        logs = "\n".join(api.fetch_job_logs(job_id=active["id"], token=os.environ["HF_TOKEN"]))
        result = parse_autonomy_result(logs) or {"ok": False, "error": "No AUTONOMY_RESULT marker in logs"}
        state.setdefault("history", []).append({**active, "completed_at": utc_now(), "result": result})
        state["active_job"] = None
        state["phase"] = next_phase_for_result(active, result)
        append_ledger_completion(active, result)
        append_journal(
            f"HF job completed: {active.get('label')} {active.get('id')} -> phase {state['phase']}."
        )
        if state["phase"] == "blocked_pretrained_yolo_baseline_eval":
            best = result.get("best", {}) if isinstance(result, dict) else {}
            metrics = best.get("metrics", {}) if isinstance(best.get("metrics"), dict) else {}
            state["blocker"] = {
                "title": "Pretrained YOLO baseline eval is effectively zero",
                "created_at": utc_now(),
                "job_id": active.get("id"),
                "model": best.get("model"),
                "map_locsim": metrics.get("map_locsim"),
                "recall_50": metrics.get("recall_50"),
                "num_detections": best.get("num_detections"),
            }
            create_issue(
                "Autonomy blocked: pretrained YOLO baseline eval is effectively zero",
                (
                    f"Job: {active.get('url') or active.get('id')}\n\n"
                    f"Best model: `{best.get('model', 'unknown')}`\n"
                    f"mAP-LocSim: `{metrics.get('map_locsim', 'unknown')}`\n"
                    f"recall_50: `{metrics.get('recall_50', 'unknown')}`\n"
                    f"num_detections: `{best.get('num_detections', 'unknown')}`\n\n"
                    "Training is intentionally blocked. The detector is producing boxes, but projected SynLoc localization is still effectively zero. "
                    "Debug class ids, bbox-to-pitch projection, camera normalization, annotation/image pairing, and official evaluator assumptions before starting `TRAIN_MODE=finetune`."
                ),
            )
        append_event("job_completed", job=active, result=result)
        return True
    if status in TERMINAL_BAD:
        logs = "\n".join(api.fetch_job_logs(job_id=active["id"], token=os.environ["HF_TOKEN"]))
        state.setdefault("history", []).append({**active, "failed_at": utc_now(), "logs_tail": logs[-4000:]})
        state["active_job"] = None
        state["phase"] = "blocked"
        append_event("job_failed", job=active, logs_tail=logs[-4000:])
        append_journal(f"HF job failed: {active.get('label')} {active.get('id')} -> blocked.")
        create_issue(
            f"Autonomy blocked: HF job {active['label']} failed",
            f"Job: {active.get('url') or active['id']}\n\nPhase: `{active.get('phase')}`\n\nLogs tail:\n\n```text\n{logs[-4000:]}\n```",
        )
        return True
    return True


def submit_next_job(api: HfApi, state: dict[str, Any], *, dry_run: bool) -> None:
    phase = state.get("phase")
    if phase == "council_after_baseline_pending":
        if dry_run:
            append_event("dry_run_council_request", question=council_question(state))
            return
        handle_council_after_baseline(state)
        return
    if phase == "awaiting_council_report":
        handle_council_report_wait(state)
        return
    if phase == "devkit_oracle_review":
        handle_devkit_oracle_review(state, dry_run=dry_run)
        return
    if phase == "devkit_detector_diagnostic_review":
        handle_devkit_detector_diagnostic_review(state)
        return
    if phase == "pose_smoke_review":
        handle_pose_smoke_review(state)
        return
    if phase == "blocked_next_worktree_needed":
        if dry_run:
            append_event("dry_run_resume_from_worktree_block")
            return
        resume_from_worktree_block(state)
        phase = state.get("phase")
    if phase == "blocked_detector_diagnostic_review":
        if dry_run:
            append_event("dry_run_resume_from_detector_diagnostic_block")
            return
        resume_detector_diagnostic_block(state)
        phase = state.get("phase")
    if phase and phase.startswith("blocked_"):
        append_event("still_blocked", phase=phase)
        return
    spec = JOB_SPECS.get(phase)
    if not spec:
        append_event("no_job_for_phase", phase=phase)
        return

    required_secrets = spec.get("required_secrets", [])
    missing = [key for key in required_secrets if not os.getenv(key)]
    for group in spec.get("required_secret_groups", []):
        if not any(os.getenv(key) for key in group):
            missing.append("/".join(group))
    if missing:
        block_once(
            state,
            phase=phase,
            missing=missing,
            title=f"Autonomy blocked: missing {', '.join(missing)}",
            body=(
                f"The controller reached `{phase}` but the GitHub Actions environment is missing: `{', '.join(missing)}`.\n\n"
                "Add the missing repository secret(s), then resume by setting `autonomy/state.json` phase back to "
                f"`{phase}` or by asking Codex to resume the blocked phase."
            ),
        )
        return

    budget = float(os.getenv("WEEKLY_BUDGET_USD", state.get("weekly_budget_usd", 25.0)))
    spent = float(state.get("spent_estimate_usd", 0.0))
    cost = float(spec["cost_estimate_usd"])
    if spent + cost > budget:
        state["phase"] = "blocked_budget"
        append_event("budget_block", phase=phase, spent=spent, cost=cost, budget=budget)
        create_issue(
            f"Autonomy needs budget approval for {spec['label']}",
            f"Current estimated weekly spend is `${spent:.2f}`. Next job `{spec['label']}` reserves `${cost:.2f}`, which exceeds the `${budget:.2f}` weekly cap.\n\nApprove more budget or lower the scope.",
        )
        return

    env = {
        "HF_DATASET_REPO": state["hf_dataset_repo"],
        "HF_MODEL_REPO": state["hf_model_repo"],
        "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY", "DMontgomery40/auto-research-2"),
        **spec.get("env", {}),
    }
    secrets = {
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "SOCCERNET_PASSWORD": os.environ["SOCCERNET_PASSWORD"],
    }
    if os.getenv("SOCCERNET_USERNAME"):
        secrets["SOCCERNET_USERNAME"] = os.environ["SOCCERNET_USERNAME"]
    signin_password = env_any("SOCCERNET_SIGNIN_PASSWORD", "SOCCERNET_PASSWORD_2", "SPIIDEO_PASSWORD")
    if signin_password:
        secrets["SOCCERNET_SIGNIN_PASSWORD"] = signin_password
    if os.getenv("SOCCERNET_PASSWORD_2"):
        secrets["SOCCERNET_PASSWORD_2"] = os.environ["SOCCERNET_PASSWORD_2"]
    script = ROOT / spec["script"]
    if dry_run:
        append_event("dry_run_submit", phase=phase, script=str(script), spec=spec)
        return

    job = api.run_uv_job(
        str(script),
        env=env,
        secrets=secrets,
        flavor=spec["flavor"],
        timeout=spec["timeout"],
        python=spec.get("python"),
        token=os.environ["HF_TOKEN"],
    )
    active = {
        **job_to_dict(job),
        "label": spec["label"],
        "phase": phase,
        "next_phase": spec["next_phase"],
        "cost_estimate_usd": cost,
        "submitted_at": utc_now(),
    }
    state["active_job"] = active
    state["phase"] = phase.replace("_pending", "_running")
    state["spent_estimate_usd"] = round(spent + cost, 2)
    append_budget_submission(active, spec)
    append_journal(f"Submitted HF job: {active.get('label')} {active.get('id')} on {spec['flavor']}.")
    append_event("job_submitted", job=active)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one autonomous controller tick.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    state = load_state()
    if not state.get("enabled", True) or os.getenv("AUTONOMY_ENABLED", "1") in {"0", "false", "False"}:
        append_event("disabled")
        return

    missing = [key for key in ("HF_TOKEN", "SOCCERNET_PASSWORD") if not os.getenv(key)]
    if missing:
        state["phase"] = "blocked_missing_secret"
        append_event("missing_secret", missing=missing)
        create_issue("Autonomy blocked: missing secrets", f"Missing required secrets: `{', '.join(missing)}`")
        write_state(state)
        return

    api = HfApi(token=os.environ["HF_TOKEN"])
    append_event("tick_start", phase=state.get("phase"), active_job=state.get("active_job"))
    if state.get("phase") == "blocked_missing_secret":
        missing = [key for key in state.get("blocked_missing", []) if not os.getenv(key)]
        if missing:
            append_event("still_blocked_missing_secret", missing=missing, blocked_phase=state.get("blocked_phase"))
            write_state(state)
            return
        state["phase"] = state.pop("blocked_phase", "dataset_cache_valid_pending")
        state.pop("blocked_missing", None)
        state.pop("blocker", None)
        append_event("blocker_cleared", phase=state.get("phase"))
    if not inspect_active_job(api, state):
        submit_next_job(api, state, dry_run=args.dry_run)
    write_state(state)
    update_current_snapshot(state)
    append_event("tick_end", phase=state.get("phase"), active_job=state.get("active_job"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        append_event("tick_crashed", error=repr(exc))
        raise
