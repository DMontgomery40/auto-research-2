#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"ERROR", "CANCELED", "DELETED"}

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
        "cost_estimate_usd": 6.0,
        "next_phase": "council_after_baseline_pending",
        "env": {"SYNLOC_SPLIT": "valid", "SYNLOC_VERSION": "fullhd", "BASELINE_MAX_IMAGES": "0"},
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


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(event: str, **payload: Any) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": utc_now(), "event": event, **payload}, sort_keys=True) + "\n")


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
        state["phase"] = active.get("next_phase", state.get("phase"))
        append_event("job_completed", job=active, result=result)
        return True
    if status in TERMINAL_BAD:
        logs = "\n".join(api.fetch_job_logs(job_id=active["id"], token=os.environ["HF_TOKEN"]))
        state.setdefault("history", []).append({**active, "failed_at": utc_now(), "logs_tail": logs[-4000:]})
        state["active_job"] = None
        state["phase"] = "blocked"
        append_event("job_failed", job=active, logs_tail=logs[-4000:])
        create_issue(
            f"Autonomy blocked: HF job {active['label']} failed",
            f"Job: {active.get('url') or active['id']}\n\nPhase: `{active.get('phase')}`\n\nLogs tail:\n\n```text\n{logs[-4000:]}\n```",
        )
        return True
    return True


def submit_next_job(api: HfApi, state: dict[str, Any], *, dry_run: bool) -> None:
    phase = state.get("phase")
    spec = JOB_SPECS.get(phase)
    if not spec:
        append_event("no_job_for_phase", phase=phase)
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
    if not inspect_active_job(api, state):
        submit_next_job(api, state, dry_run=args.dry_run)
    write_state(state)
    append_event("tick_end", phase=state.get("phase"), active_job=state.get("active_job"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        append_event("tick_crashed", error=repr(exc))
        raise
