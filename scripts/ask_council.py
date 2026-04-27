#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COUNCIL_INBOX = Path("challenge-council") / "data" / "automation_queue" / "inbox"
MAX_TEXT_CHARS = 20000


def queue_candidates() -> list[Path]:
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


def default_queue() -> Path:
    candidates = queue_candidates()
    env_queue = os.getenv("CHALLENGE_COUNCIL_INBOX")
    if env_queue:
        return Path(env_queue).expanduser()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-._") or "council-request"


def read_if_exists(path: Path) -> str:
    if not path.exists():
        return f"_Missing: {path.name}_\n"
    return path.read_text(encoding="utf-8", errors="replace")


def read_excerpt(path: Path, *, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = read_if_exists(path)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n_[Truncated at {max_chars} characters from {path.name}]_\n"


def fenced(path: Path, *, language: str = "text", max_chars: int = MAX_TEXT_CHARS) -> str:
    return f"```{language}\n{read_excerpt(path, max_chars=max_chars)}\n```"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"Could not parse {path.name}"}
    return payload if isinstance(payload, dict) else {"error": f"{path.name} was not a JSON object"}


def state_summary() -> str:
    state = load_json(ROOT / "autonomy" / "state.json")
    if not state:
        return "_Missing autonomy/state.json_\n"
    history = state.get("history") if isinstance(state.get("history"), list) else []
    summary = {
        "phase": state.get("phase"),
        "active_job": state.get("active_job"),
        "spent_estimate_usd": state.get("spent_estimate_usd"),
        "weekly_budget_usd": state.get("weekly_budget_usd"),
        "updated_at": state.get("updated_at"),
        "recent_history": history[-5:],
    }
    return json.dumps(summary, indent=2, sort_keys=True)


def events_tail(lines: int = 40) -> str:
    path = ROOT / "autonomy" / "events.jsonl"
    if not path.exists():
        return "_Missing autonomy/events.jsonl_\n"
    rows = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(rows[-lines:]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue a Challenge Council request.")
    parser.add_argument("--title", required=True, help="Short request title.")
    parser.add_argument("--question", default="", help="Specific question for the council.")
    parser.add_argument("--queue", type=Path, default=default_queue(), help="Council inbox directory.")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    request_id = f"{now}-{slugify(args.title)}"
    request_dir = args.queue / request_id
    request_dir.mkdir(parents=True, exist_ok=False)

    question = args.question.strip() or "Given the current baseline state and constraints, what are the highest expected-value next experiments?"
    body = f"""# {args.title}

Created: {now}
Project: auto-research-2
Target: 2026 Spiideo SoccerNet SynLoc

## Question

{question}

## Constraints

- Goal: beat the best tracked SynLoc score by June 30, 2026.
- Primary metric: mAP-LocSim, higher is better.
- Weekly compute budget: $25 unless owner approves more by GitHub issue.
- Keep the harness simple and markdown-first.
- Do not use leaked solutions or post-deadline winner writeups.
- The council is the outside reviewer: be blunt, call out dumb plans, and give high-context strategic hints.
- The council can recommend outside resources or model directions, but the autonomy repo still owns execution and budget gates.

## Council Dossier

{read_excerpt(ROOT / "COUNCIL_DOSSIER.md")}

## Autonomy State Summary

```json
{state_summary()}
```

## Current State

{read_excerpt(ROOT / "CURRENT.md")}

## Experiment Ledger

{read_excerpt(ROOT / "LEDGER.md")}

## Budget Ledger

{read_excerpt(ROOT / "BUDGET.md")}

## Ideas

{read_excerpt(ROOT / "IDEAS.md")}

## Operating Program

{read_excerpt(ROOT / "program.md")}

## Baseline Implementation Under Review

The current baseline script is included so the council can identify dumb assumptions directly.

{fenced(ROOT / "cloud" / "synloc_baseline_yolo.py", language="python")}

## Official Devkit Baseline Reference

{fenced(ROOT / "refs" / "sskit" / "baseline.py", language="python")}

## Recent Autonomy Events

```jsonl
{events_tail()}
```
"""
    (request_dir / "council_request.md").write_text(body, encoding="utf-8")
    (request_dir / "request.json").write_text(
        json.dumps({"workspace_label": "auto-research-2"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(request_dir)


if __name__ == "__main__":
    main()
