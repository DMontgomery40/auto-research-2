#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT.parent / "challenge-council" / "data" / "automation_queue" / "inbox"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-._") or "council-request"


def read_if_exists(path: Path) -> str:
    if not path.exists():
        return f"_Missing: {path.name}_\n"
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue a Challenge Council request.")
    parser.add_argument("--title", required=True, help="Short request title.")
    parser.add_argument("--question", default="", help="Specific question for the council.")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE, help="Council inbox directory.")
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

## Current State

{read_if_exists(ROOT / "CURRENT.md")}

## Experiment Ledger

{read_if_exists(ROOT / "LEDGER.md")}

## Ideas

{read_if_exists(ROOT / "IDEAS.md")}
"""
    (request_dir / "council_request.md").write_text(body, encoding="utf-8")
    (request_dir / "request.json").write_text(
        json.dumps({"workspace_label": "auto-research-2"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(request_dir)


if __name__ == "__main__":
    main()
