#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COUNCIL_INBOX = Path("challenge-council") / "data" / "automation_queue" / "inbox"
MAX_TEXT_CHARS = 12000


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
- Keep the harness simple and markdown-first.
- Do not use leaked solutions, post-deadline solution writeups, or prior SoccerNet result material.
- If official 2026 results are not available yet, include no results material.
- The council is the outside reviewer: be blunt, call out dumb plans, and give high-context strategic hints.
- The council can recommend outside resources or model directions, but this repo owns execution and budget gates.

## Current State

{read_excerpt(ROOT / "CURRENT.md")}

## Experiment Ledger

{read_excerpt(ROOT / "LEDGER.md")}

## Ideas

{read_excerpt(ROOT / "IDEAS.md")}

## Implementation Pointers

Do not spend Stage 1 reading full source unless needed. The key files are:

- Operating loop: `program.md`
- Current editable payload: `train.py`
- Official devkit baseline reference: `refs/sskit/baseline.py`
- Official evaluator wrapper: `scripts/evaluate_synloc.py`

The important current assumption is already summarized above: generic detector
fine-tuning failed, while the SSKit keypoint/oracle path is the strongest signal.
"""
    (request_dir / "council_request.md").write_text(body, encoding="utf-8")
    (request_dir / "request.json").write_text(
        '{\n  "workspace_label": "auto-research-2"\n}\n',
        encoding="utf-8",
    )
    print(request_dir)


if __name__ == "__main__":
    main()
