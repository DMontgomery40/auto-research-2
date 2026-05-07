#!/usr/bin/env python3
"""Print the history context every Codex research tick must honor first."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LedgerRow:
    date: str
    tag: str
    runtime: str
    command: str
    score: str
    threshold: str
    decision: str
    notes: str


def split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    previous_was_backslash = False
    for char in stripped:
        if char == "|" and not previous_was_backslash:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        previous_was_backslash = char == "\\" and not previous_was_backslash
        if char != "\\":
            previous_was_backslash = False
    cells.append("".join(current).strip())
    return cells


def strip_code(value: str) -> str:
    return value.replace("`", "").replace(r"\|", "|").strip()


def parse_ledger(path: Path) -> list[LedgerRow]:
    rows: list[LedgerRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = split_markdown_row(line)
        if len(cells) != 8 or cells[0] in {"Date", "---"}:
            continue
        if not cells[0][:4].isdigit():
            continue
        rows.append(
            LedgerRow(
                date=strip_code(cells[0]),
                tag=strip_code(cells[1]),
                runtime=strip_code(cells[2]),
                command=strip_code(cells[3]),
                score=strip_code(cells[4]),
                threshold=strip_code(cells[5]),
                decision=strip_code(cells[6]),
                notes=strip_code(cells[7]),
            )
        )
    return rows


def extract_section(path: Path, heading: str, max_lines: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        return []
    section: list[str] = []
    for line in lines[start:]:
        if line.startswith("## ") and section:
            break
        if line.strip():
            section.append(line)
        if len(section) >= max_lines:
            break
    return section


def summarize_tried(rows: list[LedgerRow]) -> list[str]:
    tags = ", ".join(row.tag for row in rows)
    return wrap_line(tags, width=100)


def wrap_line(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        next_len = current_len + len(word) + (1 if current else 0)
        if current and next_len > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = next_len
    if current:
        lines.append(" ".join(current))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize tried experiments and current no-repeat rules."
    )
    parser.add_argument("--recent", type=int, default=18, help="recent ledger rows to detail")
    parser.add_argument(
        "--next-lines", type=int, default=28, help="lines to include from CURRENT.md Next Action"
    )
    args = parser.parse_args()

    ledger_rows = parse_ledger(ROOT / "LEDGER.md")
    next_action = extract_section(ROOT / "CURRENT.md", "## Next Action", args.next_lines)

    print("# Research History Gate")
    print()
    print("Highest-priority selection rules:")
    print("- LEDGER.md and CURRENT.md override IDEAS.md; IDEAS.md is a backlog, not permission to rerun.")
    print("- Build a tried index before choosing: tag, candidate source, train mode, env knobs, score, decision, and notes.")
    print("- Reject exact repeats and single-knob repeats already marked discard/tie/no-improvement.")
    print("- If a candidate appears in history, the new pass must say which prior row it builds on and why it is not the same attempt.")
    print("- If IDEAS.md contains stale tried work, update IDEAS.md or choose a non-stale experiment.")
    print()

    if next_action:
        print("Current.md Next Action:")
        for line in next_action:
            print(line)
        print()

    if ledger_rows:
        print("All tried ledger tags:")
        for line in summarize_tried(ledger_rows):
            print(line)
        print()
        print(f"Recent {min(args.recent, len(ledger_rows))} ledger rows:")
        for row in ledger_rows[-args.recent :]:
            print(
                f"- {row.date} | {row.tag} | score={row.score} | "
                f"decision={row.decision} | command={row.command}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
