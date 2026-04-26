#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -f README.md
test -f AGENTS.md
test -f program.md
test -f CURRENT.md
test -f LEDGER.md
test -f IDEAS.md
test -f BUDGET.md
test -f COUNCIL.md
test -f .gitignore

if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo ".env is tracked; remove it from git" >&2
  exit 1
fi

if [ -d refs ] && git status --short --ignored refs | grep -v '^!! refs/' >/dev/null; then
  echo "refs/ should stay ignored" >&2
  exit 1
fi

python3 -m py_compile scripts/ask_council.py

echo "verify ok"
