# Council

The challenge council is available at:

```text
../challenge-council/
```

Automation inbox contract:

```text
../challenge-council/data/automation_queue/inbox/<request_id>/council_request.md
../challenge-council/data/automation_queue/inbox/<request_id>/request.json
```

Worker outputs:

```text
../challenge-council/data/automation_queue/done/<request_id>/final_report.md
../challenge-council/data/automation_queue/failed/<request_id>/status.json
```

Use:

```bash
python3 scripts/ask_council.py --title "SynLoc strategy after baseline"
```

Good council requests include:

- official task summary,
- current best score,
- baseline command and metrics,
- experiment ledger excerpt,
- ideas already tried,
- budget constraints,
- concrete question.

Do not use the council as a substitute for running the baseline. Do not ask it to mine leaked submissions or post-deadline winning writeups.
