# Council

The challenge council is available at:

```text
../challenge-council/
/Users/davidmontgomery/challenge-council/
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

`scripts/ask_council.py` auto-discovers the existing council inbox. Override it with `CHALLENGE_COUNCIL_INBOX` or `--queue` if this checkout moves.

Good council requests include:

- official task summary,
- current best score,
- baseline command and metrics,
- experiment ledger excerpt,
- ideas already tried,
- `COUNCIL_DOSSIER.md`,
- `autonomy/state.json` summary and recent `autonomy/events.jsonl`,
- the baseline implementation being criticized,
- budget constraints,
- concrete question.

The request generator now includes a richer dossier by default. Use the council as the external skeptical board: it should be allowed to say the agent is wasting time, missing obvious soccer-specific context, or spending too much compute on the wrong signal.

Do not use the council as a substitute for running the baseline. Do not ask it to mine leaked submissions or post-deadline winning writeups.
