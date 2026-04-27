# Budget

Weekly compute budget: `$25/week`.

Week starts: Monday.

| Date | Tag | Provider | Hardware | Est. Cost | Actual Cost | Purpose | Result |
|---|---|---|---|---:|---:|---|---|
| 2026-04-26 | setup | local | CPU | $0.00 | $0.00 | Control-plane repo setup | Done |
| 2026-04-26 | cloud-smoke attempts | HF Jobs | T4 small | $0.75 | pending | Verify GPU, imports, and private HF write | Passed on third attempt |
| 2026-04-26 | dataset-cache-valid attempts | HF Jobs | CPU upgrade | $2.00 | pending | Cache fullhd validation split in private HF dataset repo | Passed on second attempt |
| 2026-04-27 | baseline-probe | HF Jobs | T4 small | $1.50 | pending | Small YOLO validation probe | Failed before execution; patched and retrying |
| 2026-04-27 | baseline-probe retry | HF Jobs | T4 small | $1.50 | pending | Small YOLO validation probe | Failed on NumPy ABI; pinned `numpy<2` and retrying |
| 2026-04-27 | baseline-probe retry | HF Jobs | T4 small | $1.50 | pending | Small validation probe | Failed on `libGL`; switched to TorchVision detector and retrying |
| 2026-04-27 | baseline-probe retry | HF Jobs | T4 small | $1.50 | pending | Small validation probe | Failed on missing `scipy`; added dependency and retrying |

## Spend Rule

If an experiment could push the week over `$25`, open a GitHub issue before running it.

Issue should include:

- experiment name,
- why it might beat current best,
- estimated cost,
- stop condition,
- fallback if it fails.
