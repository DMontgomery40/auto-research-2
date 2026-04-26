# /// script
# dependencies = [
#   "huggingface_hub>=0.24.0",
#   "SoccerNet",
#   "sskit",
#   "xtcocotools"
# ]
# ///
from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone

from huggingface_hub import HfApi


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run(cmd: list[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=30, check=False)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def main() -> None:
    required = ["HF_TOKEN", "SOCCERNET_PASSWORD", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    import SoccerNet  # noqa: F401
    import sskit  # noqa: F401
    import xtcocotools  # noqa: F401

    api = HfApi(token=os.environ["HF_TOKEN"])
    whoami = api.whoami()
    nvidia = run(["nvidia-smi"])
    result = {
        "ok": True,
        "ts": utc_now(),
        "python": platform.python_version(),
        "hf_user": whoami.get("name"),
        "dataset_repo": os.environ["HF_DATASET_REPO"],
        "model_repo": os.environ["HF_MODEL_REPO"],
        "nvidia_smi_ok": nvidia["ok"],
        "nvidia_smi_stdout_tail": nvidia.get("stdout", ""),
        "imports": ["SoccerNet", "sskit", "xtcocotools"],
    }
    api.upload_file(
        repo_id=os.environ["HF_DATASET_REPO"],
        repo_type="dataset",
        path_in_repo=f"autonomy/smoke/{utc_now().replace(':', '-')}.json",
        path_or_fileobj=json.dumps(result, indent=2, sort_keys=True).encode("utf-8"),
        commit_message="Record auto-research smoke check",
    )
    print("AUTONOMY_RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
