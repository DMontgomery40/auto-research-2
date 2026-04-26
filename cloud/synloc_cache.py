# /// script
# dependencies = [
#   "huggingface_hub>=0.24.0",
#   "SoccerNet"
# ]
# ///
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = ["HF_TOKEN", "SOCCERNET_USERNAME", "SOCCERNET_PASSWORD", "HF_DATASET_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    from SoccerNet.Downloader import SoccerNetDownloader

    splits = [item.strip() for item in os.getenv("SYNLOC_SPLITS", "valid").split(",") if item.strip()]
    version = os.getenv("SYNLOC_VERSION", "fullhd")
    root = Path(os.getenv("SYNLOC_DOWNLOAD_ROOT", "/tmp/SoccerNet"))
    root.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(root))
    downloader.getSpiideoCredentials = lambda: (os.environ["SOCCERNET_USERNAME"], os.environ["SOCCERNET_PASSWORD"])
    kwargs = {
        "task": "SpiideoSynLoc",
        "split": splits,
        "password": os.environ["SOCCERNET_PASSWORD"],
    }
    if version == "fullhd":
        kwargs["version"] = "fullhd"
    downloader.downloadDataTask(**kwargs)

    zips = sorted(root.glob("*.zip"))
    if not zips:
        zips = sorted(root.rglob("*.zip"))
    if not zips:
        raise RuntimeError(f"No zip files found after download in {root}")

    manifest = {
        "ok": True,
        "ts": utc_now(),
        "splits": splits,
        "version": version,
        "files": [
            {
                "name": path.name,
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "zip_ok": zipfile.is_zipfile(path),
            }
            for path in zips
        ],
    }

    stage = Path("/tmp/synloc-upload")
    out_dir = stage / "raw" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in zips:
        target = out_dir / path.name
        if not target.exists():
            target.write_bytes(path.read_bytes())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_large_folder(
        repo_id=os.environ["HF_DATASET_REPO"],
        repo_type="dataset",
        folder_path=stage,
        private=True,
        num_workers=4,
    )
    print("AUTONOMY_RESULT " + json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
