#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def env_any(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def ensure_package() -> None:
    try:
        import SoccerNet  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "SoccerNet"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SoccerNet SpiideoSynLoc data.")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "SoccerNet")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test", "challenge"])
    parser.add_argument("--version", choices=["4k", "fullhd"], default="fullhd")
    args = parser.parse_args()

    load_env(ROOT / ".env")
    username = env_any("SOCCERNET_USERNAME", "SPIIDEO_USERNAME")
    data_password = os.getenv("SOCCERNET_PASSWORD")
    signin_password = env_any("SOCCERNET_SIGNIN_PASSWORD", "SOCCERNET_PASSWORD_2", "SPIIDEO_PASSWORD", "SOCCERNET_PASSWORD")
    if not username:
        raise SystemExit("SOCCERNET_USERNAME is missing from .env or environment.")
    if not data_password:
        raise SystemExit("SOCCERNET_PASSWORD is missing from .env or environment.")
    if not signin_password:
        raise SystemExit("SOCCERNET_SIGNIN_PASSWORD is missing from .env or environment.")

    ensure_package()
    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=str(args.root))
    downloader.getSpiideoCredentials = lambda: (username, signin_password)
    kwargs = {"task": "SpiideoSynLoc", "split": args.splits, "password": data_password}
    if args.version == "fullhd":
        kwargs["version"] = "fullhd"
    downloader.downloadDataTask(**kwargs)

    print(args.root / "SpiideoSynLoc")


if __name__ == "__main__":
    main()
