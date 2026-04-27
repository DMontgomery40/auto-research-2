# /// script
# dependencies = [
#   "huggingface-hub",
#   "numpy<2",
#   "opencv-python-headless",
#   "torch",
#   "torchvision",
#   "transformers",
# ]
# ///
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import torch
from huggingface_hub import HfApi, hf_hub_download


ASSET_REPO = os.getenv("V2D_ASSET_REPO", "dmontgomery40/v2d-research-assets")
MODEL_REPO = os.environ["HF_MODEL_REPO"]
TOKEN = os.environ["HF_TOKEN"]
MAX_IMAGES = int(os.getenv("SOCCERMASTER_MAX_IMAGES", "4"))
WORK = Path("/tmp/soccermaster_wiring_probe")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def download_asset(filename: str, local_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=ASSET_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=str(local_dir),
            token=TOKEN,
        )
    )


def materialize_adapter() -> Path:
    for filename in (
        "vendor/rondo/backend/app/soccermaster_adapter.py",
        "vendor/rondo/backend/app/soccermaster_bootstrap.py",
    ):
        download_asset(filename, WORK)

    model_dir = WORK / "vendor" / "rondo" / "backend" / "models" / "soccermaster"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "backbone.pt",
        "KeypointsDetection.pt",
        "LinesDetection.pt",
        "SoccerNetGSR_Detection.pt",
    ):
        src = download_asset(f"rondo_payload/models/soccermaster/{filename}", WORK)
        dst = model_dir / filename
        if not dst.exists():
            shutil.copy2(src, dst)
    return WORK / "vendor" / "rondo" / "backend" / "app" / "soccermaster_adapter.py"


def import_adapter(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("soccermaster_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def image_paths() -> list[Path]:
    return [
        download_asset(f"benchmarks/loc.synloc_quick_v1/val/{idx:06d}.jpg", WORK / "quick")
        for idx in range(MAX_IMAGES)
    ]


def raw_detection_summary(adapter: Any, pipe: Any, frame_bgr: Any) -> dict[str, Any]:
    h_orig, w_orig = frame_bgr.shape[:2]
    with torch.no_grad():
        tensor = pipe._preprocess(frame_bgr)
        spatial = pipe._extract_backbone_features(tensor)
        det_out = pipe._detection_head(spatial)

        logits = det_out["pred_logits"][0]
        roles = det_out["pred_roles"][0]
        scores = torch.sigmoid(logits).max(dim=-1).values
        role_probs = torch.softmax(roles, dim=-1)
        role_ids = role_probs.argmax(dim=-1)
        top_scores, top_indices = scores.topk(k=min(20, scores.numel()))

        role_labels = list(getattr(adapter, "ROLE_LABELS", []))
        raw_hist = Counter(
            role_labels[int(role_id)] if int(role_id) < len(role_labels) else f"role_{int(role_id)}"
            for role_id in role_ids.detach().cpu()
        )

        threshold_summaries: dict[str, dict[str, Any]] = {}
        for threshold in (0.01, 0.03, 0.05, 0.10, 0.30, 0.50):
            detections = pipe._postprocess_detections(det_out, h_orig, w_orig, threshold, 0.5)
            threshold_summaries[str(threshold)] = {
                "count": len(detections),
                "role_hist": dict(Counter(det["role"] for det in detections)),
                "top_confidences": [round(float(det["confidence"]), 6) for det in detections[:10]],
            }

        top_rows = []
        for score, idx in zip(top_scores.detach().cpu(), top_indices.detach().cpu()):
            role_id = int(role_ids[idx])
            role_name = role_labels[role_id] if role_id < len(role_labels) else f"role_{role_id}"
            top_rows.append(
                {
                    "query": int(idx),
                    "score": round(float(score), 6),
                    "role": role_name,
                    "role_prob": round(float(role_probs[idx, role_id]), 6),
                }
            )

    return {
        "raw_query_role_hist": dict(raw_hist),
        "top_queries": top_rows,
        "thresholds": threshold_summaries,
    }


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_path = materialize_adapter()
    adapter = import_adapter(adapter_path)

    pipe = adapter.SoccerMasterPipeline(device=device)
    pipe.load()
    assert pipe._detection_head is not None

    head = pipe._detection_head
    role_labels = list(getattr(adapter, "ROLE_LABELS", []))
    per_image: list[dict[str, Any]] = []
    for path in image_paths():
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"OpenCV could not read {path}")
        summary = raw_detection_summary(adapter, pipe, frame)
        summary["image"] = path.name
        summary["shape"] = list(frame.shape)
        per_image.append(summary)

    threshold_role_total: Counter[str] = Counter()
    raw_role_total: Counter[str] = Counter()
    for item in per_image:
        raw_role_total.update(item["raw_query_role_hist"])
        threshold_role_total.update(item["thresholds"]["0.05"]["role_hist"])

    athlete_roles = {"player", "goalkeeper", "referee"}
    athlete_like_at_005 = sum(threshold_role_total.get(role, 0) for role in athlete_roles)
    result = {
        "ok": True,
        "ts": utc_now(),
        "device": device,
        "asset_repo": ASSET_REPO,
        "model_repo": MODEL_REPO,
        "role_labels": role_labels,
        "detection_head": {
            "num_classes": int(head.num_classes),
            "num_roles": int(head.num_roles),
            "num_queries": int(head.num_queries),
        },
        "num_images": len(per_image),
        "raw_role_total": dict(raw_role_total),
        "role_total_at_conf_0_05": dict(threshold_role_total),
        "athlete_like_at_conf_0_05": athlete_like_at_005,
        "per_image": per_image,
        "verdict": (
            "raw-athlete-output-present"
            if athlete_like_at_005 > 0
            else "no-athlete-output-at-0.05-debug-config"
        ),
    }

    out_dir = WORK / "result"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "soccermaster_wiring_probe.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_id = "soccermaster-wiring-" + utc_now().replace(":", "-").replace(".", "-")
    HfApi(token=TOKEN).upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=f"runs/{run_id}/soccermaster_wiring_probe.json",
        repo_id=MODEL_REPO,
        token=TOKEN,
    )
    print("AUTONOMY_RESULT " + json.dumps({**result, "run_id": run_id}, sort_keys=True))


if __name__ == "__main__":
    main()
