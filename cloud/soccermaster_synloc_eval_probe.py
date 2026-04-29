# /// script
# dependencies = [
#   "huggingface-hub",
#   "numpy<2",
#   "opencv-python-headless",
#   "torch",
#   "torchvision",
#   "transformers",
#   "sskit @ git+https://github.com/Spiideo/sskit.git",
#   "scipy",
#   "xtcocotools",
# ]
# ///
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from sskit import image_to_ground
from sskit.coco import LocSimCOCOeval
from xtcocotools.coco import COCO


ASSET_REPO = os.getenv("V2D_ASSET_REPO", "dmontgomery40/v2d-research-assets")
DATASET_REPO = os.environ["HF_DATASET_REPO"]
MODEL_REPO = os.environ["HF_MODEL_REPO"]
TOKEN = os.environ["HF_TOKEN"]
SPLIT = os.getenv("SYNLOC_SPLIT", "valid")
VERSION = os.getenv("SYNLOC_VERSION", "fullhd")
MAX_IMAGES = int(os.getenv("SOCCERMASTER_EVAL_MAX_IMAGES", "64"))
THRESHOLDS = [
    float(item)
    for item in os.getenv("SOCCERMASTER_THRESHOLDS", "0.01,0.03,0.05,0.1,0.2,0.3").split(",")
    if item.strip()
]
WORK = Path("/tmp/soccermaster_synloc_eval_probe")
OFFICIAL_ROLE_LABELS = ["ball", "goalkeeper", "other", "player", "referee", "none"]
ROLE_SETS = {
    "athlete": {"player", "goalkeeper", "referee"},
    "person_plus_other": {"player", "goalkeeper", "referee", "other"},
}


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


def download_first_asset(filenames: tuple[str, ...], local_dir: Path) -> Path:
    errors: list[str] = []
    for filename in filenames:
        try:
            return download_asset(filename, local_dir)
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError("None of the candidate HF asset paths exist: " + " | ".join(errors))


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
        src = download_first_asset(
            (
                f"models/soccermaster/{filename}",
                f"rondo_payload/models/soccermaster/{filename}",
            ),
            WORK,
        )
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
    module.ROLE_LABELS = list(OFFICIAL_ROLE_LABELS)
    return module


def extract_archives(cache_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(cache_dir.rglob("*.zip"))
    if not zips:
        raise RuntimeError(f"No cached zip files found in {cache_dir}")
    for archive in zips:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)


def find_annotation(root: Path, split: str) -> Path:
    names = {
        "valid": ["val.json", "valid.json", "validation.json"],
        "val": ["val.json", "valid.json", "validation.json"],
        "test": ["test.json"],
        "challenge": ["challenge.json"],
        "train": ["train.json"],
    }.get(split, [f"{split}.json"])
    for name in names:
        matches = sorted(root.rglob(name))
        for match in matches:
            if "annotation" in str(match).lower():
                return match
    raise RuntimeError(f"Could not find annotation for split={split} under {root}")


def image_path(root: Path, file_name: str) -> Path:
    candidates = [
        root / file_name,
        root / "SpiideoSynLoc" / file_name,
        root / "images" / file_name,
    ]
    candidates.extend(root.rglob(Path(file_name).name))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(file_name)


def evaluate(gt_path: Path, pred_path: Path, empty: bool) -> dict[str, float]:
    if empty:
        return {
            "map_locsim": 0.0,
            "precision_50": 0.0,
            "recall_50": 0.0,
            "f1_50": 0.0,
            "score_threshold": 0.0,
            "frame_accuracy": 0.0,
        }
    coco = COCO(str(gt_path))
    coco_det = coco.loadRes(str(pred_path))
    coco_eval = LocSimCOCOeval(coco, coco_det, "bbox")
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "map_locsim": float(coco_eval.stats[0]),
        "precision_50": float(coco_eval.stats[12]),
        "recall_50": float(coco_eval.stats[13]),
        "f1_50": float(coco_eval.stats[14]),
        "score_threshold": float(coco_eval.stats[15]),
        "frame_accuracy": float(coco_eval.stats[16]),
    }


def load_synloc_data() -> tuple[Path, list[dict[str, Any]], Path]:
    cache_dir = WORK / "hf-synloc-cache"
    data_root = WORK / "SoccerNet"
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=cache_dir,
        allow_patterns=[f"raw/{VERSION}/*.zip", f"raw/{VERSION}/manifest.json"],
        token=TOKEN,
    )
    extract_archives(cache_dir, data_root)
    gt_path = find_annotation(data_root, SPLIT)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt["images"][: MAX_IMAGES or None]
    return gt_path, images, data_root


def detections_for_image(adapter: Any, pipe: Any, frame_bgr: Any, thresholds: list[float]) -> dict[float, list[dict[str, Any]]]:
    h_orig, w_orig = frame_bgr.shape[:2]
    with torch.no_grad():
        tensor = pipe._preprocess(frame_bgr)
        spatial = pipe._extract_backbone_features(tensor)
        det_out = pipe._detection_head(spatial)
        return {
            threshold: pipe._postprocess_detections(det_out, h_orig, w_orig, threshold, 0.5)
            for threshold in thresholds
        }


def to_synloc_record(
    *,
    image: dict[str, Any],
    det: dict[str, Any],
    det_id: int,
) -> dict[str, Any]:
    width = float(image["width"])
    height = float(image["height"])
    x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
    x1 = min(max(x1, 0.0), width - 1.0)
    x2 = min(max(x2, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    y2 = min(max(y2, 0.0), height - 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    foot = np.array([[(x1 + x2) / 2.0, y2]], dtype=np.float32)
    normalized = ((foot - np.array([[(width - 1) / 2.0, (height - 1) / 2.0]], dtype=np.float32)) / width).astype(np.float32)
    bev = image_to_ground(image["camera_matrix"], image["undist_poly"], normalized)[0]
    return {
        "area": 0,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "category_id": 1,
        "id": det_id,
        "image_id": image["id"],
        "position_on_pitch": [float(bev[0]), float(bev[1]), 0.0],
        "score": float(det["confidence"]),
    }


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    gt_path, images, data_root = load_synloc_data()
    adapter_path = materialize_adapter()
    adapter = import_adapter(adapter_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = adapter.SoccerMasterPipeline(device=device)
    pipe.load()
    assert pipe._detection_head is not None

    per_threshold_records: dict[str, dict[float, list[dict[str, Any]]]] = {
        name: {threshold: [] for threshold in THRESHOLDS}
        for name in ROLE_SETS
    }
    role_hist_by_threshold: dict[float, Counter[str]] = {threshold: Counter() for threshold in THRESHOLDS}

    next_ids: dict[str, dict[float, int]] = {
        name: {threshold: 1 for threshold in THRESHOLDS}
        for name in ROLE_SETS
    }
    for image in images:
        frame = cv2.imread(str(image_path(data_root, image["file_name"])))
        if frame is None:
            raise RuntimeError(f"OpenCV could not read {image['file_name']}")
        detections_by_threshold = detections_for_image(adapter, pipe, frame, THRESHOLDS)
        for threshold, detections in detections_by_threshold.items():
            role_hist_by_threshold[threshold].update(det["role"] for det in detections)
            for role_set_name, roles in ROLE_SETS.items():
                records = per_threshold_records[role_set_name][threshold]
                for det in detections:
                    if det["role"] not in roles:
                        continue
                    det_id = next_ids[role_set_name][threshold]
                    records.append(to_synloc_record(image=image, det=det, det_id=det_id))
                    next_ids[role_set_name][threshold] += 1

    run_id = "soccermaster-synloc-probe-" + utc_now().replace(":", "-").replace(".", "-")
    out_dir = WORK / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluated: list[dict[str, Any]] = []
    for role_set_name, by_threshold in per_threshold_records.items():
        for threshold, records in by_threshold.items():
            tag = f"{role_set_name}_thr_{threshold:g}".replace(".", "p")
            pred_path = out_dir / f"results_{tag}.json"
            pred_path.write_text(json.dumps(records), encoding="utf-8")
            metrics = evaluate(gt_path, pred_path, empty=not records)
            evaluated.append(
                {
                    "role_set": role_set_name,
                    "threshold": threshold,
                    "num_detections": len(records),
                    "predictions": pred_path.name,
                    "metrics": metrics,
                }
            )

    best = max(evaluated, key=lambda item: item["metrics"]["map_locsim"])
    shutil.copy2(out_dir / best["predictions"], out_dir / "results.json")
    (out_dir / "metadata.json").write_text(
        json.dumps({"score_threshold": best["threshold"], "role_set": best["role_set"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "split": SPLIT,
        "version": VERSION,
        "max_images": MAX_IMAGES,
        "num_images": len(images),
        "device": device,
        "role_labels": OFFICIAL_ROLE_LABELS,
        "role_label_source": "official_soccermaster_data.soccernet_gsr_detection.role_mapping",
        "threshold_role_hist": {str(k): dict(v) for k, v in role_hist_by_threshold.items()},
        "evaluated": evaluated,
        "best": best,
        "metric": "mAP-LocSim",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    HfApi(token=TOKEN).upload_folder(
        repo_id=MODEL_REPO,
        repo_type="model",
        folder_path=out_dir,
        path_in_repo=f"runs/{run_id}",
        commit_message=f"Record {run_id}",
    )
    print("AUTONOMY_RESULT " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
