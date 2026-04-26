# /// script
# dependencies = [
#   "huggingface_hub>=0.24.0",
#   "torch",
#   "torchvision",
#   "ultralytics",
#   "opencv-python-headless",
#   "git+https://github.com/Spiideo/sskit.git",
#   "xtcocotools"
# ]
# ///
from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, snapshot_download
from sskit import image_to_ground
from sskit.coco import LocSimCOCOeval
from ultralytics import YOLO
from xtcocotools.coco import COCO


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def evaluate(gt_path: Path, pred_path: Path) -> dict[str, float]:
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


def main() -> None:
    required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    split = os.getenv("SYNLOC_SPLIT", "valid")
    version = os.getenv("SYNLOC_VERSION", "fullhd")
    max_images = int(os.getenv("BASELINE_MAX_IMAGES", "64"))
    imgsz = int(os.getenv("BASELINE_IMGSZ", "1280"))
    model_name = os.getenv("BASELINE_MODEL", "yolov8x.pt")

    cache_dir = Path("/tmp/hf-synloc-cache")
    data_root = Path("/tmp/SoccerNet")
    snapshot_download(
        repo_id=os.environ["HF_DATASET_REPO"],
        repo_type="dataset",
        local_dir=cache_dir,
        allow_patterns=[f"raw/{version}/*.zip", f"raw/{version}/manifest.json"],
        token=os.environ["HF_TOKEN"],
    )
    extract_archives(cache_dir, data_root)

    gt_path = find_annotation(data_root, split)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt["images"][: max_images or None]

    model = YOLO(model_name)
    results = []
    det_id = 1
    for image in images:
        path = image_path(data_root, image["file_name"])
        width = float(image["width"])
        height = float(image["height"])
        pred = model.predict(str(path), imgsz=imgsz, verbose=False)[0]
        names = pred.names
        boxes = pred.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, conf, cls):
            if names.get(int(class_id)) != "person":
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            point = np.array([[(x1 + x2) / 2.0, y2]], dtype=np.float32)
            normalized = ((point - np.array([[(width - 1) / 2.0, (height - 1) / 2.0]], dtype=np.float32)) / width).astype(np.float32)
            bev = image_to_ground(image["camera_matrix"], image["undist_poly"], normalized)[0]
            results.append(
                {
                    "area": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": 1,
                    "id": det_id,
                    "image_id": image["id"],
                    "position_on_pitch": [float(bev[0]), float(bev[1]), 0.0],
                    "score": float(score),
                }
            )
            det_id += 1

    run_id = f"baseline-yolo-{utc_now().replace(':', '-')}"
    out_dir = Path("/tmp") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "results.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    metrics = evaluate(gt_path, pred_path)
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "split": split,
        "version": version,
        "max_images": max_images,
        "model": model_name,
        "num_images": len(images),
        "num_detections": len(results),
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(
        repo_id=os.environ["HF_MODEL_REPO"],
        repo_type="model",
        folder_path=out_dir,
        path_in_repo=f"runs/{run_id}",
        commit_message=f"Record {run_id}",
    )
    print("AUTONOMY_RESULT " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
