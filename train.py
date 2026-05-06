#!/usr/bin/env python3
# /// script
# dependencies = [
#   "huggingface_hub>=0.24.0",
#   "ultralytics-opencv-headless>=8.4.29",
#   "torch",
#   "torchvision",
#   "transformers==4.50.0",
#   "rfdetr==1.2.1",
#   "sskit @ git+https://github.com/Spiideo/sskit.git",
#   "scipy",
#   "numpy<2",
#   "xtcocotools",
#   "pyyaml",
#   "pillow"
# ]
# ///
from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from sskit import image_to_ground
from sskit.coco import LocSimCOCOeval
from ultralytics import YOLO
from xtcocotools.coco import COCO


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


_AUTONOMY_RESULT_PRINTED = False


def emit_autonomy_result(summary: dict[str, Any]) -> None:
    global _AUTONOMY_RESULT_PRINTED
    if _AUTONOMY_RESULT_PRINTED:
        return
    print("AUTONOMY_RESULT " + json.dumps(json_safe(summary), sort_keys=True), flush=True)
    _AUTONOMY_RESULT_PRINTED = True


def parse_int_set(raw: str) -> set[int]:
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def xywh_to_xyxy(box: list[float] | tuple[float, float, float, float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - intersection
    return intersection / union if union > 0 else 0.0


def image_space_diagnostics(
    gt_boxes_by_image: dict[int, list[list[float]]],
    det_boxes_by_image: dict[int, list[list[float]]],
) -> dict[str, float | int]:
    gt_best_ious: list[float] = []
    det_best_ious: list[float] = []
    for image_id, gt_boxes in gt_boxes_by_image.items():
        det_boxes = det_boxes_by_image.get(image_id, [])
        for gt_box in gt_boxes:
            gt_best_ious.append(max((box_iou_xyxy(gt_box, det_box) for det_box in det_boxes), default=0.0))
        for det_box in det_boxes:
            det_best_ious.append(max((box_iou_xyxy(det_box, gt_box) for gt_box in gt_boxes), default=0.0))

    def rate(values: list[float], threshold: float) -> float:
        return float(sum(value >= threshold for value in values) / len(values)) if values else 0.0

    diagnostics: dict[str, float | int] = {
        "gt_boxes": sum(len(boxes) for boxes in gt_boxes_by_image.values()),
        "det_boxes": sum(len(boxes) for boxes in det_boxes_by_image.values()),
        "mean_best_iou_gt_to_det": float(np.mean(gt_best_ious)) if gt_best_ious else 0.0,
        "mean_best_iou_det_to_gt": float(np.mean(det_best_ious)) if det_best_ious else 0.0,
    }
    for threshold in (0.1, 0.3, 0.5):
        suffix = str(threshold).replace(".", "_")
        diagnostics[f"gt_recall_iou_{suffix}"] = rate(gt_best_ious, threshold)
        diagnostics[f"det_precision_iou_{suffix}"] = rate(det_best_ious, threshold)
    return diagnostics


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    repo: str
    filename: str
    athlete_class_ids: set[int]


DEFAULT_BASELINES = [
    # Current April 2026 football-specific YOLO26 lead.
    "football-yolo26l|mobadam/football-player-detection|player_detector.pt|1,3",
    # Public COCO person detector candidate source. Ultralytics resolves the
    # built-in weight file at runtime, so this does not require Hub storage.
    "yolo11n-coco-person|ultralytics|yolo11n.pt|0",
]

DEFAULT_TRANSFORMER_BASELINES = [
    "rtdetr-r18-coco-person|PekingU/rtdetr_r18vd|0",
]

DEFAULT_RFDETR_BASELINES = [
    "rfdetr-soccernet|julianzu9612/RFDETR-Soccernet|weights/checkpoint_best_regular.pth|1,2,3",
]


def parse_baselines(raw: str) -> list[BaselineSpec]:
    specs: list[BaselineSpec] = []
    for entry in [item.strip() for item in raw.split(";") if item.strip()]:
        parts = entry.split("|")
        if len(parts) != 4:
            raise RuntimeError(
                "Each YOLO_BASELINES entry must be name|repo|filename|class_ids"
            )
        specs.append(
            BaselineSpec(
                name=parts[0],
                repo=parts[1],
                filename=parts[2],
                athlete_class_ids=parse_int_set(parts[3]),
            )
        )
    return specs


def parse_transformer_baselines(raw: str) -> list[BaselineSpec]:
    specs: list[BaselineSpec] = []
    for entry in [item.strip() for item in raw.split(";") if item.strip()]:
        parts = entry.split("|")
        if len(parts) != 3:
            raise RuntimeError(
                "Each TRANSFORMER_BASELINES entry must be name|model_id|class_ids"
            )
        specs.append(
            BaselineSpec(
                name=parts[0],
                repo=parts[1],
                filename=parts[1],
                athlete_class_ids=parse_int_set(parts[2]),
            )
        )
    return specs


def parse_rfdetr_baselines(raw: str) -> list[BaselineSpec]:
    specs: list[BaselineSpec] = []
    for entry in [item.strip() for item in raw.split(";") if item.strip()]:
        parts = entry.split("|")
        if len(parts) != 4:
            raise RuntimeError(
                "Each RFDETR_BASELINES entry must be name|repo|checkpoint_path|class_ids"
            )
        specs.append(
            BaselineSpec(
                name=parts[0],
                repo=parts[1],
                filename=parts[2],
                athlete_class_ids=parse_int_set(parts[3]),
            )
        )
    return specs


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


def image_path(
    root: Path,
    file_name: str,
    *,
    expected_width: int | float | None = None,
    expected_height: int | float | None = None,
) -> Path:
    candidates = [
        root / file_name,
        root / "SpiideoSynLoc" / file_name,
        root / "images" / file_name,
    ]
    candidates.extend(root.rglob(Path(file_name).name))
    candidates = [candidate for candidate in candidates if candidate.exists() and candidate.is_file()]
    if expected_width is not None and expected_height is not None:
        expected_size = (int(round(float(expected_width))), int(round(float(expected_height))))
        mismatches: list[str] = []
        for candidate in candidates:
            with Image.open(candidate) as image:
                actual_size = (int(image.width), int(image.height))
            if actual_size == expected_size:
                return candidate
            mismatches.append(f"{candidate}={actual_size[0]}x{actual_size[1]}")
        if candidates:
            raise RuntimeError(
                f"No image candidate for {file_name!r} matched annotation size "
                f"{expected_size[0]}x{expected_size[1]}; candidates: {', '.join(mismatches)}"
            )
    for candidate in candidates:
        return candidate
    raise FileNotFoundError(file_name)


def image_path_for_record(root: Path, image: dict[str, Any]) -> Path:
    return image_path(
        root,
        image["file_name"],
        expected_width=image.get("width"),
        expected_height=image.get("height"),
    )


def image_path_and_scale_for_record(
    root: Path,
    image: dict[str, Any],
    *,
    coordinate_scale_mode: str,
) -> tuple[Path, float, float, tuple[int, int], tuple[int, int]]:
    expected_width = int(round(float(image["width"])))
    expected_height = int(round(float(image["height"])))
    if coordinate_scale_mode == "strict":
        path = image_path_for_record(root, image)
    elif coordinate_scale_mode == "actual_image":
        path = image_path(root, image["file_name"])
    else:
        raise RuntimeError("SYNLOC_COORD_SCALE_MODE must be one of: strict, actual_image")
    with Image.open(path) as opened:
        actual_width, actual_height = int(opened.width), int(opened.height)
    if actual_width <= 0 or actual_height <= 0:
        raise RuntimeError(f"Image dimensions must be positive for {path}: {actual_width}x{actual_height}")
    scale_x = actual_width / expected_width
    scale_y = actual_height / expected_height
    return path, scale_x, scale_y, (expected_width, expected_height), (actual_width, actual_height)


def scale_xywh(box: tuple[float, float, float, float], scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x * scale_x, y * scale_y, w * scale_x, h * scale_y


def scale_xy(point: tuple[float, float], scale_x: float, scale_y: float) -> tuple[float, float]:
    x, y = point
    return x * scale_x, y * scale_y


def load_synloc_data(version: str, patterns: list[str]) -> Path:
    required = ["HF_TOKEN", "HF_DATASET_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    cache_dir = Path("/tmp/hf-synloc-cache")
    data_root = Path("/tmp/SoccerNet")
    snapshot_download(
        repo_id=os.environ["HF_DATASET_REPO"],
        repo_type="dataset",
        local_dir=cache_dir,
        allow_patterns=patterns or [f"raw/{version}/*.zip", f"raw/{version}/manifest.json"],
        token=os.environ["HF_TOKEN"],
    )
    extract_archives(cache_dir, data_root)
    return data_root


def evaluate(gt_path: Path, pred_path: Path) -> dict[str, float]:
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    if not predictions:
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


def evaluate_keypoints(gt_path: Path, pred_path: Path, keypoint_index: int) -> dict[str, float]:
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    if not predictions:
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
    coco_eval.params.position_from_keypoint_index = keypoint_index
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


def download_model(spec: BaselineSpec, target_dir: Path) -> Path:
    if spec.repo in {"ultralytics", "__ultralytics__", "builtin"}:
        return Path(spec.filename)
    target_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=spec.repo,
            filename=spec.filename,
            local_dir=target_dir,
            token=os.getenv("HF_TOKEN"),
        )
    )


def predictions_for_model(
    *,
    model_path: Path,
    spec: BaselineSpec,
    data_root: Path,
    gt_path: Path,
    split: str,
    max_images: int,
    imgsz: int,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt["images"][: max_images or None]
    selected_image_ids = {int(image["id"]) for image in images}
    gt_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    for ann in gt.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in selected_image_ids:
            continue
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            continue
        gt_boxes_by_image.setdefault(image_id, []).append(xywh_to_xyxy(bbox))
    det_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    model = YOLO(str(model_path))
    model_class_names = {str(key): value for key, value in getattr(model, "names", {}).items()}

    results: list[dict[str, Any]] = []
    det_id = 1
    for image in images:
        image_id = int(image["id"])
        path = image_path_for_record(data_root, image)
        width = float(image["width"])
        height = float(image["height"])
        preds = model.predict(
            source=str(path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            device=0,
        )
        if not preds:
            continue
        boxes = preds[0].boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, scores, classes):
            if int(class_id) not in spec.athlete_class_ids:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            det_boxes_by_image.setdefault(image_id, []).append([x1, y1, x2, y2])
            point = np.array([[(x1 + x2) / 2.0, y2]], dtype=np.float32)
            center = np.array([[(width - 1) / 2.0, (height - 1) / 2.0]], dtype=np.float32)
            normalized = ((point - center) / width).astype(np.float32)
            bev = image_to_ground(image["camera_matrix"], image["undist_poly"], normalized)[0]
            results.append(
                {
                    "area": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": 1,
                    "id": det_id,
                    "image_id": image_id,
                    "position_on_pitch": [float(bev[0]), float(bev[1]), 0.0],
                    "score": float(score),
                }
            )
            det_id += 1

    run_id = f"{spec.name}-baseline-{utc_now().replace(':', '-')}"
    out_dir = Path("/tmp") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "results.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    metadata = {
        "score_threshold": conf,
        "model": spec.name,
        "repo": spec.repo,
        "filename": spec.filename,
        "athlete_class_ids": sorted(spec.athlete_class_ids),
        "model_class_names": model_class_names,
        "split": split,
        "max_images": max_images,
        "imgsz": imgsz,
        "iou": iou,
        "diagnostics": image_space_diagnostics(gt_boxes_by_image, det_boxes_by_image),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metrics = evaluate(gt_path, pred_path)
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "mode": "baseline",
        "split": split,
        "version": os.getenv("SYNLOC_VERSION", "fullhd"),
        "model": spec.name,
        "repo": spec.repo,
        "filename": spec.filename,
        "athlete_class_ids": sorted(spec.athlete_class_ids),
        "model_class_names": model_class_names,
        "max_images": max_images,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "num_images": len(images),
        "num_detections": len(results),
        "metrics": metrics,
        "diagnostics": metadata["diagnostics"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "out_dir": out_dir}


def predictions_for_transformer_model(
    *,
    spec: BaselineSpec,
    data_root: Path,
    gt_path: Path,
    split: str,
    max_images: int,
    threshold: float,
) -> dict[str, Any]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt["images"][: max_images or None]
    selected_image_ids = {int(image["id"]) for image in images}
    gt_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    for ann in gt.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in selected_image_ids:
            continue
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            continue
        gt_boxes_by_image.setdefault(image_id, []).append(xywh_to_xyxy(bbox))

    det_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    processor = AutoImageProcessor.from_pretrained(spec.repo)
    model = AutoModelForObjectDetection.from_pretrained(spec.repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model_class_names = {str(key): value for key, value in getattr(model.config, "id2label", {}).items()}

    results: list[dict[str, Any]] = []
    det_id = 1
    for image in images:
        image_id = int(image["id"])
        path = image_path_for_record(data_root, image)
        width = float(image["width"])
        height = float(image["height"])
        pil_image = Image.open(path).convert("RGB")
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]], device=device)
        processed = processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]
        boxes = processed["boxes"].detach().cpu().numpy()
        scores = processed["scores"].detach().cpu().numpy()
        labels = processed["labels"].detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, scores, labels):
            if int(class_id) not in spec.athlete_class_ids:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            det_boxes_by_image.setdefault(image_id, []).append([x1, y1, x2, y2])
            point = np.array([[(x1 + x2) / 2.0, y2]], dtype=np.float32)
            center = np.array([[(width - 1) / 2.0, (height - 1) / 2.0]], dtype=np.float32)
            normalized = ((point - center) / width).astype(np.float32)
            bev = image_to_ground(image["camera_matrix"], image["undist_poly"], normalized)[0]
            results.append(
                {
                    "area": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": 1,
                    "id": det_id,
                    "image_id": image_id,
                    "position_on_pitch": [float(bev[0]), float(bev[1]), 0.0],
                    "score": float(score),
                }
            )
            det_id += 1

    run_id = f"{spec.name}-transformer-baseline-{utc_now().replace(':', '-')}"
    out_dir = Path("/tmp") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "results.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    metadata = {
        "score_threshold": threshold,
        "model": spec.name,
        "repo": spec.repo,
        "athlete_class_ids": sorted(spec.athlete_class_ids),
        "model_class_names": model_class_names,
        "split": split,
        "max_images": max_images,
        "diagnostics": image_space_diagnostics(gt_boxes_by_image, det_boxes_by_image),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metrics = evaluate(gt_path, pred_path)
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "mode": "transformer_baseline",
        "split": split,
        "version": os.getenv("SYNLOC_VERSION", "fullhd"),
        "model": spec.name,
        "repo": spec.repo,
        "athlete_class_ids": sorted(spec.athlete_class_ids),
        "model_class_names": model_class_names,
        "max_images": max_images,
        "threshold": threshold,
        "num_images": len(images),
        "num_detections": len(results),
        "metrics": metrics,
        "diagnostics": metadata["diagnostics"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "out_dir": out_dir}


def predictions_for_rfdetr_model(
    *,
    spec: BaselineSpec,
    data_root: Path,
    gt_path: Path,
    split: str,
    max_images: int,
    threshold: float,
    model_class_name: str,
) -> dict[str, Any]:
    import rfdetr

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt["images"][: max_images or None]
    selected_image_ids = {int(image["id"]) for image in images}
    gt_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    for ann in gt.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in selected_image_ids:
            continue
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            continue
        gt_boxes_by_image.setdefault(image_id, []).append(xywh_to_xyxy(bbox))

    det_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    snapshot_dir = Path(
        snapshot_download(
            repo_id=spec.repo,
            allow_patterns=[spec.filename, "config.json", "model_metadata.json"],
        )
    )
    checkpoint_path = snapshot_dir / spec.filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RF-DETR checkpoint not found: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_class = getattr(rfdetr, model_class_name)
    print(
        f"Loading {spec.repo}:{spec.filename} with {model_class_name} on {device}",
        flush=True,
    )
    model = model_class(pretrain_weights=str(checkpoint_path))
    model.model.model.to(device)
    model.model.model.eval()
    model_class_names = {"0": "ball", "1": "player", "2": "referee", "3": "goalkeeper"}

    results: list[dict[str, Any]] = []
    det_id = 1
    for image in images:
        image_id = int(image["id"])
        path = image_path_for_record(data_root, image)
        width = float(image["width"])
        height = float(image["height"])
        pil_image = Image.open(path).convert("RGB")
        with torch.no_grad():
            detections = model.predict(pil_image, threshold=threshold)
        if detections is None or len(detections) == 0:
            continue
        for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            class_id = int(class_id)
            if class_id not in spec.athlete_class_ids:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            det_boxes_by_image.setdefault(image_id, []).append([x1, y1, x2, y2])
            point = np.array([[(x1 + x2) / 2.0, y2]], dtype=np.float32)
            center = np.array([[(width - 1) / 2.0, (height - 1) / 2.0]], dtype=np.float32)
            normalized = ((point - center) / width).astype(np.float32)
            bev = image_to_ground(image["camera_matrix"], image["undist_poly"], normalized)[0]
            results.append(
                {
                    "area": 0,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": 1,
                    "id": det_id,
                    "image_id": image_id,
                    "position_on_pitch": [float(bev[0]), float(bev[1]), 0.0],
                    "score": float(score),
                }
            )
            det_id += 1

    run_id = f"{spec.name}-rfdetr-baseline-{utc_now().replace(':', '-')}"
    out_dir = Path("/tmp") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "results.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    metadata = {
        "score_threshold": threshold,
        "model": spec.name,
        "repo": spec.repo,
        "checkpoint": spec.filename,
        "rfdetr_model_class": model_class_name,
        "athlete_class_ids": sorted(spec.athlete_class_ids),
        "model_class_names": model_class_names,
        "split": split,
        "max_images": max_images,
        "diagnostics": image_space_diagnostics(gt_boxes_by_image, det_boxes_by_image),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metrics = evaluate(gt_path, pred_path)
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "mode": "rfdetr_baseline",
        "split": split,
        "version": os.getenv("SYNLOC_VERSION", "fullhd"),
        "model": spec.name,
        "repo": spec.repo,
        "checkpoint": spec.filename,
        "rfdetr_model_class": model_class_name,
        "athlete_class_ids": sorted(spec.athlete_class_ids),
        "model_class_names": model_class_names,
        "max_images": max_images,
        "threshold": threshold,
        "num_images": len(images),
        "num_detections": len(results),
        "metrics": metrics,
        "diagnostics": metadata["diagnostics"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "out_dir": out_dir}


def annotation_keypoint(ann: dict[str, Any], index: int) -> tuple[float, float, float] | None:
    keypoints = ann.get("keypoints")
    if not keypoints:
        return None
    if all(isinstance(item, list) for item in keypoints):
        if len(keypoints) <= index or len(keypoints[index]) < 2:
            return None
        point = keypoints[index]
        visibility = float(point[2]) if len(point) > 2 else 1.0
        return float(point[0]), float(point[1]), visibility
    flat = [float(item) for item in keypoints]
    start = index * 3
    if len(flat) < start + 2:
        return None
    visibility = flat[start + 2] if len(flat) > start + 2 else 1.0
    return flat[start], flat[start + 1], visibility


def target_keypoint(
    ann: dict[str, Any],
    *,
    target: str,
    source_keypoint_index: int,
) -> tuple[float, float, float] | None:
    if target == "annotation":
        return annotation_keypoint(ann, source_keypoint_index)
    if target == "bbox_bottom_center":
        x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        if w <= 0 or h <= 0:
            return None
        return x + w / 2.0, y + h, 2.0
    raise RuntimeError(
        "SYNLOC_KEYPOINT_TARGET must be one of: annotation, bbox_bottom_center"
    )


def keypoint_diagnostics(
    gt_points_by_image: dict[int, list[tuple[float, float]]],
    pred_points_by_image: dict[int, list[tuple[float, float]]],
) -> dict[str, float | int]:
    gt_best: list[float] = []
    pred_best: list[float] = []
    for image_id, gt_points in gt_points_by_image.items():
        pred_points = pred_points_by_image.get(image_id, [])
        for gx, gy in gt_points:
            gt_best.append(
                min((((gx - px) ** 2 + (gy - py) ** 2) ** 0.5 for px, py in pred_points), default=float("inf"))
            )
        for px, py in pred_points:
            pred_best.append(
                min((((px - gx) ** 2 + (py - gy) ** 2) ** 0.5 for gx, gy in gt_points), default=float("inf"))
            )

    finite_gt = [value for value in gt_best if np.isfinite(value)]
    finite_pred = [value for value in pred_best if np.isfinite(value)]

    def rate(values: list[float], threshold: float) -> float:
        return float(sum(value <= threshold for value in values) / len(values)) if values else 0.0

    diagnostics: dict[str, float | int] = {
        "gt_keypoints": sum(len(points) for points in gt_points_by_image.values()),
        "pred_keypoints": sum(len(points) for points in pred_points_by_image.values()),
        "mean_best_px_gt_to_pred": float(np.mean(finite_gt)) if finite_gt else 0.0,
        "mean_best_px_pred_to_gt": float(np.mean(finite_pred)) if finite_pred else 0.0,
    }
    for threshold in (5.0, 10.0, 25.0, 50.0):
        suffix = str(int(threshold))
        diagnostics[f"gt_recall_px_{suffix}"] = rate(gt_best, threshold)
        diagnostics[f"pred_precision_px_{suffix}"] = rate(pred_best, threshold)
    return diagnostics


def nearest_point_distance(point: tuple[float, float], others: list[tuple[float, float]]) -> float | None:
    if not others:
        return None
    px, py = point
    return float(min(((px - ox) ** 2 + (py - oy) ** 2) ** 0.5 for ox, oy in others))


@dataclass(frozen=True)
class PointSample:
    image_path: Path
    image_id: int
    annotation_id: int | None
    bbox_xywh: tuple[float, float, float, float]
    point_xy: tuple[float, float]
    annotation_bbox_xywh: tuple[float, float, float, float]
    annotation_point_xy: tuple[float, float]
    coord_scale_x: float
    coord_scale_y: float


def build_point_samples(
    data_root: Path,
    gt_path: Path,
    *,
    max_images: int,
    source_keypoint_index: int,
    keypoint_target: str,
    coordinate_scale_mode: str,
) -> tuple[list[PointSample], list[dict[str, Any]]]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt["images"][: max_images or None]
    images_by_id = {int(image["id"]): image for image in images}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in gt.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id in images_by_id:
            annotations_by_image.setdefault(image_id, []).append(ann)

    samples: list[PointSample] = []
    skipped: list[dict[str, Any]] = []
    for image_id, image in images_by_id.items():
        path, scale_x, scale_y, annotation_size, actual_size = image_path_and_scale_for_record(
            data_root,
            image,
            coordinate_scale_mode=coordinate_scale_mode,
        )
        width = float(image["width"])
        height = float(image["height"])
        for ann in annotations_by_image.get(image_id, []):
            x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            point = target_keypoint(
                ann,
                target=keypoint_target,
                source_keypoint_index=source_keypoint_index,
            )
            if w <= 1 or h <= 1 or point is None:
                skipped.append({"annotation_id": ann.get("id"), "reason": "missing_box_or_point"})
                continue
            px, py, visibility = point
            if visibility <= 0 or not np.isfinite([x, y, w, h, px, py]).all():
                skipped.append({"annotation_id": ann.get("id"), "reason": "invalid_point"})
                continue
            if not (0 <= px < width and 0 <= py < height):
                skipped.append({"annotation_id": ann.get("id"), "reason": "point_outside_image"})
                continue
            samples.append(
                PointSample(
                    image_path=path,
                    image_id=image_id,
                    annotation_id=ann.get("id"),
                    bbox_xywh=scale_xywh((x, y, w, h), scale_x, scale_y),
                    point_xy=scale_xy((px, py), scale_x, scale_y),
                    annotation_bbox_xywh=(x, y, w, h),
                    annotation_point_xy=(px, py),
                    coord_scale_x=scale_x,
                    coord_scale_y=scale_y,
                )
            )
    return samples, skipped


def crop_bounds(
    bbox_xywh: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got width={width} height={height}")
    x, y, w, h = bbox_xywh
    pad_x = w * padding
    pad_y = h * padding
    left = min(max(0, int(np.floor(x - pad_x))), width - 1)
    top = min(max(0, int(np.floor(y - pad_y))), height - 1)
    right = min(width, int(np.ceil(x + w + pad_x)))
    bottom = min(height, int(np.ceil(y + h + pad_y)))
    right = min(width, max(left + 1, right))
    bottom = min(height, max(top + 1, bottom))
    return left, top, right, bottom


def jitter_bbox_xywh(
    bbox_xywh: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    center_frac: float,
    scale_frac: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got width={image_width} height={image_height}"
        )
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        raise ValueError(f"Box dimensions must be positive, got bbox={bbox_xywh}")
    cx = x + w / 2.0 + float(rng.normal(0.0, max(0.0, center_frac))) * w
    cy = y + h / 2.0 + float(rng.normal(0.0, max(0.0, center_frac))) * h
    scale_w = float(np.exp(rng.normal(0.0, max(0.0, scale_frac))))
    scale_h = float(np.exp(rng.normal(0.0, max(0.0, scale_frac))))
    new_w = min(float(image_width), max(1.0, w * scale_w))
    new_h = min(float(image_height), max(1.0, h * scale_h))
    left = min(max(0.0, cx - new_w / 2.0), float(image_width) - 1.0)
    top = min(max(0.0, cy - new_h / 2.0), float(image_height) - 1.0)
    right = min(float(image_width), max(left + 1.0, cx + new_w / 2.0))
    bottom = min(float(image_height), max(top + 1.0, cy + new_h / 2.0))
    return left, top, right - left, bottom - top


class PointCropDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[PointSample], *, image_size: int, crop_padding: float) -> None:
        self.samples = samples
        self.image_size = image_size
        self.crop_padding = crop_padding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb = image.convert("RGB")
            left, top, right, bottom = crop_bounds(sample.bbox_xywh, rgb.width, rgb.height, self.crop_padding)
            crop = rgb.crop((left, top, right, bottom)).resize((self.image_size, self.image_size))
        array = np.asarray(crop, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        px, py = sample.point_xy
        target = torch.tensor(
            [
                float(np.clip((px - left) / max(1, right - left), 0.0, 1.0)),
                float(np.clip((py - top) / max(1, bottom - top), 0.0, 1.0)),
            ],
            dtype=torch.float32,
        )
        return tensor, target


class TinyPointRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def predict_point_from_crop(
    model: nn.Module,
    sample: PointSample,
    *,
    image_size: int,
    crop_padding: float,
    device: torch.device,
) -> tuple[float, float]:
    with Image.open(sample.image_path) as image:
        rgb = image.convert("RGB")
        left, top, right, bottom = crop_bounds(sample.bbox_xywh, rgb.width, rgb.height, crop_padding)
        crop = rgb.crop((left, top, right, bottom)).resize((image_size, image_size))
    array = np.asarray(crop, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).detach().cpu().numpy()[0]
    px = left + float(pred[0]) * max(1, right - left)
    py = top + float(pred[1]) * max(1, bottom - top)
    return px, py


def point_to_annotation_scale(sample: PointSample, point_xy: tuple[float, float]) -> tuple[float, float]:
    if sample.coord_scale_x <= 0 or sample.coord_scale_y <= 0:
        raise ValueError(f"Invalid coordinate scale for image_id={sample.image_id}: {sample.coord_scale_x}, {sample.coord_scale_y}")
    return point_xy[0] / sample.coord_scale_x, point_xy[1] / sample.coord_scale_y


def box_to_annotation_scale(sample: PointSample, box_xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if sample.coord_scale_x <= 0 or sample.coord_scale_y <= 0:
        raise ValueError(f"Invalid coordinate scale for image_id={sample.image_id}: {sample.coord_scale_x}, {sample.coord_scale_y}")
    x, y, w, h = box_xywh
    return x / sample.coord_scale_x, y / sample.coord_scale_y, w / sample.coord_scale_x, h / sample.coord_scale_y


def evaluate_point_regressor(
    *,
    model: nn.Module,
    samples: list[PointSample],
    gt_path: Path,
    image_size: int,
    crop_padding: float,
    device: torch.device,
    audit_sample_images: int,
) -> dict[str, Any]:
    model.eval()
    results: list[dict[str, Any]] = []
    gt_boxes_by_image: dict[int, list[list[float]]] = {}
    det_boxes_by_image: dict[int, list[list[float]]] = {}
    gt_points_by_image: dict[int, list[tuple[float, float]]] = {}
    pred_points_by_image: dict[int, list[tuple[float, float]]] = {}
    audit_examples: list[dict[str, Any]] = []
    det_id = 1
    for sample in samples:
        x, y, w, h = sample.annotation_bbox_xywh
        px_actual, py_actual = predict_point_from_crop(
            model,
            sample,
            image_size=image_size,
            crop_padding=crop_padding,
            device=device,
        )
        px, py = point_to_annotation_scale(sample, (px_actual, py_actual))
        gt_box = xywh_to_xyxy([x, y, w, h])
        gt_point = sample.annotation_point_xy
        gt_boxes_by_image.setdefault(sample.image_id, []).append(gt_box)
        det_boxes_by_image.setdefault(sample.image_id, []).append(gt_box)
        gt_points_by_image.setdefault(sample.image_id, []).append(gt_point)
        pred_points_by_image.setdefault(sample.image_id, []).append((px, py))
        error_px = nearest_point_distance((px, py), [gt_point])
        if audit_sample_images > 0 and len(audit_examples) < audit_sample_images:
            audit_examples.append(
                {
                    "image_id": sample.image_id,
                    "annotation_id": sample.annotation_id,
                    "bbox_xywh": [x, y, w, h],
                    "target_keypoint": [gt_point[0], gt_point[1], 2.0],
                    "predicted_keypoint": [px, py, 2.0],
                    "predicted_keypoint_actual_image": [px_actual, py_actual, 2.0],
                    "coord_scale": [sample.coord_scale_x, sample.coord_scale_y],
                    "point_error_px": error_px,
                }
            )
        results.append(
            {
                "area": 0,
                "bbox": [x, y, w, h],
                "category_id": 1,
                "id": det_id,
                "image_id": sample.image_id,
                "keypoints": [px, py, 2.0],
                "score": 1.0,
            }
        )
        det_id += 1

    run_id = f"synloc-point-regressor-eval-{utc_now().replace(':', '-')}"
    out_dir = Path("/tmp") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "results.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    metrics = evaluate_keypoints(gt_path, pred_path, 0)
    diagnostics = {
        "boxes": image_space_diagnostics(gt_boxes_by_image, det_boxes_by_image),
        "keypoints": keypoint_diagnostics(gt_points_by_image, pred_points_by_image),
    }
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "mode": "point_regressor_eval",
        "oracle_candidate_boxes": True,
        "position_from_keypoint_index": 0,
        "num_predictions": len(results),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "audit_examples": audit_examples,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "out_dir": out_dir}


def evaluate_point_regressor_on_jittered_candidates(
    *,
    model: nn.Module,
    samples: list[PointSample],
    gt_path: Path,
    image_size: int,
    crop_padding: float,
    device: torch.device,
    audit_sample_images: int,
    center_frac: float,
    scale_frac: float,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    gt_boxes_by_image: dict[int, list[list[float]]] = {}
    det_boxes_by_image: dict[int, list[list[float]]] = {}
    gt_points_by_image: dict[int, list[tuple[float, float]]] = {}
    pred_points_by_image: dict[int, list[tuple[float, float]]] = {}
    audit_examples: list[dict[str, Any]] = []
    det_id = 1
    for sample in samples:
        with Image.open(sample.image_path) as image:
            width = image.width
            height = image.height
        x, y, w, h = sample.annotation_bbox_xywh
        jx, jy, jw, jh = jitter_bbox_xywh(
            sample.bbox_xywh,
            image_width=width,
            image_height=height,
            center_frac=center_frac,
            scale_frac=scale_frac,
            rng=rng,
        )
        jittered_sample = PointSample(
            image_path=sample.image_path,
            image_id=sample.image_id,
            annotation_id=sample.annotation_id,
            bbox_xywh=(jx, jy, jw, jh),
            point_xy=sample.point_xy,
            annotation_bbox_xywh=box_to_annotation_scale(sample, (jx, jy, jw, jh)),
            annotation_point_xy=sample.annotation_point_xy,
            coord_scale_x=sample.coord_scale_x,
            coord_scale_y=sample.coord_scale_y,
        )
        px_actual, py_actual = predict_point_from_crop(
            model,
            jittered_sample,
            image_size=image_size,
            crop_padding=crop_padding,
            device=device,
        )
        gt_box = xywh_to_xyxy([x, y, w, h])
        jx_ann, jy_ann, jw_ann, jh_ann = jittered_sample.annotation_bbox_xywh
        det_box = xywh_to_xyxy([jx_ann, jy_ann, jw_ann, jh_ann])
        gt_point = sample.annotation_point_xy
        px, py = point_to_annotation_scale(sample, (px_actual, py_actual))
        gt_boxes_by_image.setdefault(sample.image_id, []).append(gt_box)
        det_boxes_by_image.setdefault(sample.image_id, []).append(det_box)
        gt_points_by_image.setdefault(sample.image_id, []).append(gt_point)
        pred_points_by_image.setdefault(sample.image_id, []).append((px, py))
        if audit_sample_images > 0 and len(audit_examples) < audit_sample_images:
            audit_examples.append(
                {
                    "image_id": sample.image_id,
                    "annotation_id": sample.annotation_id,
                    "gt_bbox_xywh": [x, y, w, h],
                    "jittered_bbox_xywh": [jx_ann, jy_ann, jw_ann, jh_ann],
                    "jittered_bbox_actual_image_xywh": [jx, jy, jw, jh],
                    "candidate_iou": box_iou_xyxy(gt_box, det_box),
                    "target_keypoint": [gt_point[0], gt_point[1], 2.0],
                    "predicted_keypoint": [px, py, 2.0],
                    "predicted_keypoint_actual_image": [px_actual, py_actual, 2.0],
                    "coord_scale": [sample.coord_scale_x, sample.coord_scale_y],
                    "point_error_px": nearest_point_distance((px, py), [gt_point]),
                }
            )
        results.append(
            {
                "area": 0,
                "bbox": [jx_ann, jy_ann, jw_ann, jh_ann],
                "category_id": 1,
                "id": det_id,
                "image_id": sample.image_id,
                "keypoints": [px, py, 2.0],
                "score": 1.0,
            }
        )
        det_id += 1

    run_id = f"synloc-point-regressor-jittered-eval-{utc_now().replace(':', '-')}"
    out_dir = Path("/tmp") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "results.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    metrics = evaluate_keypoints(gt_path, pred_path, 0)
    diagnostics = {
        "boxes": image_space_diagnostics(gt_boxes_by_image, det_boxes_by_image),
        "keypoints": keypoint_diagnostics(gt_points_by_image, pred_points_by_image),
    }
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "mode": "point_regressor_jittered_candidate_eval",
        "oracle_candidate_boxes": False,
        "jittered_gt_candidate_boxes": True,
        "position_from_keypoint_index": 0,
        "num_predictions": len(results),
        "center_frac": center_frac,
        "scale_frac": scale_frac,
        "seed": seed,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "audit_examples": audit_examples,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "out_dir": out_dir}


def evaluate_point_regressor_on_yolo_candidates(
    *,
    point_model: nn.Module,
    detector_model_path: Path,
    detector_spec: BaselineSpec,
    data_root: Path,
    gt_path: Path,
    max_images: int,
    detector_imgsz: int,
    detector_conf: float,
    detector_iou: float,
    max_detections_per_image: int,
    image_size: int,
    crop_padding: float,
    device: torch.device,
    keypoint_target: str,
    source_keypoint_index: int,
    audit_sample_images: int,
    coordinate_scale_mode: str,
) -> dict[str, Any]:
    point_model.eval()
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt["images"][: max_images or None]
    selected_image_ids = {int(image["id"]) for image in images}
    gt_annotations_by_image: dict[int, list[dict[str, Any]]] = {
        image_id: [] for image_id in selected_image_ids
    }
    gt_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    gt_points_by_image: dict[int, list[tuple[float, float]]] = {image_id: [] for image_id in selected_image_ids}
    for ann in gt.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in selected_image_ids:
            continue
        gt_annotations_by_image.setdefault(image_id, []).append(ann)
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) == 4 and float(bbox[2]) > 0 and float(bbox[3]) > 0:
            gt_boxes_by_image.setdefault(image_id, []).append(xywh_to_xyxy(bbox))
        point = target_keypoint(
            ann,
            target=keypoint_target,
            source_keypoint_index=source_keypoint_index,
        )
        if point is not None and point[2] > 0 and np.isfinite([point[0], point[1]]).all():
            gt_points_by_image.setdefault(image_id, []).append((float(point[0]), float(point[1])))

    det_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    pred_points_by_image: dict[int, list[tuple[float, float]]] = {image_id: [] for image_id in selected_image_ids}
    detector = YOLO(str(detector_model_path))
    detector_device: int | str = 0 if torch.cuda.is_available() else "cpu"

    results: list[dict[str, Any]] = []
    audit_examples: list[dict[str, Any]] = []
    det_id = 1
    for image in images:
        image_id = int(image["id"])
        path, scale_x, scale_y, _annotation_size, _actual_size = image_path_and_scale_for_record(
            data_root,
            image,
            coordinate_scale_mode=coordinate_scale_mode,
        )
        preds = detector.predict(
            source=str(path),
            imgsz=detector_imgsz,
            conf=detector_conf,
            iou=detector_iou,
            verbose=False,
            device=detector_device,
        )
        if not preds or preds[0].boxes is None:
            continue
        boxes = preds[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        candidates: list[dict[str, Any]] = []
        raw_class_counts: dict[int, int] = {}
        for box, score, class_id in zip(xyxy, scores, classes):
            raw_class_counts[int(class_id)] = raw_class_counts.get(int(class_id), 0) + 1
            if int(class_id) not in detector_spec.athlete_class_ids:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 <= x1 or y2 <= y1:
                continue
            sample = PointSample(
                image_path=path,
                image_id=image_id,
                annotation_id=None,
                bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                point_xy=(0.0, 0.0),
                annotation_bbox_xywh=(x1 / scale_x, y1 / scale_y, (x2 - x1) / scale_x, (y2 - y1) / scale_y),
                annotation_point_xy=(0.0, 0.0),
                coord_scale_x=scale_x,
                coord_scale_y=scale_y,
            )
            px_actual, py_actual = predict_point_from_crop(
                point_model,
                sample,
                image_size=image_size,
                crop_padding=crop_padding,
                device=device,
            )
            px, py = point_to_annotation_scale(sample, (px_actual, py_actual))
            ax, ay, aw, ah = sample.annotation_bbox_xywh
            candidates.append(
                {
                    "bbox": [ax, ay, aw, ah],
                    "xyxy": [ax, ay, ax + aw, ay + ah],
                    "bbox_actual_image": [x1, y1, x2 - x1, y2 - y1],
                    "keypoints": [px, py, 2.0],
                    "keypoints_actual_image": [px_actual, py_actual, 2.0],
                    "point": (px, py),
                    "score": float(score),
                    "class_id": int(class_id),
                    "coord_scale": [scale_x, scale_y],
                }
            )
        if max_detections_per_image > 0:
            candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_detections_per_image]
        if audit_sample_images > 0 and len(audit_examples) < audit_sample_images:
            gt_points = gt_points_by_image.get(image_id, [])
            gt_rows = []
            for ann in gt_annotations_by_image.get(image_id, []):
                target_point = target_keypoint(
                    ann,
                    target=keypoint_target,
                    source_keypoint_index=source_keypoint_index,
                )
                gt_rows.append(
                    {
                        "annotation_id": ann.get("id"),
                        "bbox_xywh": ann.get("bbox"),
                        "target_keypoint": None
                        if target_point is None
                        else {"x": target_point[0], "y": target_point[1], "visibility": target_point[2]},
                        "category_id": ann.get("category_id"),
                    }
                )
            pred_rows = []
            for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True)[:10]:
                pred_rows.append(
                    {
                        "bbox_xywh": candidate["bbox"],
                        "keypoints": candidate["keypoints"],
                        "score": candidate["score"],
                        "class_id": candidate["class_id"],
                        "nearest_gt_point_px": nearest_point_distance(candidate["point"], gt_points),
                    }
                )
            audit_examples.append(
                {
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "raw_class_counts": raw_class_counts,
                    "gt": gt_rows,
                    "predictions_top10_after_filter": pred_rows,
                }
            )
        for candidate in candidates:
            det_boxes_by_image.setdefault(image_id, []).append(candidate["xyxy"])
            pred_points_by_image.setdefault(image_id, []).append(candidate["point"])
            results.append(
                {
                    "area": 0,
                    "bbox": candidate["bbox"],
                    "category_id": 1,
                    "id": det_id,
                    "image_id": image_id,
                    "keypoints": candidate["keypoints"],
                    "score": candidate["score"],
                }
            )
            det_id += 1

    run_id = f"synloc-point-regressor-yolo-eval-{utc_now().replace(':', '-')}"
    out_dir = Path("/tmp") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "results.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    metrics = evaluate_keypoints(gt_path, pred_path, 0)
    diagnostics = {
        "boxes": image_space_diagnostics(gt_boxes_by_image, det_boxes_by_image),
        "keypoints": keypoint_diagnostics(gt_points_by_image, pred_points_by_image),
    }
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "mode": "point_regressor_yolo_candidate_eval",
        "oracle_candidate_boxes": False,
        "detector": detector_spec.name,
        "detector_repo": detector_spec.repo,
        "detector_filename": detector_spec.filename,
        "athlete_class_ids": sorted(detector_spec.athlete_class_ids),
        "position_from_keypoint_index": 0,
        "num_images": len(images),
        "num_predictions": len(results),
        "detector_imgsz": detector_imgsz,
        "detector_conf": detector_conf,
        "detector_iou": detector_iou,
        "max_detections_per_image": max_detections_per_image,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "audit_examples": audit_examples,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "out_dir": out_dir}


def keypoint_candidate_score(mode: str, box_score: float, keypoint_score: float) -> float:
    if mode == "combined":
        return box_score * keypoint_score
    if mode == "box":
        return box_score
    if mode == "keypoint":
        return keypoint_score
    raise RuntimeError(
        "YOLO_KEYPOINT_SCORE_MODE must be one of: combined, box, keypoint"
    )


def keypoint_score_modes(raw: str) -> list[str]:
    aliases = {
        "all": ["combined", "box", "keypoint"],
        "matrix": ["combined", "box", "keypoint"],
    }
    mode = raw.strip().lower()
    modes = aliases.get(mode, [item.strip().lower() for item in mode.split(",") if item.strip()])
    if not modes:
        modes = ["combined"]
    allowed = {"combined", "box", "keypoint"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise RuntimeError(
            "YOLO_KEYPOINT_SCORE_MODE must be combined, box, keypoint, "
            f"all, matrix, or a comma list of those modes; got {unknown}"
        )
    return list(dict.fromkeys(modes))


def make_yolo_keypoint_dataset(
    data_root: Path,
    train_gt: Path,
    val_gt: Path,
    *,
    train_max: int,
    val_max: int,
    source_keypoint_index: int,
    keypoint_target: str,
) -> tuple[Path, dict[str, Any]]:
    dataset = Path("/tmp/synloc-yolo-keypoint-dataset")
    if dataset.exists():
        shutil.rmtree(dataset)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    def convert(gt_path: Path, split: str, max_images: int) -> dict[str, int]:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in gt.get("annotations", []):
            annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        stats = {"images": 0, "annotations": 0, "labels": 0, "skipped_keypoints": 0}
        for image in gt["images"][: max_images or None]:
            src = image_path_for_record(data_root, image)
            stem = f"{int(image['id']):08d}_{Path(image['file_name']).stem}"
            image_target = dataset / "images" / split / f"{stem}{src.suffix.lower() or '.jpg'}"
            label_target = dataset / "labels" / split / f"{stem}.txt"
            try:
                image_target.symlink_to(src)
            except OSError:
                shutil.copy2(src, image_target)
            width = float(image["width"])
            height = float(image["height"])
            lines: list[str] = []
            for ann in annotations_by_image.get(int(image["id"]), []):
                stats["annotations"] += 1
                x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
                point = target_keypoint(
                    ann,
                    target=keypoint_target,
                    source_keypoint_index=source_keypoint_index,
                )
                if w <= 0 or h <= 0 or point is None:
                    stats["skipped_keypoints"] += 1
                    continue
                px, py, visibility = point
                if visibility <= 0 or not np.isfinite([px, py]).all() or not (0 <= px < width and 0 <= py < height):
                    stats["skipped_keypoints"] += 1
                    continue
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                line = (
                    f"0 {cx:.8f} {cy:.8f} {w / width:.8f} {h / height:.8f} "
                    f"{px / width:.8f} {py / height:.8f} 2"
                )
                lines.append(line)
                stats["labels"] += 1
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            stats["images"] += 1
        return stats

    train_stats = convert(train_gt, "train", train_max)
    val_stats = convert(val_gt, "val", val_max)
    data_yaml = {
        "path": str(dataset),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "athlete"},
        "kpt_shape": [1, 3],
        "flip_idx": [0],
    }
    (dataset / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    manifest = {
        "keypoint_target": keypoint_target,
        "source_keypoint_index": source_keypoint_index,
        "output_keypoint_index": 0,
        "train": train_stats,
        "val": val_stats,
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return dataset, manifest


def predictions_for_keypoint_model(
    *,
    model_path: Path,
    data_root: Path,
    gt_path: Path,
    split: str,
    max_images: int,
    imgsz: int,
    conf: float,
    iou: float,
    source_keypoint_index: int,
    keypoint_target: str,
    max_detections_per_image: int,
    audit_sample_images: int,
    score_mode: str,
) -> dict[str, Any]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt["images"][: max_images or None]
    selected_image_ids = {int(image["id"]) for image in images}
    gt_annotations_by_image: dict[int, list[dict[str, Any]]] = {
        image_id: [] for image_id in selected_image_ids
    }
    gt_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    gt_points_by_image: dict[int, list[tuple[float, float]]] = {image_id: [] for image_id in selected_image_ids}
    for ann in gt.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in selected_image_ids:
            continue
        gt_annotations_by_image.setdefault(image_id, []).append(ann)
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) == 4 and float(bbox[2]) > 0 and float(bbox[3]) > 0:
            gt_boxes_by_image.setdefault(image_id, []).append(xywh_to_xyxy(bbox))
        point = target_keypoint(
            ann,
            target=keypoint_target,
            source_keypoint_index=source_keypoint_index,
        )
        if point is not None and point[2] > 0 and np.isfinite([point[0], point[1]]).all():
            gt_points_by_image.setdefault(image_id, []).append((float(point[0]), float(point[1])))

    det_boxes_by_image: dict[int, list[list[float]]] = {image_id: [] for image_id in selected_image_ids}
    pred_points_by_image: dict[int, list[tuple[float, float]]] = {image_id: [] for image_id in selected_image_ids}
    model = YOLO(str(model_path))

    results: list[dict[str, Any]] = []
    audit_examples: list[dict[str, Any]] = []
    det_id = 1
    for image in images:
        image_id = int(image["id"])
        path = image_path_for_record(data_root, image)
        preds = model.predict(
            source=str(path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            device=0,
        )
        if not preds:
            continue
        boxes = preds[0].boxes
        keypoints = preds[0].keypoints
        if boxes is None or keypoints is None:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        kxy = keypoints.xy.detach().cpu().numpy()
        keypoint_conf = getattr(keypoints, "conf", None)
        kconf = None if keypoint_conf is None else keypoint_conf.detach().cpu().numpy()
        candidates: list[dict[str, Any]] = []
        raw_class_counts: dict[int, int] = {}
        for det_index, (box, score, class_id) in enumerate(zip(xyxy, scores, classes)):
            raw_class_counts[int(class_id)] = raw_class_counts.get(int(class_id), 0) + 1
            if int(class_id) != 0 or det_index >= len(kxy) or len(kxy[det_index]) < 1:
                continue
            kx, ky = [float(v) for v in kxy[det_index][0]]
            if not np.isfinite([kx, ky]).all():
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            keypoint_score = 1.0
            if kconf is not None and det_index < len(kconf) and len(kconf[det_index]) > 0:
                keypoint_score = max(float(kconf[det_index][0]), 1e-6)
            box_score = float(score)
            candidate_score = keypoint_candidate_score(score_mode, box_score, keypoint_score)
            candidates.append(
                {
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "keypoints": [kx, ky, 2.0],
                    "score": candidate_score,
                    "xyxy": [x1, y1, x2, y2],
                    "point": (kx, ky),
                    "box_score": box_score,
                    "keypoint_score": keypoint_score,
                    "class_id": int(class_id),
                }
            )
        if max_detections_per_image > 0:
            candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:max_detections_per_image]
        if audit_sample_images > 0 and len(audit_examples) < audit_sample_images:
            gt_points = gt_points_by_image.get(image_id, [])
            gt_rows = []
            for ann in gt_annotations_by_image.get(image_id, []):
                source_point = annotation_keypoint(ann, source_keypoint_index)
                target_point = target_keypoint(
                    ann,
                    target=keypoint_target,
                    source_keypoint_index=source_keypoint_index,
                )
                gt_rows.append(
                    {
                        "annotation_id": ann.get("id"),
                        "bbox_xywh": ann.get("bbox"),
                        "source_keypoint": None
                        if source_point is None
                        else {"x": source_point[0], "y": source_point[1], "visibility": source_point[2]},
                        "target_keypoint": None
                        if target_point is None
                        else {"x": target_point[0], "y": target_point[1], "visibility": target_point[2]},
                        "category_id": ann.get("category_id"),
                    }
                )
            pred_rows = []
            for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True)[:10]:
                point = candidate["point"]
                pred_rows.append(
                    {
                        "bbox_xywh": candidate["bbox"],
                        "keypoints": candidate["keypoints"],
                        "score": candidate["score"],
                        "box_score": candidate["box_score"],
                        "keypoint_score": candidate["keypoint_score"],
                        "class_id": candidate["class_id"],
                        "nearest_gt_point_px": nearest_point_distance(point, gt_points),
                    }
                )
            audit_examples.append(
                {
                    "image_id": image_id,
                    "file_name": image["file_name"],
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "raw_class_counts": raw_class_counts,
                    "gt": gt_rows,
                    "predictions_top10_after_filter": pred_rows,
                }
            )
        for candidate in candidates:
            det_boxes_by_image.setdefault(image_id, []).append(candidate["xyxy"])
            pred_points_by_image.setdefault(image_id, []).append(candidate["point"])
            results.append(
                {
                    "area": 0,
                    "bbox": candidate["bbox"],
                    "category_id": 1,
                    "id": det_id,
                    "image_id": image_id,
                    "keypoints": candidate["keypoints"],
                    "score": candidate["score"],
                }
            )
            det_id += 1

    run_id = f"synloc-keypoint-eval-{utc_now().replace(':', '-')}"
    out_dir = Path("/tmp") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "results.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    metrics = evaluate_keypoints(gt_path, pred_path, 0)
    diagnostics = {
        "boxes": image_space_diagnostics(gt_boxes_by_image, det_boxes_by_image),
        "keypoints": keypoint_diagnostics(gt_points_by_image, pred_points_by_image),
    }
    class_names = json_safe(getattr(model, "names", {}))
    metadata = {
        "score_threshold": metrics["score_threshold"],
        "position_from_keypoint_index": 0,
        "split": split,
        "max_images": max_images,
        "imgsz": imgsz,
        "iou": iou,
        "source_keypoint_index": source_keypoint_index,
        "keypoint_target": keypoint_target,
        "max_detections_per_image": max_detections_per_image,
        "audit_sample_images": audit_sample_images,
        "score_mode": score_mode,
        "model_class_names": class_names,
        "diagnostics": diagnostics,
        "audit_examples": audit_examples,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "mode": "keypoint",
        "split": split,
        "version": os.getenv("SYNLOC_VERSION", "fullhd"),
        "max_images": max_images,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_detections_per_image": max_detections_per_image,
        "num_images": len(images),
        "num_detections": len(results),
        "position_from_keypoint_index": 0,
        "source_keypoint_index": source_keypoint_index,
        "keypoint_target": keypoint_target,
        "audit_sample_images": audit_sample_images,
        "score_mode": score_mode,
        "model_class_names": class_names,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "audit_examples": audit_examples,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary, "out_dir": out_dir}


def make_yolo_dataset(data_root: Path, train_gt: Path, val_gt: Path, *, train_max: int, val_max: int) -> Path:
    dataset = Path("/tmp/synloc-yolo-dataset")
    if dataset.exists():
        shutil.rmtree(dataset)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    def convert(gt_path: Path, split: str, max_images: int) -> int:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in gt.get("annotations", []):
            annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        count = 0
        for image in gt["images"][: max_images or None]:
            src = image_path_for_record(data_root, image)
            stem = f"{int(image['id']):08d}_{Path(image['file_name']).stem}"
            image_target = dataset / "images" / split / f"{stem}{src.suffix.lower() or '.jpg'}"
            label_target = dataset / "labels" / split / f"{stem}.txt"
            try:
                image_target.symlink_to(src)
            except OSError:
                shutil.copy2(src, image_target)
            width = float(image["width"])
            height = float(image["height"])
            lines: list[str] = []
            for ann in annotations_by_image.get(int(image["id"]), []):
                x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
                if w <= 0 or h <= 0:
                    continue
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                lines.append(f"0 {cx:.8f} {cy:.8f} {w / width:.8f} {h / height:.8f}")
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            count += 1
        return count

    train_count = convert(train_gt, "train", train_max)
    val_count = convert(val_gt, "val", val_max)
    data_yaml = {
        "path": str(dataset),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "athlete"},
    }
    (dataset / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (dataset / "manifest.json").write_text(
        json.dumps({"train_images": train_count, "val_images": val_count}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return dataset


def upload_result(run_id: str, folder: Path) -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        api.upload_folder(
            repo_id=os.environ["HF_MODEL_REPO"],
            repo_type="model",
            folder_path=folder,
            path_in_repo=f"runs/{run_id}",
            commit_message=f"Record {run_id}",
        )
    except Exception as exc:
        print(
            "UPLOAD_RESULT_FAILED "
            + json.dumps({"run_id": run_id, "error": repr(exc)}, sort_keys=True),
            flush=True,
        )
        raise RuntimeError(f"Result upload failed for {run_id}") from exc


def run_baseline() -> dict[str, Any]:
    required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    split = os.getenv("SYNLOC_SPLIT", "valid")
    version = os.getenv("SYNLOC_VERSION", "fullhd")
    max_images = env_int("TRAIN_MAX_IMAGES", 128)
    imgsz = env_int("YOLO_IMGSZ", 960)
    conf = env_float("YOLO_CONF", 0.01)
    iou = env_float("YOLO_IOU", 0.7)
    raw_specs = os.getenv("YOLO_BASELINES", ";".join(DEFAULT_BASELINES))
    specs = parse_baselines(raw_specs)

    data_root = load_synloc_data(version, [f"raw/{version}/*.zip", f"raw/{version}/manifest.json"])
    gt_path = find_annotation(data_root, split)
    model_dir = Path("/tmp/yolo-models")

    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    upload_root = Path("/tmp/yolo-baseline-results")
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        model_path = download_model(spec, model_dir / spec.name)
        item = predictions_for_model(
            model_path=model_path,
            spec=spec,
            data_root=data_root,
            gt_path=gt_path,
            split=split,
            max_images=max_images,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
        summary = item["summary"]
        evaluated.append(summary)
        shutil.copytree(item["out_dir"], upload_root / summary["run_id"])
        if best is None or summary["metrics"]["map_locsim"] > best["metrics"]["map_locsim"]:
            best = summary

    assert best is not None
    summary = {
        "ok": True,
        "ts": utc_now(),
        "mode": "baseline",
        "run_id": f"pretrained-yolo-baseline-{utc_now().replace(':', '-')}",
        "evaluated": evaluated,
        "best": best,
        "metric": "mAP-LocSim",
        "split": split,
        "version": version,
        "max_images": max_images,
    }
    (upload_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    emit_autonomy_result(summary)
    upload_result(summary["run_id"], upload_root)
    return summary


def run_transformer_baseline() -> dict[str, Any]:
    required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    split = os.getenv("SYNLOC_SPLIT", "valid")
    version = os.getenv("SYNLOC_VERSION", "fullhd")
    max_images = env_int("TRAIN_MAX_IMAGES", 64)
    threshold = env_float("TRANSFORMER_CONF", 0.05)
    raw_specs = os.getenv("TRANSFORMER_BASELINES", ";".join(DEFAULT_TRANSFORMER_BASELINES))
    specs = parse_transformer_baselines(raw_specs)

    data_root = load_synloc_data(version, [f"raw/{version}/*.zip", f"raw/{version}/manifest.json"])
    gt_path = find_annotation(data_root, split)
    upload_root = Path("/tmp/transformer-baseline-results")
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True, exist_ok=True)

    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for spec in specs:
        item = predictions_for_transformer_model(
            spec=spec,
            data_root=data_root,
            gt_path=gt_path,
            split=split,
            max_images=max_images,
            threshold=threshold,
        )
        summary = item["summary"]
        evaluated.append(summary)
        shutil.copytree(item["out_dir"], upload_root / summary["run_id"])
        if best is None or summary["metrics"]["map_locsim"] > best["metrics"]["map_locsim"]:
            best = summary

    assert best is not None
    summary = {
        "ok": True,
        "ts": utc_now(),
        "mode": "transformer_baseline",
        "run_id": f"transformer-candidate-baseline-{utc_now().replace(':', '-')}",
        "evaluated": evaluated,
        "best": best,
        "metric": "mAP-LocSim",
        "split": split,
        "version": version,
        "max_images": max_images,
        "threshold": threshold,
        "note": "Non-YOLO COCO transformer detector candidate source; bottom-center projection only.",
    }
    (upload_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    emit_autonomy_result(summary)
    upload_result(summary["run_id"], upload_root)
    return summary


def run_rfdetr_baseline() -> dict[str, Any]:
    required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    split = os.getenv("SYNLOC_SPLIT", "valid")
    version = os.getenv("SYNLOC_VERSION", "fullhd")
    max_images = env_int("TRAIN_MAX_IMAGES", 32)
    threshold = env_float("RFDETR_CONF", 0.5)
    model_class_name = os.getenv("RFDETR_MODEL_CLASS", "RFDETRLarge").strip()
    raw_specs = os.getenv("RFDETR_BASELINES", ";".join(DEFAULT_RFDETR_BASELINES))
    specs = parse_rfdetr_baselines(raw_specs)

    data_root = load_synloc_data(version, [f"raw/{version}/*.zip", f"raw/{version}/manifest.json"])
    gt_path = find_annotation(data_root, split)
    upload_root = Path("/tmp/rfdetr-baseline-results")
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True, exist_ok=True)

    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for spec in specs:
        item = predictions_for_rfdetr_model(
            spec=spec,
            data_root=data_root,
            gt_path=gt_path,
            split=split,
            max_images=max_images,
            threshold=threshold,
            model_class_name=model_class_name,
        )
        summary = item["summary"]
        evaluated.append(summary)
        shutil.copytree(item["out_dir"], upload_root / summary["run_id"])
        if best is None or summary["metrics"]["map_locsim"] > best["metrics"]["map_locsim"]:
            best = summary

    assert best is not None
    summary = {
        "ok": True,
        "ts": utc_now(),
        "mode": "rfdetr_baseline",
        "run_id": f"rfdetr-soccernet-candidate-baseline-{utc_now().replace(':', '-')}",
        "evaluated": evaluated,
        "best": best,
        "metric": "mAP-LocSim",
        "split": split,
        "version": version,
        "max_images": max_images,
        "threshold": threshold,
        "rfdetr_model_class": model_class_name,
        "note": "SoccerNet-Tracking RF-DETR candidate source; bottom-center projection only.",
    }
    (upload_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    emit_autonomy_result(summary)
    upload_result(summary["run_id"], upload_root)
    return summary


def run_finetune() -> dict[str, Any]:
    required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    version = os.getenv("SYNLOC_VERSION", "fullhd")
    train_max = env_int("TRAIN_MAX_IMAGES", 2048)
    val_max = env_int("VAL_MAX_IMAGES", 512)
    imgsz = env_int("YOLO_IMGSZ", 960)
    epochs = env_int("YOLO_EPOCHS", 3)
    batch = env_int("YOLO_BATCH", 4)
    spec = parse_baselines(os.getenv("YOLO_BASELINE_FOR_TRAIN", DEFAULT_BASELINES[0]))[0]

    data_root = load_synloc_data(version, [f"raw/{version}/*.zip", f"raw/{version}/manifest.json"])
    train_gt = find_annotation(data_root, "train")
    val_gt = find_annotation(data_root, "valid")
    dataset = make_yolo_dataset(data_root, train_gt, val_gt, train_max=train_max, val_max=val_max)

    model_path = download_model(spec, Path("/tmp/yolo-models") / spec.name)
    project = Path("/tmp/yolo-train")
    model = YOLO(str(model_path))
    train_result = model.train(
        data=str(dataset / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=2,
        project=str(project),
        name="synloc-finetune",
        exist_ok=True,
        pretrained=True,
        verbose=False,
    )
    save_dir = Path(getattr(train_result, "save_dir", project / "synloc-finetune"))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"Training completed but best.pt was not found at {best_pt}")

    finetuned_spec = BaselineSpec(
        name=f"{spec.name}-synloc-finetuned",
        repo=spec.repo,
        filename=str(best_pt),
        athlete_class_ids={0},
    )
    item = predictions_for_model(
        model_path=best_pt,
        spec=finetuned_spec,
        data_root=data_root,
        gt_path=val_gt,
        split="valid",
        max_images=val_max,
        imgsz=imgsz,
        conf=env_float("YOLO_CONF", 0.01),
        iou=env_float("YOLO_IOU", 0.7),
    )
    summary = {
        "ok": True,
        "ts": utc_now(),
        "mode": "finetune",
        "run_id": f"yolo-synloc-finetune-{utc_now().replace(':', '-')}",
        "base_model": spec.name,
        "best_checkpoint": str(best_pt),
        "train_images": train_max,
        "val_images": val_max,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "validation": item["summary"],
    }
    upload_root = Path("/tmp") / summary["run_id"]
    if upload_root.exists():
        shutil.rmtree(upload_root)
    shutil.copytree(save_dir, upload_root / "ultralytics_train")
    shutil.copytree(item["out_dir"], upload_root / "validation")
    (upload_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    emit_autonomy_result(summary)
    upload_result(summary["run_id"], upload_root)
    return summary


def run_keypoint() -> dict[str, Any]:
    required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    version = os.getenv("SYNLOC_VERSION", "fullhd")
    train_max = env_int("TRAIN_MAX_IMAGES", 512)
    val_max = env_int("VAL_MAX_IMAGES", 128)
    imgsz = env_int("YOLO_IMGSZ", 960)
    epochs = env_int("YOLO_EPOCHS", 2)
    batch = env_int("YOLO_BATCH", 4)
    conf = env_float("YOLO_CONF", 0.01)
    iou = env_float("YOLO_IOU", 0.7)
    source_keypoint_index = env_int("SYNLOC_SOURCE_KEYPOINT_INDEX", 1)
    keypoint_target = os.getenv("SYNLOC_KEYPOINT_TARGET", "annotation").strip().lower()
    max_detections_per_image = env_int("YOLO_MAX_DETECTIONS_PER_IMAGE", 25)
    audit_sample_images = env_int("SYNLOC_AUDIT_SAMPLE_IMAGES", 5)
    score_modes = keypoint_score_modes(os.getenv("YOLO_KEYPOINT_SCORE_MODE", "combined"))
    pose_model = os.getenv("YOLO_POSE_MODEL", "yolo11n-pose.pt")

    data_root = load_synloc_data(version, [f"raw/{version}/*.zip", f"raw/{version}/manifest.json"])
    train_gt = find_annotation(data_root, "train")
    val_gt = find_annotation(data_root, "valid")
    dataset, dataset_manifest = make_yolo_keypoint_dataset(
        data_root,
        train_gt,
        val_gt,
        train_max=train_max,
        val_max=val_max,
        source_keypoint_index=source_keypoint_index,
        keypoint_target=keypoint_target,
    )

    project = Path("/tmp/yolo-keypoint-train")
    model = YOLO(pose_model)
    train_result = model.train(
        data=str(dataset / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=2,
        project=str(project),
        name="synloc-keypoint",
        exist_ok=True,
        pretrained=True,
        verbose=False,
    )
    save_dir = Path(getattr(train_result, "save_dir", project / "synloc-keypoint"))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"Training completed but best.pt was not found at {best_pt}")

    validations: dict[str, dict[str, Any]] = {}
    best_validation: dict[str, Any] | None = None
    for score_mode in score_modes:
        item = predictions_for_keypoint_model(
            model_path=best_pt,
            data_root=data_root,
            gt_path=val_gt,
            split="valid",
            max_images=val_max,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            source_keypoint_index=source_keypoint_index,
            keypoint_target=keypoint_target,
            max_detections_per_image=max_detections_per_image,
            audit_sample_images=audit_sample_images,
            score_mode=score_mode,
        )
        validation = item["summary"]
        validation["out_dir"] = str(item["out_dir"])
        validations[score_mode] = validation
        if (
            best_validation is None
            or validation["metrics"]["map_locsim"] > best_validation["metrics"]["map_locsim"]
        ):
            best_validation = validation
    assert best_validation is not None
    summary = {
        "ok": True,
        "ts": utc_now(),
        "mode": "keypoint",
        "run_id": f"synloc-keypoint-smoke-{utc_now().replace(':', '-')}",
        "pose_model": pose_model,
        "keypoint_target": keypoint_target,
        "best_checkpoint": str(best_pt),
        "train_images": train_max,
        "val_images": val_max,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "max_detections_per_image": max_detections_per_image,
        "audit_sample_images": audit_sample_images,
        "score_modes": score_modes,
        "best_score_mode": best_validation["score_mode"],
        "dataset": dataset_manifest,
        "validation": best_validation,
        "validations": validations,
    }
    upload_root = Path("/tmp") / summary["run_id"]
    if upload_root.exists():
        shutil.rmtree(upload_root)
    shutil.copytree(save_dir, upload_root / "ultralytics_train")
    shutil.copytree(Path(best_validation["out_dir"]), upload_root / "validation")
    for score_mode, validation in validations.items():
        shutil.copytree(Path(validation["out_dir"]), upload_root / f"validation_{score_mode}")
        del validation["out_dir"]
    (upload_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    emit_autonomy_result(summary)
    upload_result(summary["run_id"], upload_root)
    return summary


def run_point_regressor() -> dict[str, Any]:
    required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    version = os.getenv("SYNLOC_VERSION", "fullhd")
    train_max = env_int("TRAIN_MAX_IMAGES", 256)
    val_max = env_int("VAL_MAX_IMAGES", 64)
    epochs = env_int("POINT_EPOCHS", 3)
    batch = env_int("POINT_BATCH", 32)
    image_size = env_int("POINT_IMAGE_SIZE", 128)
    lr = env_float("POINT_LR", 0.001)
    crop_padding = env_float("POINT_CROP_PADDING", 0.15)
    source_keypoint_index = env_int("SYNLOC_SOURCE_KEYPOINT_INDEX", 1)
    keypoint_target = os.getenv("SYNLOC_KEYPOINT_TARGET", "annotation").strip().lower()
    audit_sample_images = env_int("SYNLOC_AUDIT_SAMPLE_IMAGES", 8)
    candidate_mode = os.getenv("POINT_CANDIDATE_MODE", "oracle").strip().lower()
    if candidate_mode not in {"oracle", "jittered", "yolo"}:
        raise RuntimeError("POINT_CANDIDATE_MODE must be one of: oracle, jittered, yolo")
    detector_imgsz = env_int("POINT_DETECTOR_IMGSZ", env_int("YOLO_IMGSZ", 960))
    detector_conf = env_float("POINT_DETECTOR_CONF", env_float("YOLO_CONF", 0.01))
    detector_iou = env_float("POINT_DETECTOR_IOU", env_float("YOLO_IOU", 0.7))
    max_detections_per_image = env_int("POINT_MAX_DETECTIONS_PER_IMAGE", 25)
    jitter_center_frac = env_float("POINT_JITTER_CENTER_FRAC", 0.10)
    jitter_scale_frac = env_float("POINT_JITTER_SCALE_FRAC", 0.15)
    jitter_seed = env_int("POINT_JITTER_SEED", 20260505)
    coordinate_scale_mode = os.getenv("SYNLOC_COORD_SCALE_MODE", "strict").strip().lower()
    if coordinate_scale_mode not in {"strict", "actual_image"}:
        raise RuntimeError("SYNLOC_COORD_SCALE_MODE must be one of: strict, actual_image")

    data_root = load_synloc_data(version, [f"raw/{version}/*.zip", f"raw/{version}/manifest.json"])
    train_gt = find_annotation(data_root, "train")
    val_gt = find_annotation(data_root, "valid")
    train_samples, train_skipped = build_point_samples(
        data_root,
        train_gt,
        max_images=train_max,
        source_keypoint_index=source_keypoint_index,
        keypoint_target=keypoint_target,
        coordinate_scale_mode=coordinate_scale_mode,
    )
    val_samples, val_skipped = build_point_samples(
        data_root,
        val_gt,
        max_images=val_max,
        source_keypoint_index=source_keypoint_index,
        keypoint_target=keypoint_target,
        coordinate_scale_mode=coordinate_scale_mode,
    )
    if not train_samples or not val_samples:
        raise RuntimeError(
            f"Point regressor needs train and validation samples; got train={len(train_samples)} val={len(val_samples)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(env_int("POINT_SEED", 20260505))
    model = TinyPointRegressor().to(device)
    dataset = PointCropDataset(train_samples, image_size=image_size, crop_padding=crop_padding)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    epoch_losses: list[float] = []
    model.train()
    for _epoch in range(epochs):
        losses: list[float] = []
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)) if losses else 0.0)

    detector_spec = None
    if candidate_mode == "oracle":
        item = evaluate_point_regressor(
            model=model,
            samples=val_samples,
            gt_path=val_gt,
            image_size=image_size,
            crop_padding=crop_padding,
            device=device,
            audit_sample_images=audit_sample_images,
        )
    elif candidate_mode == "jittered":
        item = evaluate_point_regressor_on_jittered_candidates(
            model=model,
            samples=val_samples,
            gt_path=val_gt,
            image_size=image_size,
            crop_padding=crop_padding,
            device=device,
            audit_sample_images=audit_sample_images,
            center_frac=jitter_center_frac,
            scale_frac=jitter_scale_frac,
            seed=jitter_seed,
        )
    else:
        detector_spec = parse_baselines(os.getenv("POINT_DETECTOR_BASELINE", DEFAULT_BASELINES[0]))[0]
        detector_path = download_model(detector_spec, Path("/tmp/synloc-point-detectors"))
        item = evaluate_point_regressor_on_yolo_candidates(
            point_model=model,
            detector_model_path=detector_path,
            detector_spec=detector_spec,
            data_root=data_root,
            gt_path=val_gt,
            max_images=val_max,
            detector_imgsz=detector_imgsz,
            detector_conf=detector_conf,
            detector_iou=detector_iou,
            max_detections_per_image=max_detections_per_image,
            image_size=image_size,
            crop_padding=crop_padding,
            device=device,
            keypoint_target=keypoint_target,
            source_keypoint_index=source_keypoint_index,
            audit_sample_images=audit_sample_images,
            coordinate_scale_mode=coordinate_scale_mode,
        )
    validation = item["summary"]
    summary = {
        "ok": True,
        "ts": utc_now(),
        "mode": "point_regressor",
        "run_id": f"synloc-point-regressor-smoke-{utc_now().replace(':', '-')}",
        "candidate_mode": candidate_mode,
        "oracle_candidate_boxes": candidate_mode == "oracle",
        "keypoint_target": keypoint_target,
        "coordinate_scale_mode": coordinate_scale_mode,
        "source_keypoint_index": source_keypoint_index,
        "train_images": train_max,
        "val_images": val_max,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_skipped": len(train_skipped),
        "val_skipped": len(val_skipped),
        "epochs": epochs,
        "batch": batch,
        "image_size": image_size,
        "crop_padding": crop_padding,
        "lr": lr,
        "detector": None if detector_spec is None else detector_spec.name,
        "detector_repo": None if detector_spec is None else detector_spec.repo,
        "detector_filename": None if detector_spec is None else detector_spec.filename,
        "detector_athlete_class_ids": None if detector_spec is None else sorted(detector_spec.athlete_class_ids),
        "detector_imgsz": detector_imgsz if candidate_mode == "yolo" else None,
        "detector_conf": detector_conf if candidate_mode == "yolo" else None,
        "detector_iou": detector_iou if candidate_mode == "yolo" else None,
        "max_detections_per_image": max_detections_per_image if candidate_mode == "yolo" else None,
        "jitter_center_frac": jitter_center_frac if candidate_mode == "jittered" else None,
        "jitter_scale_frac": jitter_scale_frac if candidate_mode == "jittered" else None,
        "jitter_seed": jitter_seed if candidate_mode == "jittered" else None,
        "epoch_losses": epoch_losses,
        "validation": validation,
        "metric": "mAP-LocSim",
        "note": (
            "Uses GT boxes as oracle candidates to isolate direct point quality; not challenge-submittable."
            if candidate_mode == "oracle"
            else (
                "Uses deterministically jittered GT boxes to measure direct point tolerance to candidate-box noise; not challenge-submittable."
                if candidate_mode == "jittered"
                else "Uses YOLO detector boxes as real candidates for the direct crop point regressor."
            )
        ),
    }
    upload_root = Path("/tmp") / summary["run_id"]
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), upload_root / "point_regressor.pt")
    shutil.copytree(Path(item["out_dir"]), upload_root / "validation")
    (upload_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    emit_autonomy_result(summary)
    upload_result(summary["run_id"], upload_root)
    return summary


def main() -> None:
    mode = os.getenv("TRAIN_MODE", "baseline").strip().lower()
    if mode == "baseline":
        summary = run_baseline()
    elif mode in {"transformer_baseline", "rtdetr_baseline", "detr_baseline"}:
        summary = run_transformer_baseline()
    elif mode in {"rfdetr_baseline", "rfdetr", "soccernet_rfdetr"}:
        summary = run_rfdetr_baseline()
    elif mode in {"train", "finetune"}:
        summary = run_finetune()
    elif mode in {"keypoint", "pose", "footpoint"}:
        summary = run_keypoint()
    elif mode in {"point_regressor", "direct_point", "ground_point"}:
        summary = run_point_regressor()
    else:
        raise RuntimeError(f"Unsupported TRAIN_MODE={mode!r}")
    emit_autonomy_result(summary)


if __name__ == "__main__":
    main()
