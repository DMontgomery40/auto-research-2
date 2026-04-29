#!/usr/bin/env python3
# /// script
# dependencies = [
#   "huggingface_hub>=0.24.0",
#   "ultralytics-opencv-headless>=8.4.29",
#   "torch",
#   "torchvision",
#   "sskit @ git+https://github.com/Spiideo/sskit.git",
#   "scipy",
#   "numpy<2",
#   "xtcocotools",
#   "pyyaml"
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
import yaml
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


def parse_int_set(raw: str) -> set[int]:
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    repo: str
    filename: str
    athlete_class_ids: set[int]


DEFAULT_BASELINES = [
    # Current April 2026 football-specific YOLO26 lead.
    "football-yolo26l|mobadam/football-player-detection|player_detector.pt|1,3",
    # Existing SoccerNet/Soccana YOLO-family soccer detector lead.
    "soccana-yolo11n|Adit-jain/soccana|Model/weights/best.pt|0",
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


def download_model(spec: BaselineSpec, target_dir: Path) -> Path:
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
    model = YOLO(str(model_path))

    results: list[dict[str, Any]] = []
    det_id = 1
    for image in images:
        path = image_path(data_root, image["file_name"])
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
                    "image_id": image["id"],
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
        "split": split,
        "max_images": max_images,
        "imgsz": imgsz,
        "iou": iou,
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
        "max_images": max_images,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "num_images": len(images),
        "num_detections": len(results),
        "metrics": metrics,
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
            src = image_path(data_root, image["file_name"])
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
    api.upload_folder(
        repo_id=os.environ["HF_MODEL_REPO"],
        repo_type="model",
        folder_path=folder,
        path_in_repo=f"runs/{run_id}",
        commit_message=f"Record {run_id}",
    )


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
    upload_result(summary["run_id"], upload_root)
    return summary


def main() -> None:
    mode = os.getenv("TRAIN_MODE", "baseline").strip().lower()
    if mode == "baseline":
        summary = run_baseline()
    elif mode in {"train", "finetune"}:
        summary = run_finetune()
    else:
        raise RuntimeError(f"Unsupported TRAIN_MODE={mode!r}")
    print("AUTONOMY_RESULT " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
