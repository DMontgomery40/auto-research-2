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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download
from sskit.coco import LocSimCOCOeval
from ultralytics import YOLO
from xtcocotools.coco import COCO


WORK = Path("/tmp/synloc-pose-smoke")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


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


def load_synloc_data(version: str) -> Path:
    required = ["HF_TOKEN", "HF_DATASET_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    cache_dir = WORK / "hf-synloc-cache"
    data_root = WORK / "SoccerNet"
    snapshot_download(
        repo_id=os.environ["HF_DATASET_REPO"],
        repo_type="dataset",
        local_dir=cache_dir,
        allow_patterns=[f"raw/{version}/*.zip", f"raw/{version}/manifest.json"],
        token=os.environ["HF_TOKEN"],
    )
    extract_archives(cache_dir, data_root)
    return data_root


def grouped_keypoints(raw: Any) -> list[list[float]]:
    if not raw:
        return []
    if isinstance(raw[0], (int, float)):
        return [
            [float(raw[i]), float(raw[i + 1]), float(raw[i + 2]) if i + 2 < len(raw) else 1.0]
            for i in range(0, len(raw), 3)
            if i + 1 < len(raw)
        ]
    return [
        [float(item[0]), float(item[1]), float(item[2]) if len(item) > 2 else 1.0]
        for item in raw
        if len(item) >= 2
    ]


def yolo_visibility(value: float) -> int:
    return 2 if value > 0 else 0


def subset_gt(gt: dict[str, Any], images: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    ids = {int(image["id"]) for image in images}
    subset = {
        **gt,
        "images": images,
        "annotations": [ann for ann in gt.get("annotations", []) if int(ann["image_id"]) in ids],
    }
    out_path.write_text(json.dumps(subset), encoding="utf-8")
    return subset


def make_pose_dataset(
    *,
    data_root: Path,
    gt: dict[str, Any],
    train_images: list[dict[str, Any]],
    val_images: list[dict[str, Any]],
) -> Path:
    dataset = WORK / "yolo-pose-dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in gt.get("annotations", []):
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    def write_split(images: list[dict[str, Any]], split: str) -> int:
        count = 0
        for image in images:
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
                kpts = grouped_keypoints(ann.get("keypoints", []))
                if w <= 0 or h <= 0 or len(kpts) < 2:
                    continue
                values = [
                    "0",
                    f"{(x + w / 2.0) / width:.8f}",
                    f"{(y + h / 2.0) / height:.8f}",
                    f"{w / width:.8f}",
                    f"{h / height:.8f}",
                ]
                for kpt in kpts[:2]:
                    values.extend(
                        [
                            f"{min(max(kpt[0] / width, 0.0), 1.0):.8f}",
                            f"{min(max(kpt[1] / height, 0.0), 1.0):.8f}",
                            str(yolo_visibility(kpt[2])),
                        ]
                    )
                lines.append(" ".join(values))
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            count += 1
        return count

    train_count = write_split(train_images, "train")
    val_count = write_split(val_images, "val")
    data_yaml = {
        "path": str(dataset),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "athlete"},
        "kpt_shape": [2, 3],
        "flip_idx": [0, 1],
    }
    (dataset / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (dataset / "manifest.json").write_text(
        json.dumps({"train_images": train_count, "val_images": val_count}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return dataset


def evaluate(gt_path: Path, pred_path: Path, *, empty: bool) -> dict[str, float]:
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
    coco_eval.params.position_from_keypoint_index = 1
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


def predict_pose(
    *,
    model: YOLO,
    data_root: Path,
    images: list[dict[str, Any]],
    imgsz: int,
    conf: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    det_id = 1
    for image in images:
        path = image_path(data_root, image["file_name"])
        preds = model.predict(source=str(path), imgsz=imgsz, conf=conf, verbose=False, device=0)
        if not preds:
            continue
        pred = preds[0]
        if pred.boxes is None or pred.keypoints is None:
            continue
        boxes = pred.boxes.xyxy.detach().cpu().numpy()
        scores = pred.boxes.conf.detach().cpu().numpy()
        keypoints = pred.keypoints.data.detach().cpu().numpy()
        for box, score, kpts in zip(boxes, scores, keypoints):
            if len(kpts) < 2:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            records.append(
                {
                    "area": 0,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "category_id": 1,
                    "id": det_id,
                    "image_id": int(image["id"]),
                    "keypoints": [
                        [float(kpt[0]), float(kpt[1]), float(kpt[2]) if len(kpt) > 2 else 1.0]
                        for kpt in kpts[:2]
                    ],
                    "num_keypoints": 2,
                    "score": float(score),
                }
            )
            det_id += 1
    return records


def main() -> None:
    required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    split = os.getenv("SYNLOC_SPLIT", "valid")
    version = os.getenv("SYNLOC_VERSION", "fullhd")
    train_max = env_int("POSE_TRAIN_MAX_IMAGES", 64)
    val_max = env_int("POSE_VAL_MAX_IMAGES", 64)
    val_start = env_int("POSE_VAL_START", train_max)
    imgsz = env_int("YOLO_IMGSZ", 640)
    epochs = env_int("YOLO_EPOCHS", 1)
    batch = env_int("YOLO_BATCH", 4)
    conf = float(os.getenv("YOLO_CONF", "0.01"))

    data_root = load_synloc_data(version)
    gt_path = find_annotation(data_root, split)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    train_images = gt["images"][:train_max]
    val_images = gt["images"][val_start : val_start + val_max]
    if not train_images or not val_images:
        raise RuntimeError("Not enough cached validation images for pose smoke split")

    dataset = make_pose_dataset(data_root=data_root, gt=gt, train_images=train_images, val_images=val_images)
    val_gt_path = WORK / "val_subset.json"
    val_gt = subset_gt(gt, val_images, val_gt_path)

    base_model = os.getenv("POSE_BASE_MODEL", "yolo11n-pose.pt")
    model = YOLO(base_model)
    train_result = model.train(
        data=str(dataset / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=2,
        project=str(WORK / "yolo-pose-train"),
        name="synloc-pose-smoke",
        exist_ok=True,
        pretrained=True,
        verbose=False,
    )
    save_dir = Path(getattr(train_result, "save_dir", WORK / "yolo-pose-train" / "synloc-pose-smoke"))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"Pose smoke completed but best.pt was not found at {best_pt}")

    trained = YOLO(str(best_pt))
    records = predict_pose(model=trained, data_root=data_root, images=val_images, imgsz=imgsz, conf=conf)
    pred_path = WORK / "results.json"
    pred_path.write_text(json.dumps(records), encoding="utf-8")
    metrics = evaluate(val_gt_path, pred_path, empty=not records)

    run_id = "synloc-pose-smoke-" + utc_now().replace(":", "-").replace(".", "-")
    out_dir = WORK / "artifact"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copy2(pred_path, out_dir / "results.json")
    shutil.copy2(val_gt_path, out_dir / "val_subset.json")
    shutil.copy2(dataset / "manifest.json", out_dir / "dataset_manifest.json")
    shutil.copy2(best_pt, out_dir / "best.pt")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "score_threshold": conf,
                "position_from_keypoint_index": 1,
                "warning": "Smoke only: trained on a validation slice and evaluated on a later validation slice. Do not promote as a leaderboard score.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "mode": "pose_smoke",
        "metric": "mAP-LocSim",
        "split": split,
        "version": version,
        "base_model": base_model,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "val_annotations": len(val_gt.get("annotations", [])),
        "num_predictions": len(records),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "conf": conf,
        "metrics": metrics,
        "promotion_allowed": False,
        "interpretation": "Pipeline smoke for SynLoc keypoint/ground-point training; validation-slice training makes this non-promotable.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
        repo_id=os.environ["HF_MODEL_REPO"],
        repo_type="model",
        folder_path=out_dir,
        path_in_repo=f"runs/{run_id}",
        commit_message=f"Record {run_id}",
    )
    print("AUTONOMY_RESULT " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
