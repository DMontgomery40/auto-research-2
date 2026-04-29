#!/usr/bin/env python3
# /// script
# dependencies = [
#   "huggingface_hub>=0.24.0",
#   "sskit @ git+https://github.com/Spiideo/sskit.git",
#   "scipy",
#   "numpy<2",
#   "torch",
#   "xtcocotools"
# ]
# ///
from __future__ import annotations

import json
import math
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from sskit.coco import BBoxLocSimCOCOeval, LocSimCOCOeval
from xtcocotools.coco import COCO


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


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


def load_synloc_data(version: str) -> Path:
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
        allow_patterns=[f"raw/{version}/*.zip", f"raw/{version}/manifest.json"],
        token=os.environ["HF_TOKEN"],
    )
    extract_archives(cache_dir, data_root)
    return data_root


def subset_gt(gt_path: Path, max_images: int, out_path: Path) -> dict[str, Any]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    images = gt["images"][: max_images or None]
    image_ids = {int(image["id"]) for image in images}
    subset = {
        **gt,
        "images": images,
        "annotations": [ann for ann in gt["annotations"] if int(ann["image_id"]) in image_ids],
    }
    out_path.write_text(json.dumps(subset), encoding="utf-8")
    return subset


def as_pitch3(value: list[float]) -> list[float]:
    if len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return [float(value[0]), float(value[1]), 0.0]


def bottom_center_keypoint(ann: dict[str, Any]) -> list[float]:
    x, y, w, h = [float(v) for v in ann["bbox"]]
    return [x + w / 2.0, y + h, 1.0]


def exact_position_results(gt: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for det_id, ann in enumerate(gt["annotations"], start=1):
        results.append(
            {
                "area": float(ann.get("area", 0)),
                "bbox": ann["bbox"],
                "category_id": 1,
                "id": det_id,
                "image_id": ann["image_id"],
                "position_on_pitch": as_pitch3(ann["position_on_pitch"]),
                "score": 0.99,
            }
        )
    return results


def gt_keypoint_results(gt: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for det_id, ann in enumerate(gt["annotations"], start=1):
        results.append(
            {
                "area": float(ann.get("area", 0)),
                "bbox": ann["bbox"],
                "category_id": 1,
                "id": det_id,
                "image_id": ann["image_id"],
                "keypoints": ann["keypoints"],
                "num_keypoints": len(ann["keypoints"]),
                "score": 0.99,
            }
        )
    return results


def bbox_bottom_center_keypoint_results(gt: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for det_id, ann in enumerate(gt["annotations"], start=1):
        results.append(
            {
                "area": float(ann.get("area", 0)),
                "bbox": ann["bbox"],
                "category_id": 1,
                "id": det_id,
                "image_id": ann["image_id"],
                "keypoints": [bottom_center_keypoint(ann)],
                "num_keypoints": 1,
                "score": 0.99,
            }
        )
    return results


def bbox_only_results(gt: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for det_id, ann in enumerate(gt["annotations"], start=1):
        results.append(
            {
                "area": float(ann.get("area", 0)),
                "bbox": ann["bbox"],
                "category_id": 1,
                "id": det_id,
                "image_id": ann["image_id"],
                "score": 0.99,
            }
        )
    return results


def evaluate(
    gt_path: Path,
    pred_path: Path,
    *,
    eval_cls: type[LocSimCOCOeval] = LocSimCOCOeval,
    position_from_keypoint_index: int | None = None,
) -> dict[str, float | None]:
    coco = COCO(str(gt_path))
    coco_det = coco.loadRes(str(pred_path))
    coco_eval = eval_cls(coco, coco_det, "bbox")
    coco_eval.params.useSegm = None
    coco_eval.params.score_threshold = 0.5
    if position_from_keypoint_index is not None:
        coco_eval.params.position_from_keypoint_index = position_from_keypoint_index
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return json_safe(
        {
            "map_locsim": float(coco_eval.stats[0]),
            "precision_50": float(coco_eval.stats[12]),
            "recall_50": float(coco_eval.stats[13]),
            "f1_50": float(coco_eval.stats[14]),
            "score_threshold": float(coco_eval.stats[15]),
            "frame_accuracy": float(coco_eval.stats[16]),
        }
    )


def write_case(
    out_dir: Path,
    gt_path: Path,
    name: str,
    results: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    eval_cls: type[LocSimCOCOeval] = LocSimCOCOeval,
    position_from_keypoint_index: int | None = None,
) -> dict[str, Any]:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    pred_path = case_dir / "results.json"
    metadata_path = case_dir / "metadata.json"
    pred_path.write_text(json.dumps(results), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metrics = evaluate(
        gt_path,
        pred_path,
        eval_cls=eval_cls,
        position_from_keypoint_index=position_from_keypoint_index,
    )
    summary = {
        "name": name,
        "num_predictions": len(results),
        "metadata": metadata,
        "metrics": metrics,
    }
    (case_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")

    split = os.getenv("SYNLOC_SPLIT", "valid")
    version = os.getenv("SYNLOC_VERSION", "fullhd")
    max_images = int(os.getenv("DEVKIT_ORACLE_MAX_IMAGES", "64"))

    data_root = load_synloc_data(version)
    gt_source = find_annotation(data_root, split)

    run_id = f"synloc-devkit-oracle-{utc_now().replace(':', '-')}"
    out_dir = Path("/tmp") / run_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    gt_path = out_dir / "gt_subset.json"
    gt = subset_gt(gt_source, max_images, gt_path)
    cases = [
        write_case(
            out_dir,
            gt_path,
            "exact_position_on_pitch",
            exact_position_results(gt),
            metadata={"score_threshold": 0.5},
        ),
        write_case(
            out_dir,
            gt_path,
            "gt_projected_ground_keypoint",
            gt_keypoint_results(gt),
            metadata={"score_threshold": 0.5, "position_from_keypoint_index": 1},
            position_from_keypoint_index=1,
        ),
        write_case(
            out_dir,
            gt_path,
            "gt_bbox_bottom_center_keypoint",
            bbox_bottom_center_keypoint_results(gt),
            metadata={"score_threshold": 0.5, "position_from_keypoint_index": 0},
            position_from_keypoint_index=0,
        ),
        write_case(
            out_dir,
            gt_path,
            "gt_bbox_bottom_center_devkit_eval",
            bbox_only_results(gt),
            metadata={"score_threshold": 0.5, "position_from_keypoint_index": "BBoxLocSimCOCOeval"},
            eval_cls=BBoxLocSimCOCOeval,
            position_from_keypoint_index=0,
        ),
    ]

    summary = {
        "ok": True,
        "ts": utc_now(),
        "run_id": run_id,
        "mode": "devkit_oracle",
        "split": split,
        "version": version,
        "max_images": max_images,
        "num_images": len(gt["images"]),
        "num_annotations": len(gt["annotations"]),
        "cases": cases,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

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
