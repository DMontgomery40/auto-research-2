#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def default_gt_path() -> Path:
    base = ROOT / "data" / "SoccerNet" / "SpiideoSynLoc" / "annotations"
    for name in ("val.json", "valid.json", "validation.json"):
        path = base / name
        if path.exists():
            return path
    return base / "val.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SynLoc predictions with SSKit mAP-LocSim.")
    parser.add_argument("--gt", type=Path, default=default_gt_path())
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--position-from-keypoint-index", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from sskit.coco import LocSimCOCOeval
    from xtcocotools.coco import COCO

    coco = COCO(str(args.gt))
    coco_det = coco.loadRes(str(args.pred))
    coco_eval = LocSimCOCOeval(coco, coco_det, "bbox")
    coco_eval.params.useSegm = None
    if args.score_threshold is not None:
        coco_eval.params.score_threshold = args.score_threshold
    if args.position_from_keypoint_index is not None:
        coco_eval.params.position_from_keypoint_index = args.position_from_keypoint_index

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    summary = {
        "map_locsim": float(coco_eval.stats[0]),
        "precision_50": float(coco_eval.stats[12]),
        "recall_50": float(coco_eval.stats[13]),
        "f1_50": float(coco_eval.stats[14]),
        "score_threshold": float(coco_eval.stats[15]),
        "frame_accuracy": float(coco_eval.stats[16]),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
