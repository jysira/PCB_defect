from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from common import ROOT, label_dir, load_data_config, names_from_config
from src.data import image_files


def xywhn_to_xyxy(row: list[float], width: int, height: int) -> np.ndarray:
    _, cx, cy, bw, bh = row
    return np.array([(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union


def load_truth(path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    if not path.exists(): return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(v) for v in line.split()]
        if len(values) == 5: rows.append((int(values[0]), xywhn_to_xyxy(values, width, height)))
    return rows


def category_for_miss(box: np.ndarray, image: np.ndarray) -> str:
    h, w = image.shape[:2]; area_ratio = (box[2]-box[0]) * (box[3]-box[1]) / (w*h)
    if area_ratio < 0.01: return "small_object_miss"
    x1, y1, x2, y2 = np.clip(box.astype(int), [0, 0, 0, 0], [w-1, h-1, w, h])
    roi = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if roi.size and float(roi.std()) < 18: return "low_contrast_miss"
    return "other_miss"


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 IoU=0.5 匹配，生成漏检/误检错误报告")
    parser.add_argument("--weights", default="D:/pycharm/pythonProject/PCB_defect/runs/01_detection/training/PCB_defect_clahe/weights/best.pt")
    parser.add_argument("--data", default="D:/pycharm/pythonProject/PCB_defect/data/pcb_detection_clahe/dataset.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=.25)
    args = parser.parse_args()
    cfg, root = load_data_config(args.data)
    names = names_from_config(cfg)
    image_dir = root / cfg[args.split]
    labels = label_dir(image_dir, root)
    out = ROOT / "reports" / "errors"; out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights); records = []; errors, truth_count, found_count = Counter(), Counter(), Counter()
    for image_path in image_files(image_dir):
        image = cv2.imread(str(image_path)); h, w = image.shape[:2]
        truth = load_truth(labels / image_path.relative_to(image_dir).with_suffix(".txt"), w, h)
        prediction = model.predict(str(image_path), conf=args.conf, verbose=False)[0]
        preds = [(int(b.cls[0]), b.xyxy[0].cpu().numpy()) for b in prediction.boxes]
        used = set()
        for cls, gt in truth:
            truth_count[names[cls]] += 1
            candidates = [(idx, iou(gt, box)) for idx, (pcls, box) in enumerate(preds) if idx not in used and pcls == cls]
            best = max(candidates, key=lambda x: x[1], default=(-1, 0))
            if best[1] >= .5: used.add(best[0]); found_count[names[cls]] += 1
            else:
                category = category_for_miss(gt, image); errors[category] += 1
                records.append([str(image_path), names[cls], "false_negative", category])
        for idx, (cls, _) in enumerate(preds):
            if idx not in used: errors["false_positive"] += 1; records.append([str(image_path), names[cls], "false_positive", "false_positive"])
    with (out / "error_records_clahe.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["image", "class", "error_type", "category"]); writer.writerows(records)
    recalls = {name: found_count[name] / truth_count[name] if truth_count[name] else 0 for name in names.values()}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5)); axes[0].bar(recalls.keys(), recalls.values(), color="#2878B5"); axes[0].set_ylim(0, 1); axes[0].set_title("Recall by class"); axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(errors.keys(), errors.values(), color="#E18727"); axes[1].set_title("Error categories"); axes[1].tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(out / "error_summary_clahe.png", dpi=180)
    print("Recall:", {k: f"{v:.2%}" for k, v in recalls.items()}); print(f"报告: {out}")


if __name__ == "__main__": main()
