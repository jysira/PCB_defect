from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, label_dir, load_data_config, names_from_config
from src.data import image_files


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 YOLO 工业缺陷数据集并生成图表")
    parser.add_argument("--data", default="D:/pycharm/PycharmProjects/PythonProject1/PCB_defect/configs/PCB_defect.yaml")
    parser.add_argument("--output", default="reports/data")
    args = parser.parse_args()
    cfg, root = load_data_config(args.data)
    names = names_from_config(cfg)
    output = (ROOT / args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    counts, sizes, areas, split_images, split_objects = Counter(), [], [], {}, {}
    invalid = []
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        image_dir = root / cfg[split]
        files = image_files(image_dir)
        split_images[split] = len(files); objects = 0
        for img_path in files:
            image = cv2.imread(str(img_path))
            if image is None:
                invalid.append(f"无法读取: {img_path}"); continue
            h, w = image.shape[:2]; sizes.append((w, h))
            txt = label_dir(image_dir, root) / img_path.relative_to(image_dir).with_suffix(".txt")
            if not txt.exists():
                invalid.append(f"缺标签: {txt}"); continue
            for line in txt.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) != 5:
                    invalid.append(f"标签格式错误: {txt}: {line}"); continue
                cls, _, _, bw, bh = map(float, p)
                if int(cls) not in names or not all(0 <= v <= 1 for v in (bw, bh)):
                    invalid.append(f"标签范围错误: {txt}: {line}"); continue
                counts[names[int(cls)]] += 1; areas.append(bw * bh); objects += 1
        split_objects[split] = objects
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].bar(list(names.values()), [counts[n] for n in names.values()], color="#2878B5"); axes[0].set_title("Defect class distribution"); axes[0].tick_params(axis="x", rotation=30)
    axes[1].hist(areas or [0], bins=30, color="#E18727"); axes[1].set_title("Normalized bounding-box area"); axes[1].set_xlabel("area / image area")
    axes[2].scatter([x[0] for x in sizes], [x[1] for x in sizes], s=12, alpha=.65, color="#3FA7A3"); axes[2].set_title("Image dimensions"); axes[2].set_xlabel("width"); axes[2].set_ylabel("height")
    fig.tight_layout(); fig.savefig(output / "dataset_distributions.png", dpi=180); plt.close(fig)
    summary = {"splits": split_images, "objects_by_split": split_objects, "class_counts": dict(counts),
               "total_images": sum(split_images.values()), "total_objects": sum(split_objects.values()),
               "image_width": {"min": min((x[0] for x in sizes), default=0), "max": max((x[0] for x in sizes), default=0)},
               "invalid_items": invalid}
    (output / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)); print(f"图表: {output / 'dataset_distributions.png'}")


if __name__ == "__main__": main()
