from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import yaml

from common import ROOT, label_dir, load_data_config
from src.data import image_files
from src.preprocessing import apply_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description="生成预处理后的 YOLO 数据集")
    parser.add_argument("--data", default=str(ROOT / "configs/PCB_defect.yaml"))
    parser.add_argument("--output", required=True, help="预处理后检测数据集的输出目录")
    parser.add_argument("--method", default="clahe", choices=["none", "gaussian", "median", "clahe", "bilateral", "canny", "threshold", "morphology"])
    args = parser.parse_args()
    cfg, root = load_data_config(args.data)
    out_root = Path(args.output).resolve()
    for split in ("train", "val", "test"):
        if split not in cfg: continue
        source_images = root / cfg[split]; source_labels = label_dir(source_images, root)
        target_images = out_root / "images" / split; target_labels = out_root / "labels" / split
        for path in image_files(source_images):
            image = cv2.imread(str(path)); relative = path.relative_to(source_images)
            dst = target_images / relative; dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), apply_preprocessing(image, args.method))
            src_label = source_labels / relative.with_suffix(".txt"); dst_label = target_labels / relative.with_suffix(".txt")
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            if src_label.exists(): shutil.copy2(src_label, dst_label)
            else: dst_label.touch()
    new_cfg = dict(cfg); new_cfg["path"] = str(out_root)
    cfg_path = out_root / "dataset.yaml"; cfg_path.write_text(yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"已生成 {args.method} 数据集: {out_root}\n配置文件: {cfg_path}")


if __name__ == "__main__": main()
