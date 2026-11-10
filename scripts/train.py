from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 手机屏幕缺陷训练")
    parser.add_argument("--data", default="D:/pycharm/pythonProject/PCB_defect/data/pcb_detection_clahe/dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=-1, help="-1 自动 batch")
    parser.add_argument("--device", default=0, help="例如 0 或 cpu")
    parser.add_argument("--name", default="PCB_defect_clahe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    model = YOLO(args.model)
    results = model.train(data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                          device=args.device, workers=args.workers, project=str(ROOT / "runs" / "train"), name=args.name, seed=args.seed,
                          pretrained=True, patience=30, cos_lr=True, close_mosaic=10, plots=True, resume=args.resume)
    print(f"训练完成。结果目录: {results.save_dir}")


if __name__ == "__main__": main()
