from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="在 val/test 集上验证 YOLO 模型")
    parser.add_argument("--weights", default="D:/pycharm/pythonProject/PCB_defect/runs/01_detection/training/PCB_defect_clahe/weights/best.pt")
    parser.add_argument("--data", default="D:/pycharm/pythonProject/PCB_defect/data/pcb_detection_clahe/dataset.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    metrics = YOLO(args.weights).val(data=str(Path(args.data).resolve()), split=args.split, imgsz=args.imgsz,
                                     device=args.device, project=str(ROOT / "runs" / "01_detection/evaluation"), name=Path(args.weights).stem, plots=True)
    result = {"precision": round(float(metrics.box.mp), 5), "recall": round(float(metrics.box.mr), 5),
              "map50": round(float(metrics.box.map50), 5), "map50_95": round(float(metrics.box.map), 5),
              "per_class_map50": [round(float(x), 5) for x in metrics.box.maps]}
    out = ROOT / "reports" / "metrics_bilateral.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2)); print(f"指标文件: {out}")


if __name__ == "__main__": main()
