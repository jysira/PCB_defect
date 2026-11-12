from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 YOLO 为 ONNX")
    parser.add_argument("--weights", default="D:/pycharm/pythonProject/PCB_defect/runs/01_detection/training/PCB_defect_clahe/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    output = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True, nms=True,device=args.device)
    print(f"ONNX 已导出: {output}")


if __name__ == "__main__": main()
