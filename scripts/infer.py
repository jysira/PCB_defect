from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="图像/视频/摄像头实时缺陷检测")
    parser.add_argument("--weights", default="D:/pycharm/PycharmProjects/PythonProject1/PCB_defect/runs/train/PCB_defect/weights/best.engine", help=".pt、.onnx 或 .engine")
    parser.add_argument("--source", default="C:/Users/曲敬悠/Desktop                              ", help="图片/视频路径、目录，或摄像头编号 0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    for result in model.predict(source=source, conf=args.conf, imgsz=args.imgsz, device=args.device, stream=True, save=args.save, verbose=False):
        annotated = result.plot()
        for box in result.boxes:
            cls = int(box.cls[0]); name = result.names[cls]; score = float(box.conf[0]); xyxy = [round(float(v), 1) for v in box.xyxy[0]]
            print(f"Defect: {name}; Confidence: {score:.3f}; BBox: {xyxy}")
        if isinstance(source, int):
            cv2.imshow("Screen defect detection (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    cv2.destroyAllWindows()


if __name__ == "__main__": main()
