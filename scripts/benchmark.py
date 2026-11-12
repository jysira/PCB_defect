from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from common import ROOT


def benchmark_one(
    weights: str,
    image: np.ndarray,
    imgsz: int,
    runs: int,
    device: str,
) -> dict:
    """对单个模型做预热 + 精确计时，返回性能指标。"""
    model = YOLO(weights)

    # 预热：让 CUDA context、kernel 缓存、显存分配都稳定下来
    for _ in range(10):
        model.predict(image, imgsz=imgsz, device=device, verbose=False)

    # 判断是否走 CUDA 计时路径。
    # .pt 在 GPU 上由 PyTorch 异步执行，.engine 由 TensorRT 异步执行，都可以用 CUDA Event。
    # .onnx 走 onnxruntime，不经过 PyTorch 的 CUDA 流，需逐次同步计时。
    use_cuda_event = torch.cuda.is_available() and weights.endswith((".pt", ".engine"))

    if use_cuda_event:
        # 先同步并重置显存统计，避免把预热阶段的占用算进来
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(runs):
            model.predict(image, imgsz=imgsz, device=device, verbose=False)
        end_event.record()
        torch.cuda.synchronize()  # 等待所有 kernel 真正执行完

        elapsed_ms = start_event.elapsed_time(end_event) / runs
    else:
        # ONNX 或 CPU：每次推理后立即同步，测单次真实延迟
        started = time.perf_counter()
        for _ in range(runs):
            model.predict(image, imgsz=imgsz, device=device, verbose=False)
        elapsed_ms = (time.perf_counter() - started) * 1000 / runs

    peak_mb = None
    if torch.cuda.is_available():
        peak_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    return {
        "model": Path(weights).name,
        "latency_ms": round(elapsed_ms, 3),
        "fps": round(1000 / elapsed_ms, 2),
        "peak_gpu_memory_mb": peak_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 PyTorch / ONNX / TensorRT 推理延迟与 FPS")
    parser.add_argument("--weights", nargs="+", default=[
        r"D:\pycharm\pythonProject\PCB_defect\runs\01_detection\training\PCB_defect_clahe\weights\best.pt",
        r"D:\pycharm\pythonProject\PCB_defect\runs\01_detection\training\PCB_defect_clahe\weights\best.onnx",
        r"D:\pycharm\pythonProject\PCB_defect\runs\01_detection\training\PCB_defect_clahe\weights\exports_nrxrkkhq\best_fp16.engine",
        r"D:\pycharm\pythonProject\PCB_defect\runs\01_detection\training\PCB_defect_clahe\weights\exports_nrxrkkhq\best_int8.engine",
    ], help="可同时给 .pt .onnx .engine")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    # 用随机噪声而非全零，避免后端对全零输入走短路优化导致延迟偏低
    rng = np.random.default_rng(seed=42)
    image = rng.integers(0, 256, size=(args.imgsz, args.imgsz, 3), dtype=np.uint8)

    rows = [benchmark_one(w, image, args.imgsz, args.runs, args.device) for w in args.weights]

    out = ROOT / "reports" / "benchmark_clahe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"性能报告: {out}")


if __name__ == "__main__":
    main()