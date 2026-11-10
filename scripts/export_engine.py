from __future__ import annotations

import argparse
import gc
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

PROJECT = Path(r"D:\pycharm\PycharmProjects\PythonProject1\PCB_defect")


def main() -> None:
    parser = argparse.ArgumentParser(description="依次导出 FP16 和 INT8 TensorRT 引擎")
    parser.add_argument("--weights", default=str(PROJECT / "D:/pycharm/pythonProject/PCB_defect/runs/train/PCB_defect_clahe/weights/best.pt"))
    parser.add_argument("--data", default=str(PROJECT / "D:/pycharm/pythonProject/PCB_defect/data/pcb_detection_clahe/dataset.yaml"), help="INT8 校准数据配置，校准只使用 train")
    parser.add_argument("--output-dir", default=None, help="默认在权重目录下新建 exports_* 文件夹")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=1.0, help="训练数据用于校准的比例 (0, 1]")
    args = parser.parse_args()
    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    if not weights.is_file() or not data.is_file():
        parser.error(f"请检查权重和数据配置路径：{weights}, {data}")
    if not 0 < args.fraction <= 1:
        parser.error("fraction 必须在 (0, 1] 范围内")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 Python 环境无法使用 CUDA，请检查 GPU 版 PyTorch 和驱动")

    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not cfg.get("train"):
        raise ValueError("校准配置缺少 train 数据路径")
    root = Path(cfg.get("path", data.parent))
    if not root.is_absolute():
        raise ValueError("请将数据 YAML 中的 path 设置为数据集绝对路径")
    # 不同版本导出器可能读取 train 或 val；全部指向训练数据以避免测试集泄漏。
    cfg["path"] = str(root)
    cfg["val"] = cfg["train"]
    cfg["test"] = cfg["train"]

    if args.output_dir:
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="exports_", dir=weights.parent))
    calibration = output / "calibration_train.yaml"
    calibration.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"输出目录：{output}", flush=True)

    failures = []
    for name, bits in (("fp16", 16), ("int8", 8)):
        model = None
        try:
            # 为每种精度使用独立权重名，保护已有 best.engine 和 ONNX 文件。
            staged_weights = output / f"{weights.stem}_{name}.pt"
            shutil.copy2(weights, staged_weights)
            model = YOLO(str(staged_weights))
            options = dict(format="engine", imgsz=args.imgsz, device=args.device,
                           batch=1, dynamic=True, simplify=True, nms=True)
            if "quantize" in DEFAULT_CFG_DICT:
                options["quantize"] = bits
            else:
                options.update(half=(bits == 16), int8=(bits == 8))
            if bits == 8:
                options.update(data=str(calibration), fraction=args.fraction)
            print(f"\n开始导出 {name.upper()}...", flush=True)
            result = Path(model.export(**options)).resolve()
            if not result.is_file() or result.stat().st_size == 0:
                raise RuntimeError(f"未生成有效引擎文件：{result}")
            print(f"{name.upper()} 导出成功：{result}", flush=True)
        except Exception as exc:
            failures.append(f"{name.upper()}: {type(exc).__name__}: {exc}")
            print(f"导出失败：{failures[-1]}", flush=True)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit("部分导出失败，成功文件已保留：\n" + "\n".join(failures))
    print(f"\nFP16 和 INT8 均导出成功：{output}")


if __name__ == "__main__":
    main()
