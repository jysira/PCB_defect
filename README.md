# PCB 项目运行索引

在项目根目录激活 `industrial_vision` 环境后执行下面的命令。

| 用途 | 数据配置 | 保留结果 |
|---|---|---|
| 六类目标检测 | `data/pcb_detection_clahe/dataset.yaml` | `runs/train/PCB_defect_clahe` |
| 六类语义分割原图基线 | `data/pcb_semantic_merged_grouped/dataset.yaml` | `runs/semantic_merged/pcb` |
| 六类语义分割 CLAHE | `data/pcb_semantic_merged_clahe/dataset.yaml` | `runs/semantic_clahe/pcb` |
| 原图分割验证与测试 | 同原图基线 | `runs/semantic_merged_eval` |
| 分割 ONNX 实际图片验证 | CLAHE 测试图 | `runs/deploy_verify/semantic_onnx` |

`data/images` 与 `data/labels` 是检测原始数据。`data/pcb_reviewed_300`、`semantic_pcb/dataset` 和 `semantic_final/dataset` 是合并分割数据的来源。检测 `.pt`、`.onnx` 和 TensorRT 引擎保存在对应检测运行的 `weights` 下。分割 ONNX 位于 `runs/semantic_clahe/pcb/weights/best.onnx`。

```cmd
conda activate industrial_vision
python scripts/train.py --data data/pcb_detection_clahe/dataset.yaml
python scripts/evaluate.py --split test
python scripts/yolo_semantic.py train
python scripts/train_semantic_clahe.py
python scripts/yolo_semantic.py val --model runs/semantic_merged/pcb/weights/best.pt --split test
python scripts/yolo_semantic.py val --model runs/semantic_clahe/pcb/weights/best.pt --data data/pcb_semantic_merged_clahe/dataset.yaml --split test
```

导出分割 TensorRT FP16 时使用当前 GPU 环境构建引擎。

```cmd
yolo semantic export model=runs/semantic_clahe/pcb/weights/best.pt format=engine quantize=16 imgsz=640 batch=1 device=0
```

历史 `args.yaml` 用于追溯原始运行，可能仍记录整理前的路径。旧实验目录从工作目录移入 Windows 回收站。
