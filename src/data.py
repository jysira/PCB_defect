from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(directory: str | Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


class DefectDataset(Dataset):
    """读取 YOLO 格式标签的独立 PyTorch Dataset。

    返回 RGB float Tensor [C,H,W] 与 labels [N,5]，标签列为
    [class_id, x_center, y_center, width, height]（归一化坐标）。
    """

    def __init__(self, image_dir: str | Path, label_dir: str | Path,
                 image_size: int = 640, transform: Callable | None = None):
        self.images = image_files(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.transform = transform
        if not self.images:
            raise FileNotFoundError(f"未在 {image_dir} 找到图像")

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, image_path: Path) -> Path:
        return self.label_dir / image_path.relative_to(self.images[0].parent).with_suffix(".txt")

    def _read_labels(self, path: Path) -> np.ndarray:
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"无效 YOLO 标签 {path}: {line}")
            rows.append([float(x) for x in values])
        labels = np.asarray(rows, dtype=np.float32)
        if np.any(labels[:, 1:] < 0) or np.any(labels[:, 1:] > 1):
            raise ValueError(f"标签坐标超出 [0, 1]: {path}")
        return labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"无法读取图片: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        labels = self._read_labels(self._label_path(image_path))
        if self.transform:
            rgb, labels = self.transform(rgb, labels)
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        return image, torch.from_numpy(labels)


def defect_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """目标数量不同，标签保留为按图像分组的 list。"""
    images, labels = zip(*batch)
    return torch.stack(images), list(labels)
