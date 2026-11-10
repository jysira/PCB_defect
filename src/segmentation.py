"""COCO polygon to semantic-mask dataset for PCB defect segmentation."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CocoSemanticDataset(Dataset):
    """Read Roboflow COCO polygon annotations as class-index masks.

    Background is 0; defect classes are mapped to 1..N in sorted COCO
    category-id order. The category named ``PCB`` is ignored as background.
    """

    def __init__(self, split_dir: str | Path, image_size: int = 640):
        self.split_dir = Path(split_dir)
        annotation_file = self.split_dir / "_annotations.coco.json"
        if not annotation_file.is_file():
            raise FileNotFoundError(f"缺少 COCO 标注文件: {annotation_file}")
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        self.image_size = image_size
        self.categories = sorted(
            (c for c in data["categories"] if c["name"].lower() != "pcb"),
            key=lambda c: c["id"],
        )
        self.class_names = [c["name"] for c in self.categories]
        self.category_to_label = {c["id"]: i + 1 for i, c in enumerate(self.categories)}
        self.images = sorted(data["images"], key=lambda x: x["file_name"])
        self.annotations: dict[int, list[dict]] = {}
        for annotation in data["annotations"]:
            if annotation["category_id"] in self.category_to_label:
                self.annotations.setdefault(annotation["image_id"], []).append(annotation)
        missing = [x["file_name"] for x in self.images if not (self.split_dir / x["file_name"]).is_file()]
        if missing:
            raise FileNotFoundError(f"COCO 中有 {len(missing)} 张图像缺失，例如: {missing[0]}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.images[index]
        image_path = self.split_dir / record["file_name"]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"无法读取图像: {image_path}")
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for annotation in self.annotations.get(record["id"], []):
            label = self.category_to_label[annotation["category_id"]]
            segmentation = annotation.get("segmentation", [])
            if isinstance(segmentation, list):
                polygons = []
                for points in segmentation:
                    if len(points) >= 6 and len(points) % 2 == 0:
                        polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
                        polygons.append(np.rint(polygon).astype(np.int32))
                if polygons:
                    cv2.fillPoly(mask, polygons, int(label))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        image_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image_tensor, mask_tensor
