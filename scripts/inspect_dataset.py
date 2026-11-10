from __future__ import annotations

import argparse
from torch.utils.data import DataLoader

from common import label_dir, load_data_config
from src.data import DefectDataset, defect_collate


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data", default="D:/pycharm/PycharmProjects/PythonProject1/PCB_defect/configs/PCB_defect.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    cfg, root = load_data_config(args.data)
    image_dir = root / cfg["train"]
    dataset = DefectDataset(image_dir, label_dir(image_dir, root), args.imgsz)
    images, labels = next(iter(DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=defect_collate)))
    print(f"Dataset: {len(dataset)} images; batch images: {tuple(images.shape)}")
    print("objects per image:", [len(x) for x in labels])


if __name__ == "__main__": main()
