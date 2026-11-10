from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_data_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (config_path.parent.parent / root).resolve()
    return cfg, root


def names_from_config(cfg: dict) -> dict[int, str]:
    names = cfg["names"]
    return {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))


def label_dir(image_dir: Path, dataset_root: Path) -> Path:
    return dataset_root / "labels" / image_dir.name
