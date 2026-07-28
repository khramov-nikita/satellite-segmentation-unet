"""Чтение/запись изображений с поддержкой путей Unicode (Windows)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def print_as_list(data: Any, indent: int = 0) -> None:
    """Печатает dict/list как маркированный список (удобно вместо сырого JSON)."""
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}- {key}:")
                print_as_list(value, indent + 1)
            else:
                print(f"{prefix}- {key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                print(f"{prefix}- [{i}]:")
                print_as_list(item, indent + 1)
            else:
                print(f"{prefix}- {item}")
    else:
        print(f"{prefix}- {data}")


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """Сохраняет изображение (BGR/серое) по пути с поддержкой Unicode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        Image.fromarray(image.astype(np.uint8), mode="L").save(path)
        return
    # ожидаем BGR (как у OpenCV) или RGB — сохраняем как RGB через конверсию если 3 канала
    if image.shape[2] == 3:
        # если вызывающий передал BGR — конвертируем в RGB для PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)
    else:
        Image.fromarray(image.astype(np.uint8)).save(path)


def imwrite_rgb(path: Path, image_rgb: np.ndarray) -> None:
    """Сохраняет RGB-изображение в PNG/JPEG с поддержкой Unicode-путей."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(path)


def imwrite_mask(path: Path, mask: np.ndarray) -> None:
    """Сохраняет одноканальную маску классов как grayscale-изображение."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def imread_rgb(path: Path) -> np.ndarray:
    """Читает изображение и возвращает массив RGB uint8."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def imread_mask(path: Path) -> np.ndarray:
    """Читает маску классов как одноканальный массив."""
    with Image.open(path) as img:
        return np.array(img.convert("L"))
