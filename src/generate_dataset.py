"""Генерация синтетических RGB-патчей и масок земного покрова 256x256."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import CLASS_NAMES, NUM_CLASSES
from io_utils import imwrite_mask, imwrite_rgb


def _fill_background(mask: np.ndarray, rng: np.random.Generator) -> None:
    """Заполняет маску классом «прочее» (фон)."""
    mask[:, :] = 0
    # лёгкий «шум» фона как прочее
    noise = rng.random(mask.shape) < 0.02
    mask[noise] = 0


def _draw_fields(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    """Рисует прямоугольные области класса «посевы/поля»."""
    h, w = mask.shape
    for _ in range(int(rng.integers(2, 5))):
        x1, y1 = int(rng.integers(0, w // 2)), int(rng.integers(0, h // 2))
        x2, y2 = int(rng.integers(x1 + 40, w)), int(rng.integers(y1 + 40, h))
        color = (
            int(rng.integers(40, 90)),
            int(rng.integers(120, 200)),
            int(rng.integers(40, 100)),
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)


def _draw_forest(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    """Рисует эллиптические области класса «лес»."""
    h, w = mask.shape
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        axes = (int(rng.integers(30, 80)), int(rng.integers(30, 80)))
        angle = int(rng.integers(0, 180))
        color = (
            int(rng.integers(20, 60)),
            int(rng.integers(80, 160)),
            int(rng.integers(20, 60)),
        )
        cv2.ellipse(image, (cx, cy), axes, angle, 0, 360, color, -1)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 2, -1)


def _draw_water(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    """Рисует полигональную область класса «вода»."""
    h, w = mask.shape
    pts = np.array(
        [
            [
                int(rng.integers(0, w)),
                int(rng.integers(0, h)),
            ]
            for _ in range(int(rng.integers(4, 7)))
        ],
        dtype=np.int32,
    )
    color = (
        int(rng.integers(150, 220)),
        int(rng.integers(80, 140)),
        int(rng.integers(20, 60)),
    )
    cv2.fillPoly(image, [pts], color)
    cv2.fillPoly(mask, [pts], 3)


def _draw_urban(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> None:
    """Рисует компактные прямоугольники класса «застройка»."""
    h, w = mask.shape
    for _ in range(int(rng.integers(3, 8))):
        x1, y1 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
        bw, bh = int(rng.integers(12, 40)), int(rng.integers(12, 40))
        color = (
            int(rng.integers(80, 160)),
            int(rng.integers(80, 160)),
            int(rng.integers(80, 160)),
        )
        cv2.rectangle(image, (x1, y1), (x1 + bw, y1 + bh), color, -1)
        cv2.rectangle(mask, (x1, y1), (x1 + bw, y1 + bh), 4, -1)


def synthesize_sample(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Синтезирует один RGB-патч и соответствующую маску классов."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    # базовый фон — «почва/прочее»
    base = int(rng.integers(70, 110))
    image[:, :] = (base - 10, base, base - 20)
    image = image + rng.integers(-8, 9, size=image.shape, dtype=np.int16)
    image = np.clip(image, 0, 255).astype(np.uint8)

    mask = np.zeros((size, size), dtype=np.uint8)
    _fill_background(mask, rng)
    _draw_fields(image, mask, rng)
    _draw_forest(image, mask, rng)
    _draw_water(image, mask, rng)
    _draw_urban(image, mask, rng)
    # лёгкое размытие изображения как «атмосфера»
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image, mask


def save_split(
    out_dir: Path,
    split: str,
    n: int,
    size: int,
    seed: int,
) -> dict:
    """Генерирует и сохраняет выборку split (train/val/test), возвращает статистику."""
    img_dir = out_dir / split / "images"
    mask_dir = out_dir / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    class_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)

    for i in range(n):
        image, mask = synthesize_sample(size, rng)
        stem = f"{split}_{i:04d}"
        # image уже в BGR-логике рисования OpenCV? рисуем в ndarray как BGR-цвета в RGB-слотах —
        # для синтеза цвета задавались как (B,G,R)-подобные кортежи в cv2; сохраняем как RGB-массив
        imwrite_rgb(img_dir / f"{stem}.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        imwrite_mask(mask_dir / f"{stem}.png", mask)
        for c in range(NUM_CLASSES):
            class_pixels[c] += int(np.sum(mask == c))

    return {
        "split": split,
        "num_samples": n,
        "class_pixel_counts": {CLASS_NAMES[c]: int(class_pixels[c]) for c in range(NUM_CLASSES)},
    }


def main() -> None:
    """Точка входа: создаёт полный синтетический датасет и пишет dataset_meta.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiment" / "data",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--train", type=int, default=140)
    parser.add_argument("--val", type=int, default=30)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stats = [
        save_split(args.out, "train", args.train, args.size, args.seed),
        save_split(args.out, "val", args.val, args.size, args.seed + 1),
        save_split(args.out, "test", args.test, args.size, args.seed + 2),
    ]
    meta = {
        "size": args.size,
        "classes": CLASS_NAMES,
        "splits": stats,
        "note": "Синтетический land-cover датасет для воспроизводимого эксперимента КП",
    }
    (args.out / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
