"""Загрузка патчей и аугментации."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import NUM_CLASSES
from io_utils import imread_mask, imread_rgb


def list_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    """Собирает пары путей (изображение, маска) для указанной выборки."""
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    images = sorted(img_dir.glob("*.png"))
    pairs = []
    for img_path in images:
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def load_pair(img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Читает RGB-снимок и маску классов, нормализует изображение в [0, 1]."""
    image = imread_rgb(img_path).astype(np.float32) / 255.0
    mask = imread_mask(mask_path).astype(np.int32)
    return image, mask


def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Применяет случайные отражения, повороты и изменение яркости к паре снимок–маска."""
    if rng.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        factor = float(rng.uniform(0.8, 1.2))
        image = np.clip(image * factor, 0.0, 1.0)
    return image.copy(), mask.copy()


def make_dataset(
    data_root: Path,
    split: str,
    batch_size: int = 8,
    shuffle: bool = False,
    augment_on: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Создаёт tf.data.Dataset из патчей выборки с опциональными аугментациями."""
    pairs = list_pairs(data_root / split)
    if not pairs:
        raise FileNotFoundError(f"Нет данных в {data_root / split}")

    rng = np.random.default_rng(seed)

    def generator():
        """Итератор, отдающий пары (изображение, маска) для tf.data."""
        idxs = np.arange(len(pairs))
        if shuffle:
            rng.shuffle(idxs)
        for i in idxs:
            image, mask = load_pair(*pairs[int(i)])
            if augment_on:
                image, mask = augment(image, mask, rng)
            yield image, mask

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(pairs), 100), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def one_hot_masks(masks: tf.Tensor) -> tf.Tensor:
    """Преобразует целочисленную маску классов в one-hot тензор."""
    return tf.one_hot(masks, depth=NUM_CLASSES)
