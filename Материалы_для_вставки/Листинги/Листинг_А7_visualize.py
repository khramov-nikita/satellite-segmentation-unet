"""Визуализация эталонных и предсказанных масок."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import CLASS_COLORS, CLASS_NAMES
from data import list_pairs, load_pair
from io_utils import imwrite_rgb


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "experiment" / "data")
    parser.add_argument("--out", type=Path, default=root / "experiment" / "outputs")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    weights = args.out / "unet_best.keras"
    if not weights.exists():
        weights = args.out / "unet_final.keras"

    model = tf.keras.models.load_model(weights, compile=False)
    pairs = list_pairs(args.data / "test")[: args.n]
    args.out.mkdir(parents=True, exist_ok=True)

    for i, (img_path, mask_path) in enumerate(pairs):
        image, mask = load_pair(img_path, mask_path)
        pred = model.predict(image[None, ...], verbose=0)[0]
        pred_cls = np.argmax(pred, axis=-1).astype(np.int32)
        err = (mask != pred_cls).astype(np.uint8) * 255

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(image)
        axes[0].set_title("Снимок")
        axes[1].imshow(colorize_mask(mask))
        axes[1].set_title("Эталон")
        axes[2].imshow(colorize_mask(pred_cls))
        axes[2].set_title("Предсказание")
        axes[3].imshow(err, cmap="gray")
        axes[3].set_title("Ошибки")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(" | ".join(CLASS_NAMES), fontsize=9)
        fig.tight_layout()
        out_path = args.out / f"viz_test_{i + 1}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")

    # легенда
    legend = np.zeros((60, 500, 3), dtype=np.uint8)
    x = 10
    for cls, name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS[cls]
        cv2.rectangle(legend, (x, 15), (x + 30, 45), color[::-1], -1)  # BGR for cv2
        cv2.putText(
            legend,
            name,
            (x + 35, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        x += 100
    imwrite_rgb(args.out / "legend.png", cv2.cvtColor(legend, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
