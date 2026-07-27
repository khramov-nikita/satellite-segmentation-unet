"""Визуализация эталонных и предсказанных масок."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import CLASS_COLORS, CLASS_NAMES
from data import list_pairs, load_pair
from io_utils import imwrite_rgb


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Раскрашивает целочисленную маску классов в RGB по легенде."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def build_class_legend_handles() -> list[Patch]:
    """Создаёт цветные маркеры легенды (квадрат + имя класса)."""
    handles = []
    for cls, name in enumerate(CLASS_NAMES):
        rgb = np.array(CLASS_COLORS[cls], dtype=np.float64) / 255.0
        handles.append(Patch(facecolor=rgb, edgecolor="black", linewidth=0.8, label=name))
    return handles


def main() -> None:
    """Точка входа: сохраняет визуализации сегментации и легенду классов."""
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

        fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
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

        handles = build_class_legend_handles()
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(CLASS_NAMES),
            fontsize=9,
            frameon=True,
            title="Легенда классов",
            bbox_to_anchor=(0.5, 1.02),
        )
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        out_path = args.out / f"viz_test_{i + 1}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")

    # отдельный файл легенды для приложений
    legend = np.zeros((60, 500, 3), dtype=np.uint8)
    x = 10
    for cls, name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS[cls]
        cv2.rectangle(legend, (x, 15), (x + 30, 45), color[::-1], -1)
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
