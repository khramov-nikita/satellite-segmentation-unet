# -*- coding: utf-8 -*-
"""Генерация рисунков для главы 1 курсового проекта."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASS_COLORS, CLASS_NAMES
from io_utils import imread_rgb, imread_mask
from visualize import colorize_mask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Материалы_для_вставки" / "Изображения"
OUT.mkdir(parents=True, exist_ok=True)


def fig_1_1_tensor() -> None:
    """Рисует схему снимка как тензора H×W×C (рис. 1.1)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Рис. 1.1. Спутниковый снимок как тензор H × W × C", fontsize=12, pad=12)

    # три «слоя» каналов
    colors = ["#c44e52", "#55a868", "#4c72b0"]
    labels = ["канал R (или B4)", "канал G (или B3)", "канал B / NIR"]
    for i, (c, lab) in enumerate(zip(colors, labels)):
        x0 = 1.2 + i * 0.45
        y0 = 1.2 + i * 0.45
        rect = FancyBboxPatch(
            (x0, y0), 4.2, 3.2,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=c, alpha=0.35, edgecolor=c, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x0 + 4.35, y0 + 2.8, lab, fontsize=9, color=c, va="center")

    ax.annotate("", xy=(5.8, 0.7), xytext=(1.2, 0.7),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(3.5, 0.35, "W (ширина, пиксели)", ha="center", fontsize=10)

    ax.annotate("", xy=(0.7, 4.8), xytext=(0.7, 1.2),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(0.35, 3.0, "H (высота)", ha="center", va="center", rotation=90, fontsize=10)

    ax.text(7.2, 3.5, "C — число\nспектральных\nканалов", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#888"))
    ax.text(7.2, 1.5, "Пиксель = вектор\nяркостей по каналам", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#fff8e7", edgecolor="#c9a227"))

    fig.tight_layout()
    fig.savefig(OUT / "Рис_1_1_тензор_снимка.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_1_2_preprocess() -> None:
    """Сохраняет триптих: исходный фрагмент / нормализация / маска (рис. 1.2)."""
    # берём реальный патч из датасета
    img_dir = ROOT / "experiment" / "data" / "test" / "images"
    mask_dir = ROOT / "experiment" / "data" / "test" / "masks"
    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        raise FileNotFoundError("Нет тестовых изображений для рис. 1.2")

    path = imgs[1] if len(imgs) > 1 else imgs[0]
    image = imread_rgb(path).astype(np.float32)
    mask = imread_mask(mask_dir / path.name)

    # «до» — слегка затемнённый/ненормированный вид
    raw = np.clip(image * 0.65, 0, 255).astype(np.uint8)
    norm = (image / 255.0)
    # локальная нормализация контраста для демонстрации
    p2, p98 = np.percentile(image, (2, 98))
    stretched = np.clip((image - p2) / (p98 - p2 + 1e-6), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(raw)
    axes[0].set_title("Исходный фрагмент")
    axes[1].imshow(stretched)
    axes[1].set_title("После нормализации\nконтраста / [0, 1]")
    axes[2].imshow(colorize_mask(mask))
    axes[2].set_title("Эталонная маска\nклассов")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Рис. 1.2. Предобработка и маска земного покрова", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "Рис_1_2_предобработка_и_маска.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _block(ax, xy, w, h, text, fc="#dceaf7", ec="#2c5aa0"):
    """Рисует скруглённый прямоугольник-блок со подписью на осях matplotlib."""
    rect = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=fc, edgecolor=ec, linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9)


def fig_1_3_cnn_block() -> None:
    """Рисует блок свёртка → ReLU → pooling (рис. 1.3)."""
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Рис. 1.3. Базовый блок CNN: свёртка → ReLU → pooling", fontsize=12, pad=10)

    _block(ax, (0.3, 1.0), 2.0, 1.2, "Вход\nкарта признаков")
    _block(ax, (3.0, 1.0), 2.2, 1.2, "Свёртка\n3×3 / 5×5", fc="#e8f5e9", ec="#2e7d32")
    _block(ax, (5.8, 1.0), 2.0, 1.2, "ReLU\nнелинейность", fc="#fff3e0", ec="#ef6c00")
    _block(ax, (8.4, 1.0), 2.4, 1.2, "Pooling\n↓ H, W", fc="#f3e5f5", ec="#6a1b9a")

    for x0, x1 in [(2.3, 3.0), (5.2, 5.8), (7.8, 8.4)]:
        ax.annotate("", xy=(x1, 1.6), xytext=(x0, 1.6),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))

    ax.text(6, 0.35,
            "Свёртка извлекает локальные признаки; pooling уменьшает размерность\n"
            "и усиливает устойчивость к малым сдвигам (shift invariance).",
            ha="center", fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "Рис_1_3_блок_CNN.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_1_4_unet() -> None:
    """Рисует схему U-Net с encoder, decoder и skip-connections (рис. 1.4)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Рис. 1.4. Схема архитектуры U-Net (encoder–decoder + skip)", fontsize=12, pad=10)

    # encoder (слева вниз)
    enc = [
        (1.0, 6.5, "Enc 1\n256×256"),
        (1.0, 5.0, "Enc 2\n128×128"),
        (1.0, 3.5, "Enc 3\n64×64"),
        (1.0, 2.0, "Enc 4\n32×32"),
    ]
    for x, y, t in enc:
        _block(ax, (x, y), 1.8, 0.9, t, fc="#bbdefb", ec="#1565c0")

    _block(ax, (1.0, 0.5), 1.8, 0.9, "Bottleneck\n16×16", fc="#90caf9", ec="#0d47a1")

    # decoder (справа вверх)
    dec = [
        (7.0, 2.0, "Dec 4\n32×32"),
        (7.0, 3.5, "Dec 3\n64×64"),
        (7.0, 5.0, "Dec 2\n128×128"),
        (7.0, 6.5, "Dec 1\n256×256"),
    ]
    for x, y, t in dec:
        _block(ax, (x, y), 1.8, 0.9, t, fc="#c8e6c9", ec="#2e7d32")

    _block(ax, (7.0, 7.5), 1.8, 0.55, "Softmax → маска", fc="#ffe082", ec="#f9a825")

    # vertical arrows encoder
    for y0, y1 in [(6.5, 5.9), (5.0, 4.4), (3.5, 2.9), (2.0, 1.4)]:
        ax.annotate("", xy=(1.9, y1), xytext=(1.9, y0),
                    arrowprops=dict(arrowstyle="->", color="#1565c0"))
    # bottleneck to first decoder
    ax.annotate("", xy=(7.0, 2.45), xytext=(2.8, 0.95),
                arrowprops=dict(arrowstyle="->", color="#555", connectionstyle="arc3,rad=-0.2"))
    # decoder up
    for y0, y1 in [(2.9, 3.5), (4.4, 5.0), (5.9, 6.5), (7.4, 7.5)]:
        ax.annotate("", xy=(7.9, y1), xytext=(7.9, y0),
                    arrowprops=dict(arrowstyle="->", color="#2e7d32"))

    # skip connections
    for y in (6.95, 5.45, 3.95, 2.45):
        ax.annotate("", xy=(7.0, y), xytext=(2.8, y),
                    arrowprops=dict(arrowstyle="->", color="#c62828", ls="--", lw=1.2))

    ax.text(5.0, 7.2, "skip-connections\n(детали границ)", ha="center", fontsize=9, color="#c62828")
    ax.text(1.9, 7.7, "Энкодер\n↓ признаки", ha="center", fontsize=9, color="#1565c0")
    ax.text(7.9, 0.3, "Декодер ↑ разрешение", ha="center", fontsize=9, color="#2e7d32")

    fig.tight_layout()
    fig.savefig(OUT / "Рис_1_4_схема_UNet.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def update_readme() -> None:
    """Дописывает в README соответствие файлов рисункам главы 1."""
    readme_path = ROOT / "Материалы_для_вставки" / "README.md"
    extra = """
## Рисунки главы 1 (готовые файлы)

| Файл | Куда в работе |
|------|----------------|
| Рис_1_1_тензор_снимка.png | Рисунок 1.1 — снимок как тензор H×W×C |
| Рис_1_2_предобработка_и_маска.png | Рисунок 1.2 — исходный / нормализация / маска |
| Рис_1_3_блок_CNN.png | Рисунок 1.3 — свёртка → ReLU → pooling |
| Рис_1_4_схема_UNet.png | Рисунок 1.4 — схема U-Net |

Пересоздать: `python src/make_chapter1_figures.py`
"""
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
        if "Рис_1_1_тензор_снимка.png" not in text:
            # вставить после заголовка Изображения или в конец
            text = text.replace(
                "Схемы рис. 1.1–1.4, 2.2–2.3 в тексте описаны (в т.ч. Mermaid) — при необходимости нарисуйте в редакторе по описанию в `Курсовой_проект.md`.",
                "Рисунки 1.1–1.4 лежат в этой же папке (см. таблицу ниже). Рис. 2.2–2.3 — схемы метода; 2.3 есть как Mermaid в тексте, при необходимости скопируйте скрин или перерисуйте.",
            )
            text += "\n" + extra
            readme_path.write_text(text, encoding="utf-8")
    else:
        readme_path.write_text(extra, encoding="utf-8")


def main() -> None:
    """Точка входа: генерирует все рисунки главы 1 и обновляет README."""
    fig_1_1_tensor()
    fig_1_2_preprocess()
    fig_1_3_cnn_block()
    fig_1_4_unet()
    update_readme()
    print("Saved to", OUT)
    for p in sorted(OUT.glob("Рис_1_*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
