"""Оценка модели на test и сохранение метрик."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import list_pairs, load_pair
from io_utils import print_as_list
from metrics import compute_numpy_metrics


def main() -> None:
    """Точка входа: оценивает модель на test и сохраняет test_metrics.json."""
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "experiment" / "data")
    parser.add_argument("--out", type=Path, default=root / "experiment" / "outputs")
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()

    weights = args.weights or (args.out / "unet_best.keras")
    if not weights.exists():
        weights = args.out / "unet_final.keras"

    model = tf.keras.models.load_model(weights, compile=False)

    pairs = list_pairs(args.data / "test")
    y_true_all = []
    y_pred_all = []
    for img_path, mask_path in pairs:
        image, mask = load_pair(img_path, mask_path)
        pred = model.predict(image[None, ...], verbose=0)[0]
        pred_cls = np.argmax(pred, axis=-1).astype(np.int32)
        y_true_all.append(mask)
        y_pred_all.append(pred_cls)

    y_true = np.stack(y_true_all)
    y_pred = np.stack(y_pred_all)
    metrics = compute_numpy_metrics(y_true, y_pred)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nРезультат (test_metrics):")
    print_as_list(metrics)


if __name__ == "__main__":
    main()
