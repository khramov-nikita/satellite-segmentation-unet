"""Обучение U-Net на синтетическом land-cover датасете."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import make_dataset, one_hot_masks
from metrics import dice_coef_metric, mean_iou_metric
from model_unet import build_unet


def plot_history(history: dict, out_path: Path) -> None:
    """Строит и сохраняет графики loss и mean IoU по эпохам."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.get("loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    key = "mean_iou_metric"
    val_key = "val_mean_iou_metric"
    if key in history:
        axes[1].plot(history[key], label="train IoU")
        axes[1].plot(history.get(val_key, []), label="val IoU")
    axes[1].set_title("Mean IoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Точка входа: обучает U-Net, сохраняет веса, историю и кривые обучения."""
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "experiment" / "data")
    parser.add_argument("--out", type=Path, default=root / "experiment" / "outputs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-filters", type=int, default=32)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(args.data, "train", args.batch_size, shuffle=True, augment_on=True)
    val_ds = make_dataset(args.data, "val", args.batch_size, shuffle=False, augment_on=False)

    train_ds = train_ds.map(lambda x, y: (x, one_hot_masks(y)), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x, one_hot_masks(y)), num_parallel_calls=tf.data.AUTOTUNE)

    model = build_unet(base_filters=args.base_filters)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", mean_iou_metric, dice_coef_metric],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mean_iou_metric",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.out / "unet_best.keras"),
            monitor="val_mean_iou_metric",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.out / "unet_final.keras")
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (args.out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    plot_history(hist, args.out / "training_curves.png")

    meta = {
        "epochs_requested": args.epochs,
        "epochs_ran": len(hist.get("loss", [])),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "base_filters": args.base_filters,
        "optimizer": "Adam",
        "loss": "categorical_crossentropy",
        "best_val_mean_iou": max(hist.get("val_mean_iou_metric", [0.0])),
        "best_val_accuracy": max(hist.get("val_accuracy", [0.0])),
    }
    (args.out / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
