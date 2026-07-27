"""Метрики сегментации: IoU, Dice, pixel accuracy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import CLASS_NAMES, NUM_CLASSES


def mean_iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_cls = tf.argmax(y_true, axis=-1)
    y_pred_cls = tf.argmax(y_pred, axis=-1)
    ious = []
    for c in range(NUM_CLASSES):
        yt = tf.cast(tf.equal(y_true_cls, c), tf.float32)
        yp = tf.cast(tf.equal(y_pred_cls, c), tf.float32)
        intersection = tf.reduce_sum(yt * yp)
        union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - intersection
        ious.append(tf.where(union > 0, intersection / union, 0.0))
    return tf.reduce_mean(tf.stack(ious))


def dice_coef_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_cls = tf.argmax(y_true, axis=-1)
    y_pred_cls = tf.argmax(y_pred, axis=-1)
    dices = []
    for c in range(NUM_CLASSES):
        yt = tf.cast(tf.equal(y_true_cls, c), tf.float32)
        yp = tf.cast(tf.equal(y_pred_cls, c), tf.float32)
        intersection = tf.reduce_sum(yt * yp)
        denom = tf.reduce_sum(yt) + tf.reduce_sum(yp)
        dices.append(tf.where(denom > 0, 2.0 * intersection / denom, 0.0))
    return tf.reduce_mean(tf.stack(dices))


def compute_numpy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred: [N,H,W] int labels."""
    acc = float(np.mean(y_true == y_pred))
    per_class_iou = []
    per_class_dice = []
    for c in range(NUM_CLASSES):
        yt = y_true == c
        yp = y_pred == c
        inter = np.logical_and(yt, yp).sum()
        union = np.logical_or(yt, yp).sum()
        denom = yt.sum() + yp.sum()
        iou = float(inter / union) if union > 0 else 0.0
        dice = float(2 * inter / denom) if denom > 0 else 0.0
        per_class_iou.append(iou)
        per_class_dice.append(dice)
    return {
        "pixel_accuracy": acc,
        "mean_iou": float(np.mean(per_class_iou)),
        "mean_dice": float(np.mean(per_class_dice)),
        "per_class_iou": {CLASS_NAMES[i]: per_class_iou[i] for i in range(NUM_CLASSES)},
        "per_class_dice": {CLASS_NAMES[i]: per_class_dice[i] for i in range(NUM_CLASSES)},
    }
