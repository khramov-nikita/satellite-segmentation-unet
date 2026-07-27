"""Архитектура U-Net для семантической сегментации."""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classes import NUM_CLASSES


def conv_block(x: tf.Tensor, filters: int, dropout: float = 0.0) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def build_unet(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    num_classes: int = NUM_CLASSES,
    base_filters: int = 32,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = conv_block(p2, base_filters * 4, dropout=0.1)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = conv_block(p3, base_filters * 8, dropout=0.2)
    p4 = layers.MaxPooling2D(2)(c4)

    bn = conv_block(p4, base_filters * 16, dropout=0.3)

    u5 = layers.UpSampling2D(2)(bn)
    u5 = layers.Concatenate()([u5, c4])
    c5 = conv_block(u5, base_filters * 8, dropout=0.2)

    u6 = layers.UpSampling2D(2)(c5)
    u6 = layers.Concatenate()([u6, c3])
    c6 = conv_block(u6, base_filters * 4, dropout=0.1)

    u7 = layers.UpSampling2D(2)(c6)
    u7 = layers.Concatenate()([u7, c2])
    c7 = conv_block(u7, base_filters * 2)

    u8 = layers.UpSampling2D(2)(c7)
    u8 = layers.Concatenate()([u8, c1])
    c8 = conv_block(u8, base_filters)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(c8)
    return models.Model(inputs, outputs, name="unet_landcover")
