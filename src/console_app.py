# -*- coding: utf-8 -*-
"""Простое консольное меню для пайплайна сегментации спутниковых снимков."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent


def ask_int(prompt: str, default: int) -> int:
    """Спрашивает целое число; пустой ввод возвращает значение по умолчанию."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Некорректное число, использую {default}.")
        return default


def ask_float(prompt: str, default: float) -> float:
    """Спрашивает вещественное число; пустой ввод возвращает значение по умолчанию."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Некорректное число, использую {default}.")
        return default


def run_script(script_name: str, args: list[str] | None = None) -> bool:
    """Запускает скрипт из src/ через текущий интерпретатор Python."""
    script = SRC / script_name
    cmd = [sys.executable, str(script), *(args or [])]
    print("\n>>>", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return False
    if result.returncode != 0:
        print(f"Ошибка: код возврата {result.returncode}")
        return False
    print(f"Готово (код {result.returncode}).")
    return True


def action_generate() -> None:
    """Генерирует синтетический датасет с параметрами пользователя."""
    print("\n— Генерация датасета —")
    train = ask_int("Число train", 140)
    val = ask_int("Число val", 30)
    test = ask_int("Число test", 30)
    size = ask_int("Размер патча", 256)
    run_script(
        "generate_dataset.py",
        ["--train", str(train), "--val", str(val), "--test", str(test), "--size", str(size)],
    )


def action_train() -> None:
    """Обучает U-Net с параметрами пользователя."""
    print("\n— Обучение U-Net —")
    epochs = ask_int("Число эпох", 20)
    batch = ask_int("Batch size", 4)
    lr = ask_float("Learning rate", 0.001)
    run_script(
        "train.py",
        ["--epochs", str(epochs), "--batch-size", str(batch), "--lr", str(lr)],
    )


def action_evaluate() -> None:
    """Оценивает модель на тестовой выборке."""
    print("\n— Оценка на test —")
    run_script("evaluate.py")


def action_visualize() -> None:
    """Строит визуализации сегментации."""
    print("\n— Визуализация —")
    n = ask_int("Число примеров", 3)
    run_script("visualize.py", ["--n", str(n)])


def full_pipeline() -> None:
    """Запрашивает параметры один раз и выполняет шаги 1→2→3→4."""
    print("\n— Полный пайплайн (генерация → обучение → оценка → визуализация) —")
    train = ask_int("Число train", 140)
    val = ask_int("Число val", 30)
    test = ask_int("Число test", 30)
    size = ask_int("Размер патча", 256)
    epochs = ask_int("Число эпох", 20)
    batch = ask_int("Batch size", 4)
    lr = ask_float("Learning rate", 0.001)
    n_viz = ask_int("Число примеров визуализации", 3)

    steps = [
        (
            "generate_dataset.py",
            ["--train", str(train), "--val", str(val), "--test", str(test), "--size", str(size)],
        ),
        ("train.py", ["--epochs", str(epochs), "--batch-size", str(batch), "--lr", str(lr)]),
        ("evaluate.py", []),
        ("visualize.py", ["--n", str(n_viz)]),
    ]
    for i, (script, args) in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] {script}")
        if not run_script(script, args):
            print("Пайплайн остановлен из-за ошибки.")
            return
    print("\nПолный пайплайн завершён успешно.")


def print_menu() -> None:
    """Печатает главное меню."""
    print(
        """
========================================
  Сегментация спутниковых снимков (U-Net)
========================================
  1. Сгенерировать датасет
  2. Обучить U-Net
  3. Оценить на test
  4. Визуализировать результаты
  5. Полный пайплайн (1→2→3→4)
  0. Выход
----------------------------------------"""
    )


def main() -> None:
    """Точка входа: цикл интерактивного меню."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    print(f"Корень проекта: {ROOT}")
    while True:
        print_menu()
        choice = input("Выберите пункт: ").strip()
        if choice == "1":
            action_generate()
        elif choice == "2":
            action_train()
        elif choice == "3":
            action_evaluate()
        elif choice == "4":
            action_visualize()
        elif choice == "5":
            full_pipeline()
        elif choice == "0":
            print("Выход.")
            break
        else:
            print("Неизвестный пункт. Введите 0–5.")
        input("\nНажмите Enter, чтобы вернуться в меню…")


if __name__ == "__main__":
    main()
