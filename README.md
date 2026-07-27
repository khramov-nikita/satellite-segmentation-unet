# Курсовой проект: обработка спутниковых снимков посредством нейронных сетей

## Документы

- `Курсовой_проект.md` — полный текст курсового проекта
- `План_курсового_проекта.md` — рабочий план написания
- `Искусственные нейронные сети. Книги/` — учебная литература (Хайкин)

## Эксперимент

Синтетический land-cover датасет + U-Net (TensorFlow/Keras).

```bash
python -m pip install -r requirements.txt
python src/generate_dataset.py --train 140 --val 30 --test 30
python src/train.py --epochs 20 --batch-size 4
python src/evaluate.py
python src/visualize.py --n 3
```

Результаты: `experiment/outputs/` (веса, метрики, визуализации).

## Структура кода

| Путь | Назначение |
|------|------------|
| `src/` | исходники прототипа |
| `experiment/data/` | train/val/test патчи |
| `experiment/outputs/` | артефакты обучения |
