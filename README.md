# Churn MLOps

Сквозной проект курса MLOps: предсказание оттока клиентов телеком-оператора
(бинарная классификация, табличные данные). К концу семестра из него вырастет
ML-сервис с пайплайном, тестами, CI/CD и мониторингом.

## Как запустить

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
make install                       # зависимости
make check                         # должно напечатать environment: OK
make data                          # сгенерировать data/raw/churn.csv
make prepare                       # train/val/test в data/processed/
make train                         # обучить модель, метрики на val
```

Остальные команды: `make test` (тесты), `make lint` (ruff). Список всех целей — `make help`.

Результаты обучения:
- `models/model.joblib` — весь Pipeline (препроцессор + модель), в Git не хранится;
- `reports/train_metrics.json` — ROC-AUC, PR-AUC и F1 на val.

## Как это устроено

| Стадия | Файл | Что делает |
|---|---|---|
| данные | `src/data/generate.py` | генерирует синтетический датасет с фиксированным seed |
| prepare | `src/data/prepare.py` | проверяет колонки, заполняет пропуски в `total_charges` (`monthly_charges * tenure_months`), делит на train/val/test (70/10/20) со stratify по `churn` |
| признаки | `src/features.py` | `ColumnTransformer`: числовые (медиана + масштабирование), категориальные (самая частая + one-hot), бинарные без изменений |
| train | `src/train.py` | Pipeline из препроцессора и модели; модель выбирается в `params.yaml`, метрики считаются на val |

Все настройки лежат в `params.yaml`: seed, размеры выборок, списки признаков,
модель и её гиперпараметры. Чтобы сменить модель, достаточно поменять
`train.model` на `logreg`, `random_forest` или `gradient_boosting` — код при
этом не меняется.

## Результаты экспериментов

Метрики на val, подробности и вывод — в [reports/EXPERIMENTS.md](reports/EXPERIMENTS.md).

| Модель | ROC-AUC |
|---|---|
| `logreg` | 0.809 |
| `gradient_boosting` | 0.804 |
| `random_forest` | 0.801 |

## Правила проекта

1. **Никаких абсолютных путей.** Только `src.config.resolve()`.
2. **Никаких магических чисел в коде.** Всё числовое — в `params.yaml`.
3. **Данные и модели не коммитятся в Git.** С занятия 4 — в DVC.
4. **Секреты — только через переменные окружения.** Имена переменных — в `.env.example`, значения в `.env` (он в `.gitignore`).
5. **Работа идёт через ветки и Pull Request.** Перед коммитом срабатывает `pre-commit`
   (`pre-commit install` один раз после клонирования).

## Что уже сделано

- **Занятие 1.** Код из ноутбука превращён в воспроизводимые модули: два запуска
  подряд дают одинаковые файлы и метрики. Найденные проблемы ноутбука — в
  [notebooks/PROBLEMS.md](notebooks/PROBLEMS.md).
- **Занятие 2.** Настроены `pre-commit` с `ruff`, `.env.example`, защита ветки `main`.
- **ДЗ 1.** Выбор модели через `params.yaml`, сравнение трёх моделей.
