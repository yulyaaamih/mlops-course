# Таблица лидеров: топ-5 по ROC-AUC

Эксперимент MLflow: `churn`

| Выбрана | Прогон | Модель | Параметры | ROC-AUC | PR-AUC | git_sha | Дата |
|---|---|---|---|---|---|---|---|
| best_model | sincere-carp-91 | `logreg` | C=0.1, max_iter=1000 | 0.8087 | 0.5195 | 8aa7b6b | 2026-10-16 09:54 |
|  | abrasive-worm-619 | `logreg` | C=1.0, max_iter=1000 | 0.8086 | 0.5193 | 8aa7b6b | 2026-10-15 14:27 |
|  | dapper-loon-630 | `gradient_boosting` | learning_rate=0.05, max_depth=3, n_estimators=200 | 0.8044 | 0.5017 | 8aa7b6b | 2026-10-15 14:23 |
|  | nervous-midge-759 | `random_forest` | max_depth=5, min_samples_leaf=3, n_estimators=400 | 0.8013 | 0.4955 | 8aa7b6b | 2026-10-15 14:24 |
|  | burly-bird-3 | `random_forest` | max_depth=5, min_samples_leaf=3, n_estimators=157 | 0.8012 | 0.4955 | 8aa7b6b | 2026-10-15 14:24 |
