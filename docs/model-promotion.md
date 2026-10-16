# Регламент продвижения моделей

Модель `churn-classifier` в MLflow Model Registry. Сервис грузит версию
с алиасом `champion` (`models:/churn-classifier@champion`), поэтому при переводе
в `Production` на версию ставится и стадия, и алиас.

## Жизненный цикл версии

| Стадия | Когда |
|---|---|
| `None` | версия только что зарегистрирована после `python -m src.train` |
| `Staging` | версия прошла условия для Staging, идёт проверка |
| `Production` | версия прошла проверки, на ней алиас `champion` |
| `Archived` | версия снята из Production, хранится для отката |

## Условия перевода в Staging

Версия переводится в `Staging`, если:

* прогон в MLflow полный: есть параметры, метрики, теги `git_sha` и `data_md5`,
  артефакты `params.yaml`, ROC-кривая и матрица ошибок;
* `git_sha` указывает на закоммиченный код, а не на рабочую копию с правками;
* ROC-AUC на валидации не ниже `evaluate.min_roc_auc` (0.75) из `params.yaml`;
* у модели есть сигнатура (модель залогирована с `input_example`).

## Проверки перед Production

Версия из `Staging` переводится в `Production`, только если все проверки пройдены:

1. **Качество на test.** `make eval` проходит порог `min_roc_auc`, и ROC-AUC
   не хуже, чем у текущей версии с алиасом `champion`. Сравнение на одной и той же
   тестовой выборке (одинаковый `data_md5`).
2. **Тесты зелёные.** `make test` и `make lint` проходят на коммите из `git_sha`.
3. **Сигнатура не изменилась.** Набор входных колонок совпадает с `features`
   в `params.yaml`. Это же проверяет `model_loader` при загрузке, но лучше
   поймать расхождение до продвижения, а не при старте сервиса.
4. **Порог пересмотрен.** Если поменялся алгоритм, порог `evaluate.threshold`
   заново подобран по валидации, обоснование записано в `README.md`.

После перевода: прежняя версия уходит в `Archived`, алиас `champion` переставляется
на новую, `make leaderboard` обновляет `reports/LEADERBOARD.md`.

## Кто принимает решение

В учебном проекте решение принимает автор проекта. В команде решение
принимает ответственный за модель. Перед переводом в `Production` второй
человек смотрит результаты проверок выше и подтверждает перевод. Откат
может выполнить любой дежурный без согласования: он безопасен и быстро
обратим.

## Откат на предыдущую версию

### Как выполняется

1. В MLflow UI: Models, `churn-classifier`, проблемная версия, Stage, **Transition to Archived**.
2. Предыдущая рабочая версия: Stage, **Transition to Production**.
3. Перезагрузить модель в сервисе.

Переобучение не нужно: все версии уже лежат в реестре, меняется только метка.

Время: около 30 секунд.

### Проверка

Инцидент: в `Production` стояла версия 5 (`GradientBoostingClassifier`).
Версия 5 переведена в `Archived`, версия 2 (`LogisticRegression`) в `Production`.

Команда для проверки, какая модель загрузилась:

```bash
python -W ignore -c "from src.service.model_loader import holder; from mlflow import MlflowClient; holder.load(); v = MlflowClient().get_latest_versions('churn-classifier', stages=['Production'])[0].version; print(f'Production: v{v}, {holder.model[-1]}')"
```

До отката:

```
Production: v5, GradientBoostingClassifier(n_estimators=200, random_state=42)
```

После отката:

```
Production: v2, LogisticRegression(C=0.1, max_iter=1000, random_state=42)
```

### Откат на алиасах

После перехода загрузчика на алиас откат делается так же быстро: на странице
предыдущей версии в поле Aliases поставить `champion`. Алиас сам снимется
с текущей версии, потому что у модели он может быть только на одной версии.
Проверка:

```bash
python -W ignore -c "from src.service.model_loader import holder; holder.load(); print(holder.version)"
```
