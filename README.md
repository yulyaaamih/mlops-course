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

## Данные

Датасет `data/raw/churn.csv` и модель `models/model.joblib` версионируются
через [DVC](https://dvc.org). В Git лежат только маленькие файлы-указатели
`data/raw/churn.csv.dvc` и `models/model.joblib.dvc` с md5 содержимого. Сами
файлы хранятся в remote-хранилище. Описание датасета, полей и известных
дефектов — в [data/README.md](data/README.md).

### Где лежит remote

Remote по умолчанию называется `minio`. Это S3-совместимое хранилище
[MinIO](https://min.io):

| Параметр | Значение |
|---|---|
| бакет и путь | `s3://mlops/churn` |
| адрес (endpoint) | `http://localhost:9000` |
| веб-консоль | `http://localhost:9001` |

Настройки лежат в `.dvc/config`. MinIO запущен в Docker на машине автора
проекта, поэтому `localhost` подходит только ей. Остальным нужно либо
подключиться к её MinIO по адресу в локальной сети, либо поднять свой
(см. ниже).

### Как получить данные

```bash
dvc remote modify --local minio access_key_id <ключ>
dvc remote modify --local minio secret_access_key <секрет>
dvc pull
```

`dvc pull` скачает версии данных и модели, на которые указывают `.dvc`-файлы
в текущем коммите. Ключи можно не прописывать через `dvc remote modify`, а
передать переменными окружения `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY`
(их имена есть в `.env.example`).

Флаг `--local` записывает настройки в `.dvc/config.local`. Этот файл не
попадает в Git, поэтому ключи не утекут в репозиторий.

### Новому участнику проекта

1. Склонируйте репозиторий и установите зависимости (раздел «Как запустить»).
   DVC ставится вместе с ними через `make install`.
2. Получите доступ к хранилищу. Есть два варианта:
   - **Подключиться к общему MinIO.** Попросите у автора адрес и ключи и
     пропишите их у себя:
     ```bash
     dvc remote modify --local minio endpointurl http://<адрес>:9000
     dvc remote modify --local minio access_key_id <ключ>
     dvc remote modify --local minio secret_access_key <секрет>
     ```
   - **Поднять свой MinIO.** Пригодится, если общий недоступен:
     ```bash
     docker run -d --name minio -p 9000:9000 -p 9001:9001 \
       -v ~/minio-data:/data quay.io/minio/minio server /data --console-address ":9001"
     ```
     Затем в веб-консоли `http://localhost:9001` создайте бакет `mlops`.
     Своё хранилище будет пустым. Сгенерируйте данные (`make data`), это даст
     тот же файл, что и в DVC: при тех же `seed` и `n_samples` md5 совпадает.
     После этого выполните `dvc push`.
3. Выполните `dvc pull`. Проверьте, что `wc -l data/raw/churn.csv` выдаёт
   `20001`, а `dvc status` пишет, что всё синхронизировано.
4. Дальше работайте как обычно: `make prepare`, `make train`.

**Если вы поменяли данные или модель:**

```bash
dvc add data/raw/churn.csv          # или models/model.joblib
git add data/raw/churn.csv.dvc
git commit -m "..."
dvc push                            # без этого другие не смогут скачать новую версию
```

**Если нужна старая версия данных:** переключитесь на нужный коммит
(`git checkout <коммит>`) и выполните `dvc checkout`. Пример с выводом
терминала — в [reports/ROLLBACK.md](reports/ROLLBACK.md).

### Почему `.dvc`-файл коммитим, а датасет — нет

В `.dvc`-файле записан md5 данных, весит он меньше 100 байт. Он лежит в Git,
поэтому у каждого коммита своя версия данных: `git checkout` + `dvc checkout`
возвращают и код, и данные. Сам CSV хранится в DVC-кэше и в MinIO.

Датасет в Git не кладём: Git хранит каждую версию файла целиком и навсегда,
репозиторий быстро пухнет, а у GitHub лимит 100 МБ на файл.

Если сделать наоборот, DVC перестанет работать. Без `.dvc`-файла в клоне
`dvc pull` не знает, что скачивать, откатить данные нельзя, и уже не понять,
на какой версии данных обучена модель.

## Пайплайн

Стадии описаны в `dvc.yaml`: `generate -> prepare -> validate -> train -> evaluate`.
Запуск — `make pipeline` (это `dvc repro`): DVC перезапускает только те стадии,
у которых поменялись код, данные или параметры.

### Что будет, если убрать `src/features.py` из `deps` стадии `train`

DVC перестанет замечать изменения в препроцессоре. Поменяли, например,
заполнение пропусков или добавили признак в `features.py` — `dvc repro`
скажет, что `train` не изменился, и оставит старую модель. Метрики и
`dvc.lock` тоже останутся старыми, и будет казаться, что изменение ни на
что не повлияло.

Это опаснее падения скрипта, потому что ошибку будет не заметной.

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
