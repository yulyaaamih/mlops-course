# Будет расти по ходу курса. Правило: команду, которую приходится
# вспоминать по истории терминала, оформляем целью здесь.
.PHONY: help install check data prepare train test lint

PY := python

help:            ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:         ## Установить зависимости
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

check:           ## Проверить окружение
	$(PY) -m src.smoke_check

data:            ## Сгенерировать сырой датасет
	$(PY) -m src.data.generate

prepare:         ## Подготовить train/val/test
	$(PY) -m src.data.prepare

train:           ## Обучить модель
	$(PY) -m src.train

test:            ## Прогнать тесты
	pytest

lint:            ## Проверить стиль
	ruff check src tests
