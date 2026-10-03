"""Первый тест проекта: проверяет, что окружение и конфиг живы.
Настоящие тесты появятся на занятии 8."""
from __future__ import annotations

from src.config import load_params
from src.data.generate import generate


def test_params_load():
    params = load_params()
    assert params["seed"] == 42


def test_generate_returns_data():
    df = generate(n=100, seed=1)
    assert len(df) == 100
    assert "churn" in df.columns
