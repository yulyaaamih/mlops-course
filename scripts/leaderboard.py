"""Таблица лидеров из MLflow.

Забирает топ прогонов по ROC-AUC через API и пишет reports/LEADERBOARD.md.
Прогон, чья версия модели в реестре помечена алиасом best_model, отмечается в таблице.

Запуск: make leaderboard
"""

from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()

REPORT_PATH = "reports/LEADERBOARD.md"
METRICS = ["metrics.roc_auc", "metrics.pr_auc", "metrics.f1"]
METRIC_TOLERANCE = 1e-9


def get_best_model_run_id(client: MlflowClient, model_name: str, alias: str) -> str | None:
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None
    return version.run_id


def fetch_runs(experiment_name: str) -> pd.DataFrame:
    return mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.roc_auc DESC"],
    )


def or_dash(value) -> str:
    return "—" if value is None or pd.isna(value) else str(value)


def hyperparams_of(run: pd.Series) -> str:
    prefix = f"params.{run.get('params.model')}."
    return ", ".join(
        f"{col.removeprefix(prefix)}={value}"
        for col, value in sorted(run.items())
        if col.startswith(prefix) and pd.notna(value)
    )


def run_keys(runs: pd.DataFrame) -> pd.Series:
    return runs["params.model"].astype(str) + " (" + runs.apply(hyperparams_of, axis=1) + ")"


def find_unstable_configs(runs: pd.DataFrame) -> list[str]:
    problems = []
    for key, group in runs.groupby(run_keys(runs)):
        spread = (group[METRICS].max() - group[METRICS].min()).max()
        if len(group) > 1 and spread > METRIC_TOLERANCE:
            names = ", ".join(group["tags.mlflow.runName"].astype(str))
            problems.append(f"`{key}`: метрики различаются у прогонов {names}")
    return problems


def drop_duplicate_runs(runs: pd.DataFrame, best_run_id: str | None) -> pd.DataFrame:
    is_best = runs["run_id"] == best_run_id
    return (
        runs.assign(_key=run_keys(runs), _best=is_best)
        .sort_values("_best", ascending=False, kind="stable")
        .drop_duplicates("_key")
        .sort_index()
        .drop(columns=["_key", "_best"])
    )


def format_row(run: pd.Series, best_run_id: str | None) -> str:
    model = or_dash(run.get("params.model"))
    hyperparams = hyperparams_of(run)
    roc_auc = run.get("metrics.roc_auc")
    pr_auc = run.get("metrics.pr_auc")
    cells = [
        "best_model" if run["run_id"] == best_run_id else "",
        or_dash(run.get("tags.mlflow.runName")),
        f"`{model}`",
        hyperparams or "—",
        "—" if pd.isna(roc_auc) else f"{roc_auc:.4f}",
        "—" if pd.isna(pr_auc) else f"{pr_auc:.4f}",
        or_dash(run.get("tags.git_sha")),
        run["start_time"].strftime("%Y-%m-%d %H:%M"),
    ]
    return "| " + " | ".join(cells) + " |"


def build_report(
    runs: pd.DataFrame,
    best_run_id: str | None,
    alias: str,
    experiment_name: str,
    problems: list[str],
) -> str:
    lines = [
        f"# Таблица лидеров: топ-{len(runs)} по ROC-AUC",
        "",
        f"Эксперимент MLflow: `{experiment_name}`",
        "",
        "| Выбрана | Прогон | Модель | Параметры | ROC-AUC | PR-AUC | git_sha | Дата |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [format_row(run, best_run_id) for _, run in runs.iterrows()]
    if best_run_id is None:
        lines += ["", f"Алиас `{alias}` в реестре не поставлен."]
    elif best_run_id not in set(runs["run_id"]):
        lines += ["", f"Модель с алиасом `{alias}` (run `{best_run_id}`) не входит в топ."]
    if problems:
        lines += ["", "## Невоспроизводимые конфигурации", ""]
        lines += [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = load_params()["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])

    best_run_id = get_best_model_run_id(MlflowClient(), cfg["registered_model_name"], cfg["best_model_alias"])
    runs = fetch_runs(cfg["experiment_name"])
    problems = find_unstable_configs(runs)
    top = drop_duplicate_runs(runs, best_run_id).head(cfg["leaderboard_size"])

    path = resolve(REPORT_PATH)
    report = build_report(top, best_run_id, cfg["best_model_alias"], cfg["experiment_name"], problems)
    path.write_text(report, encoding="utf-8")
    log.info("таблица лидеров сохранена в %s", path)


if __name__ == "__main__":
    main()
