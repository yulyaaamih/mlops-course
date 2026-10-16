from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score

from src.config import load_params, resolve
from src.logging_setup import setup_logging
from src.train import load_data, split_xy

log = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Перевести кандидата в champion, если он лучше на test")
    parser.add_argument("--candidate", type=int, help="номер версии кандидата, по умолчанию последняя")
    return parser.parse_args()


def latest_version(client: MlflowClient, name: str) -> int:
    return max(int(mv.version) for mv in client.search_model_versions(f"name='{name}'"))


def champion_version(client: MlflowClient, name: str, alias: str) -> int | None:
    try:
        return int(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def test_roc_auc(name: str, version: int, X, y) -> float:
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def main() -> None:
    args = parse_args()
    params = load_params()
    cfg = params["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    client = MlflowClient()
    name, alias = cfg["registered_model_name"], cfg["best_model_alias"]

    candidate = args.candidate or latest_version(client, name)
    champion = champion_version(client, name, alias)
    if candidate == champion:
        log.info("v%s уже %s, сравнивать не с чем", candidate, alias)
        return

    X, y = split_xy(load_data(resolve(params["data"]["processed_dir"]) / "test.csv"), params)
    candidate_auc = test_roc_auc(name, candidate, X, y)

    if champion is None:
        log.info("алиаса %s нет, ставим на v%s (ROC-AUC %.4f)", alias, candidate, candidate_auc)
        client.set_registered_model_alias(name, alias, candidate)
        return

    champion_auc = test_roc_auc(name, champion, X, y)
    gain = candidate_auc - champion_auc
    log.info(
        "кандидат v%s: %.4f, %s v%s: %.4f, прирост %+.4f",
        candidate,
        candidate_auc,
        alias,
        champion,
        champion_auc,
        gain,
    )

    min_gain = cfg["promote_min_gain"]
    if gain < min_gain:
        log.info("прирост меньше %s, %s остаётся на v%s", min_gain, alias, champion)
        return

    client.set_registered_model_alias(name, alias, candidate)
    log.info("%s перевешен с v%s на v%s", alias, champion, candidate)


if __name__ == "__main__":
    main()
