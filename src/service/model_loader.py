import json

import joblib

from src.config import feature_columns, load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()


class ModelHolder:
    def __init__(self) -> None:
        self.model = None
        self.version = "not-loaded"
        self.input_columns: set[str] = set()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        params = load_params()
        if not (params["mlflow"]["enabled"] and self._try_registry(params)):
            self._load_local(params)
        self._check_signature(params)

    def _check_signature(self, params) -> None:
        expected = set(feature_columns(params))
        actual = self.input_columns
        if actual != expected:
            raise RuntimeError(
                f"сигнатура модели {self.version} разошлась с params.yaml: "
                f"лишние {sorted(actual - expected)}, недостающие {sorted(expected - actual)}"
            )

    def _try_registry(self, params) -> bool:
        try:
            import mlflow
            from mlflow import MlflowClient

            mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
            name = params["mlflow"]["registered_model_name"]
            alias = params["mlflow"]["best_model_alias"]
            mv = MlflowClient().get_model_version_by_alias(name, alias)
            uri = f"models:/{name}@{alias}"
            self.model = mlflow.sklearn.load_model(uri)
            self.input_columns = set(mlflow.models.get_model_info(uri).signature.inputs.input_names())
            self.version = f"registry:{name}@{alias}/v{mv.version}"
            return True
        except Exception as exc:
            log.warning("MLflow Registry недоступен (%s), берём локальный файл", exc)
            return False

    def _load_local(self, params) -> None:
        self.model = joblib.load(resolve(params["service"]["model_path"]))
        self.input_columns = set(self.model.feature_names_in_)
        meta_path = resolve(params["service"]["meta_path"])
        if not meta_path.exists():
            self.version = "local:unknown"
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.version = f"local:{meta['model']}@{meta['git_sha']}"


holder = ModelHolder()
