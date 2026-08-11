"""Thin wrapper around MLflow so every entry-point logs consistently."""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator

import mlflow
from omegaconf import DictConfig, OmegaConf


def init_mlflow(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.paths.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.paths.mlflow.experiment_name)


@contextlib.contextmanager
def mlflow_run(cfg: DictConfig, run_name: str | None = None) -> Iterator[None]:
    """Context manager that starts a run, logs the full resolved config as
    params + a YAML artifact, and always ends the run (even on exception)."""
    init_mlflow(cfg)
    with mlflow.start_run(run_name=run_name or cfg.get("run_name")):
        flat_params = _flatten(OmegaConf.to_container(cfg, resolve=True))
        # MLflow rejects param values longer than 500 chars / batches > 100
        for i in range(0, len(flat_params), 100):
            batch = dict(list(flat_params.items())[i : i + 100])
            mlflow.log_params({k: str(v)[:500] for k, v in batch.items()})
        yield


def log_config_artifact(cfg: DictConfig, out_dir: str = "/tmp") -> None:
    path = Path(out_dir) / "resolved_config.yaml"
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    mlflow.log_artifact(str(path))


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
