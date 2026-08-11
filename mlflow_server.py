"""Convenience launcher for a local MLflow tracking server backed by the
same file store used by all training/eval scripts (configs/paths.yaml).

Usage:
    python mlflow_server.py --port 5000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    paths_cfg = yaml.safe_load(Path("configs/paths.yaml").read_text())
    tracking_uri = paths_cfg["mlflow"]["tracking_uri"]
    backend_store = tracking_uri.replace("file:", "")

    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", f"file:{backend_store}",
        "--host", args.host,
        "--port", str(args.port),
    ]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
