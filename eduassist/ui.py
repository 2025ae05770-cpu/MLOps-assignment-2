"""Launch the local MLflow server against the evaluation's SQLite database."""

import argparse
import subprocess
import sys

from eduassist.pipeline import tracking_uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the assignment MLflow UI.")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    uri = tracking_uri()
    print(f"MLflow UI: http://127.0.0.1:{args.port}", flush=True)
    subprocess.run([
        sys.executable, "-m", "mlflow", "server", "--backend-store-uri", uri,
        "--host", "127.0.0.1", "--port", str(args.port),
    ], check=True)


if __name__ == "__main__":
    main()