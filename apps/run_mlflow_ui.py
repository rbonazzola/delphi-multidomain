"""
Launch the MLflow UI and print the SSH tunnel command.

Usage:
    python apps/run_mlflow_ui.py [--port PORT] [extra mlflow ui args...]

The backend store URI is read from $MLFLOW_TRACKING_URI.
"""

import os
import sys
import socket
import getpass
import shutil

PORT_DEFAULT = 5000


def main():
    # Parse --port from args (pass everything else through to mlflow)
    args = sys.argv[1:]
    port = PORT_DEFAULT
    for i, a in enumerate(args):
        if a == "--port" and i + 1 < len(args):
            port = int(args[i + 1])

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        print("Warning: $MLFLOW_TRACKING_URI is not set.", flush=True)

    print(
        f"\n  ssh -J mitigate -L {port}:localhost:{port} {getpass.getuser()}@{socket.gethostname()}\n",
        flush=True,
    )

    cmd = ["mlflow", "ui"]
    if tracking_uri:
        cmd += ["--backend-store-uri", tracking_uri]
    if "--port" not in args:
        cmd += ["--port", str(port)]
    cmd += args

    mlflow_bin = shutil.which("mlflow")
    if mlflow_bin:
        os.execv(mlflow_bin, cmd)
    else:
        os.execv(sys.executable, [sys.executable, "-m"] + cmd)


if __name__ == "__main__":
    main()
