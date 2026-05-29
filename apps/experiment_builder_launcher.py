"""
Launch the Delphi Experiment Builder.

Usage:
    python apps/experiment_builder_launcher.py [--server.port PORT] [extra streamlit args...]

Prints the SSH tunnel command, then replaces this process with:
    streamlit run apps/experiment_builder.py
"""

import os
import sys
import socket
import getpass
from pathlib import Path

_APP = Path(__file__).resolve().parent / "experiment_builder.py"
_PORT_DEFAULT = 8501


def main():
    port = _PORT_DEFAULT
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--server.port" and i + 1 < len(args):
            port = int(args[i + 1])

    print(
        f"\n  ssh -J mitigate -L {port}:localhost:{port} {getpass.getuser()}@{socket.gethostname()}\n",
        flush=True,
    )

    os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(_APP)] + args)


if __name__ == "__main__":
    main()
