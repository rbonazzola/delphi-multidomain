#!/usr/bin/env bash
# Launch all Delphi apps in the background and print SSH tunnel commands.
# Run from the project root:
#   bash launch_apps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="$(hostname)"
USER="$(whoami)"

PORT_MLFLOW=5000
PORT_TRAJ=8501
PORT_EXPBUILDER=8502
PORT_LOSS=8503

echo ""
echo "Apps:"
echo "  MLflow UI:           http://localhost:${PORT_MLFLOW}"
echo "  Trajectory Explorer: http://localhost:${PORT_TRAJ}"
echo "  Experiment Builder:  http://localhost:${PORT_EXPBUILDER}"
echo "  Loss Viewer:         http://localhost:${PORT_LOSS}"
echo ""
echo "SSH tunnel (run on your local machine):"
echo ""
echo "  ssh -J mitigate \\"
echo "    -L ${PORT_MLFLOW}:localhost:${PORT_MLFLOW} \\"
echo "    -L ${PORT_TRAJ}:localhost:${PORT_TRAJ} \\"
echo "    -L ${PORT_EXPBUILDER}:localhost:${PORT_EXPBUILDER} \\"
echo "    -L ${PORT_LOSS}:localhost:${PORT_LOSS} \\"
echo "    ${USER}@${HOST}"
echo ""

python "${SCRIPT_DIR}/run_mlflow_ui.py" --port "${PORT_MLFLOW}" 2>&1 | grep -v "^$\|ssh -J" &
python -m streamlit run "${SCRIPT_DIR}/traj_explorer.py"         --server.port "${PORT_TRAJ}"        --server.headless true &
python -m streamlit run "${SCRIPT_DIR}/experiment_builder.py"    --server.port "${PORT_EXPBUILDER}"  --server.headless true &
python -m streamlit run "${SCRIPT_DIR}/delphi_loss_viewer/app.py" --server.port "${PORT_LOSS}"       --server.headless true &

wait
