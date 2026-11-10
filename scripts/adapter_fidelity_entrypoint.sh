#!/usr/bin/env bash
# In-container entrypoint for the main analytic-vs-PhysX matrix (called by
# scripts/run_adapter_fidelity_docker.sh; all arguments are forwarded to
# app.run_adapter_fidelity_matrix).
#
# Reconstructed on 2026-11-10: the original file was lost with the September working
# copy. It mirrors adapter_fidelity_extras_entrypoint.sh; validate it on the first
# container run before relying on it.
set -uo pipefail

PROJECT_DIR="/workspace/LunarRocket"     # mount point inside the container
PY=/isaac-sim/python.sh
DEPS=/workspace/isaac_rl_deps            # persistent pip target (host cache mount)

# Python packages the matrix needs that Isaac Sim's interpreter may lack.
if [ ! -f "$DEPS/.installed" ]; then
  "$PY" -m pip install --quiet --target "$DEPS" PyYAML requests pillow pandas pyarrow \
    && touch "$DEPS/.installed"
fi
export PYTHONPATH="$PROJECT_DIR:$DEPS:${PYTHONPATH:-}"

cd "$PROJECT_DIR"
# run_adapter_fidelity_matrix launches each run with sys.executable, i.e. Isaac's python.
exec "$PY" -m app.run_adapter_fidelity_matrix "$@"
