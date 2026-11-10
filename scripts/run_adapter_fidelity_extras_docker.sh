#!/usr/bin/env bash
# Run the three post-matrix "extras" tests inside the Isaac Sim container:
#   T1 six_dof=false ablation, T2 dt sweep, T3 PhysX determinism.
# See scripts/adapter_fidelity_extras_entrypoint.sh for the run list.
#
#   ./scripts/run_adapter_fidelity_extras_docker.sh
set -euo pipefail

export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
CACHE_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}"

docker rm -f lunar-rocket-fidelity-extras >/dev/null 2>&1 || true

mkdir -p \
  "$PROJECT_ROOT/outputs/adapter_fidelity/extras" \
  "$PROJECT_ROOT/assets/rocket" \
  "$PROJECT_ROOT/assets/moon_heightmaps" \
  "$CACHE_ROOT/cache/ov" "$CACHE_ROOT/cache/pip" "$CACHE_ROOT/cache/glcache" \
  "$CACHE_ROOT/cache/kit" "$CACHE_ROOT/data" "$CACHE_ROOT/documents" \
  "$CACHE_ROOT/rl_deps_py312"
chmod 0777 "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/outputs/adapter_fidelity" \
  "$PROJECT_ROOT/outputs/adapter_fidelity/extras" || true

docker run --rm \
  --name lunar-rocket-fidelity-extras \
  --entrypoint /bin/bash \
  --gpus all --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace/LunarRocket -e PYTHONUNBUFFERED=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PROJECT_ROOT:/workspace/LunarRocket" \
  -v "$CACHE_ROOT/cache/ov:/root/.cache/ov" \
  -v "$CACHE_ROOT/cache/pip:/root/.cache/pip" \
  -v "$CACHE_ROOT/cache/glcache:/root/.cache/nvidia" \
  -v "$CACHE_ROOT/cache/kit:/root/.cache/kit" \
  -v "$CACHE_ROOT/data:/root/.local/share/ov" \
  -v "$CACHE_ROOT/documents:/root/Documents" \
  -v "$CACHE_ROOT/rl_deps_py312:/workspace/isaac_rl_deps" \
  -w /workspace/LunarRocket \
  "$IMAGE" \
  -lc "umask 000 && /workspace/LunarRocket/scripts/adapter_fidelity_extras_entrypoint.sh"

echo
echo "[extras] container done. Analyse with:"
echo "  ./.venv-analytic/Scripts/python.exe analysis/analyze_adapter_fidelity_extras.py"
