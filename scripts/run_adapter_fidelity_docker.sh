#!/usr/bin/env bash
# Run the analytic-vs-PhysX speed/accuracy matrix inside the Isaac Sim container,
# so BOTH adapters are timed on the same Python / same GPU host.
#
#   ./scripts/run_adapter_fidelity_docker.sh
#
# Env overrides:
#   FID_REPEATS=5  FID_HORIZON=1000  FID_WARMUP=50  FID_EFFECTIVE_DT=0.016666667
#   FID_SCENARIOS="c1_nominal c2_divert"   # default: s1_hover s2_freefall_then_burn c1..c5
#   FID_ADAPTERS="analytic physx"          # drop physx to dry-run without Isaac
#   FID_OUT_DIR=outputs/adapter_fidelity  FID_INIT_YAW_CLAMP=30
set -euo pipefail

# Git Bash / MSYS on Windows rewrites /workspace/... into C:/Program Files/Git/...
# before it reaches the Docker CLI. Disable that for the container-side paths.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
CACHE_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}"

FID_REPEATS="${FID_REPEATS:-5}"
FID_HORIZON="${FID_HORIZON:-1000}"
FID_WARMUP="${FID_WARMUP:-50}"
FID_EFFECTIVE_DT="${FID_EFFECTIVE_DT:-0.016666666666666666}"
FID_BASE_SEED="${FID_BASE_SEED:-1234}"
FID_SCENARIOS="${FID_SCENARIOS:-}"
FID_ADAPTERS="${FID_ADAPTERS:-analytic physx}"
FID_OUT_DIR="${FID_OUT_DIR:-outputs/adapter_fidelity}"
FID_INIT_YAW_CLAMP="${FID_INIT_YAW_CLAMP:-30}"

ARGS=(--repeats "$FID_REPEATS" --base-seed "$FID_BASE_SEED"
      --horizon "$FID_HORIZON" --warmup "$FID_WARMUP"
      --effective-dt "$FID_EFFECTIVE_DT" --adapters $FID_ADAPTERS
      --out-dir "$FID_OUT_DIR" --init-yaw-clamp-deg "$FID_INIT_YAW_CLAMP")
[ -n "$FID_SCENARIOS" ] && ARGS+=(--scenarios $FID_SCENARIOS)

docker rm -f lunar-rocket-fidelity >/dev/null 2>&1 || true

mkdir -p \
  "$PROJECT_ROOT/$FID_OUT_DIR" \
  "$PROJECT_ROOT/outputs/adapter_fidelity" \
  "$PROJECT_ROOT/assets/rocket" \
  "$PROJECT_ROOT/assets/moon_heightmaps" \
  "$CACHE_ROOT/cache/ov" "$CACHE_ROOT/cache/pip" "$CACHE_ROOT/cache/glcache" \
  "$CACHE_ROOT/cache/kit" "$CACHE_ROOT/data" "$CACHE_ROOT/documents" \
  "$CACHE_ROOT/rl_deps_py312"
chmod 0777 "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/outputs/adapter_fidelity" || true

docker run --rm \
  --name lunar-rocket-fidelity \
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
  -lc "umask 000 && /workspace/LunarRocket/scripts/adapter_fidelity_entrypoint.sh ${ARGS[*]}"

echo
echo "[fidelity] matrix done. Now run the analysis (host venv or in-container):"
echo "  python analysis/analyze_adapter_fidelity.py"
