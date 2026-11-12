#!/usr/bin/env bash
# Run the analytic-vs-PhysX speed/accuracy matrix on a native Isaac Sim install, so both
# adapters are timed with the same Python on the same GPU host.
#
#   ./scripts/run_matrix_native.sh
#
# Env overrides:
#   FID_REPEATS=5  FID_HORIZON=1000  FID_WARMUP=50  FID_EFFECTIVE_DT=0.016666666666666666
#   FID_SCENARIOS="c1_nominal c2_divert"   # default: s1_hover s2_freefall_then_burn c1..c5
#   FID_ADAPTERS="analytic physx"          # drop physx to dry-run without Isaac
#   FID_OUT_DIR=outputs/adapter_fidelity  FID_INIT_YAW_CLAMP=30
set -euo pipefail

# shellcheck source=native_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/native_env.sh"
cd "$SVF_REPO"

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

mkdir -p "$FID_OUT_DIR"
# run_adapter_fidelity_matrix starts every run with sys.executable, i.e. $ISAAC_PY.
exec "$ISAAC_PY" -m app.run_adapter_fidelity_matrix "${ARGS[@]}" "$@"
