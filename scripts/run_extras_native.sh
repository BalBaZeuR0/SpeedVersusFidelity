#!/usr/bin/env bash
# Three "extras" tests run after the main matrix (all closed-loop unless noted), on a
# native Isaac Sim install:
#
#   T1  six_dof=false ablation      c1..c5, analytic + physx, --no-six-dof, 3 repeats
#   T2  integration-step (dt) sweep s2 (open loop) + c1, dt in {1/120, 1/30}
#   T3  PhysX determinism           c1_nominal physx, same seed, run twice
#
#   ./scripts/run_extras_native.sh
#
# Each single run is its own subprocess (crash isolation, clean timing), exactly
# like run_adapter_fidelity_matrix.py. A run that hangs (Kit startup can deadlock) is
# killed after RUN_TIMEOUT_S seconds (default 300).
set -uo pipefail

# shellcheck source=native_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/native_env.sh" || exit 1
cd "$SVF_REPO"

PY="$ISAAC_PY"
MOD="app.benchmark_adapter_fidelity"
DT60=0.016666666666666666
DT120=0.008333333333333333
DT30=0.03333333333333333
EX="outputs/adapter_fidelity/extras"
mkdir -p "$EX"

fail=0
run() { echo "+ $PY -m $MOD $*"; timeout -k 10 "${RUN_TIMEOUT_S:-300}" "$PY" -m "$MOD" "$@" || { echo "  !! rc=$?"; fail=$((fail+1)); }; }

# ------------------------------------------------------------------ T3 (cheapest first)
echo "=== T3  PhysX determinism ==="
run --emit-init "$EX/determinism/init.json" --seed 1234 --effective-dt "$DT60" \
    --init-yaw-clamp-deg 30
for tag in a b; do
  run --adapter physx --scenario c1_nominal --closed-loop \
      --seed 1234 --repeat 0 --horizon 1000 --warmup 50 --effective-dt "$DT60" \
      --init-json "$EX/determinism/init.json" --out-dir "$EX/determinism_$tag"
done

# ------------------------------------------------------------------ T1  six_dof=false ablation
echo "=== T1  six_dof=false ablation (c1..c5, 3 repeats) ==="
for r in 0 1 2; do
  seed=$((1234 + r))
  init="$EX/ablation/init__r0${r}.json"
  run --emit-init "$init" --seed "$seed" --effective-dt "$DT60" --init-yaw-clamp-deg 30
  for adapter in analytic physx; do
    for sc in c1_nominal c2_divert c3_fast_descent c4_tight_guidance c5_high_tilt; do
      run --adapter "$adapter" --scenario "$sc" --closed-loop --no-six-dof \
          --seed "$seed" --repeat "$r" --horizon 1000 --warmup 50 --effective-dt "$DT60" \
          --init-json "$init" --out-dir "$EX/ablation_nosixdof"
    done
  done
done

# ------------------------------------------------------------------ T2  dt sweep
echo "=== T2  integration-step (dt) sweep ==="
# dt=1/120 -> horizon 2000 ; dt=1/30 -> horizon 500 ; keeps sim time ~16.7 s
run --emit-init "$EX/dt/init.json" --seed 1234 --effective-dt "$DT60" --init-yaw-clamp-deg 30
for spec in "120:$DT120:2000" "030:$DT30:500"; do
  tag="${spec%%:*}"; rest="${spec#*:}"; dt="${rest%%:*}"; hz="${rest##*:}"
  for adapter in analytic physx; do
    # s2 is open-loop (scripted axial burn); c1 is closed-loop
    run --adapter "$adapter" --scenario s2_freefall_then_burn \
        --seed 1234 --repeat 0 --horizon "$hz" --warmup 50 --effective-dt "$dt" \
        --init-json "$EX/dt/init.json" --out-dir "$EX/dt_$tag"
    run --adapter "$adapter" --scenario c1_nominal --closed-loop \
        --seed 1234 --repeat 0 --horizon "$hz" --warmup 50 --effective-dt "$dt" \
        --init-json "$EX/dt/init.json" --out-dir "$EX/dt_$tag"
  done
done

echo
echo "[extras] done, $fail failed run(s). Outputs under $SVF_REPO/$EX/"
exit 0
