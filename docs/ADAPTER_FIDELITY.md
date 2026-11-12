# Analytic vs PhysX adapter fidelity study

Quantifying the speed / accuracy tradeoff between the two world adapters that both
implement `LunarWorldAdapter` (`app/env_lunar_landing.py`), for RL training use.

| | analytic | physx |
|---|---|---|
| class | `AnalyticLunarLandingAdapter` (`app/analytic_world_adapter.py`) | `IsaacWorldAdapter` (`app/world_adapter.py`) |
| translation | semi-implicit Euler, point mass | PhysX rigid body via `LunarLandingWorld` |
| attitude | **kinematic** - body instantly aligned to the thrust vector | true 6-DOF rigid-body rotation |
| thrust model | body-frame gimballed force | body-frame gimballed force **+ gimbal moment `τ = lever × F`** (`thrust.lever_arm_m`, `thrust.six_dof`) - ported from `source/lunar_rocket_lab` |
| Isaac needed | no | yes |

PhysX is the reference; the analytic run is scored against it.

## Two protocols

* **Open-loop baseline (S1, S2)** - a fixed action sequence is replayed through both
  adapters. Measures the raw integration-error floor. Only near-zero-gimbal scenarios
  are usable: with the 6-DOF thrust model the vehicle attitude is an undamped double
  integrator in vacuum, so any sustained gimbal command tumbles PhysX open-loop.
* **Closed-loop (c1..c5)** - one shared `DescentController` (guidance + TVC attitude
  PD, `app/adapter_fidelity_controller.py`) runs in the loop on both adapters from an
  identical initial condition. Measures how far the two closed-loop trajectories
  separate - i.e. how wrong a controller / policy developed on the fast analytic
  adapter is on the real physics.

## What runs what

| file | role |
|---|---|
| `app/adapter_fidelity_controller.py` | shared closed-loop `DescentController` + mission variants `CLOSED_LOOP_SCENARIOS` (c1..c5) |
| `app/adapter_fidelity_scenarios.py` | open-loop action-sequence generators (S1/S2 baseline; S3-S6 legacy) |
| `app/benchmark_adapter_fidelity.py` | one (scenario, repeat, adapter) run → npz + JSON meta. Drives adapters **directly**. `--emit-init` writes the shared initial condition; `--closed-loop` runs the controller in the loop, otherwise a fixed action sequence is replayed. |
| `app/run_adapter_fidelity_matrix.py` | runs the matrix as isolated subprocesses. Default suite: `s1_hover s2_freefall_then_burn c1_nominal c2_divert c3_fast_descent c4_tight_guidance c5_high_tilt` |
| `analysis/analyze_adapter_fidelity.py` | pairs analytic/physx, computes speed + pos/vel/tilt/omega divergence + landing agreement → `metrics.csv`, `summary.csv`, figures |
| `scripts/run_matrix_native.sh` | runs the matrix with a native Isaac Sim 6.0.1 install (`ISAAC_PY`, see `scripts/native_env.sh`) so both adapters are timed with the same Python on the same host. `FID_OUT_DIR`, `FID_SCENARIOS`, `FID_REPEATS`, ... env overrides |
| `scripts/run_extras_native.sh` | extras: 6-DOF ablation, dt sweep, PhysX determinism |
| `tests/test_adapter_fidelity.py` | static checks, no Isaac (25 tests) |

## Quick start

```bash
# analytic-only dry run (needs numpy + pandas; see requirements.txt)
python -m app.run_adapter_fidelity_matrix --adapters analytic --repeats 2 --horizon 1000

# full matrix incl. PhysX, native Isaac Sim
FID_REPEATS=5 FID_HORIZON=1000 FID_WARMUP=100 ./scripts/run_matrix_native.sh

# single-scenario check into an isolated dir
FID_OUT_DIR=outputs/smoke FID_REPEATS=1 FID_SCENARIOS="c1_nominal" ./scripts/run_matrix_native.sh

# metrics + figures
python analysis/analyze_adapter_fidelity.py
```

## Sanity checks (first PhysX run)

- [ ] `init_sync_err` in the JSON meta is ~0 (both adapters start from the same state)
- [ ] one `world.step()` advances exactly `--effective-dt`
- [ ] same seed → same trajectory (determinism)
- [ ] config mass / max thrust / lever arm identical for both adapters
- [ ] closed-loop: PhysX stays bounded (tilt well under 90°, soft touchdown)

Full plan: `D:/obsidianVault/IsaacSim_Adapter/01_Plan.md` and `DURUM.md`.
