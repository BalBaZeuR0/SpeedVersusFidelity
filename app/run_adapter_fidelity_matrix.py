"""Drive the full analytic-vs-PhysX experiment matrix in one process-group.

The default suite is:
  * two **open-loop** scenarios (s1_hover, s2_freefall_then_burn) - the pure
    integration-error floor, where a fixed action sequence is replayed through both
    adapters;
  * the **closed-loop** mission variants (c1..c5), where one shared DescentController
    runs in the loop on both adapters from an identical initial condition. This is
    the meaningful comparison for RL: the vehicle attitude is open-loop unstable in
    vacuum, so a fixed gimbal sequence just tumbles PhysX.

For each repeat (seed) it emits the canonical initial condition, then runs every
scenario through the analytic adapter and through the PhysX adapter. Each
(scenario, repeat, adapter) run is a **subprocess** so a PhysX crash cannot take the
whole matrix down and the two adapters get independent, un-contaminated timing.

Run this with the Isaac Sim python. See scripts/run_matrix_native.sh.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapter_fidelity_controller import CLOSED_LOOP_SCENARIOS
from app.adapter_fidelity_scenarios import SCENARIOS, S6_NAME

PY = sys.executable
MOD = "app.benchmark_adapter_fidelity"

OPEN_LOOP_BASELINE = ["s1_hover", "s2_freefall_then_burn"]
DEFAULT_SCENARIOS = OPEN_LOOP_BASELINE + sorted(CLOSED_LOOP_SCENARIOS)


def sh(cmd: list[str], timeout_s: float | None = None) -> int:
    """Run one child; kill its whole process group if it exceeds ``timeout_s``.

    Isaac Sim's Kit startup occasionally deadlocks (all threads in futex wait, no GPU
    context); without a timeout one such run stalls the matrix forever.
    """
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"  !! timeout after {timeout_s:.0f} s, killing pid {proc.pid}", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, AttributeError):
            proc.kill()
        proc.wait()
        return 124


def run_with_result(cmd: list[str], result: Path, timeout_s: float, retries: int) -> int:
    """Run ``cmd`` until it writes a fresh ``result`` file, at most ``1 + retries`` times.

    Isaac's SimulationApp can swallow a Python exception and still exit 0, so a run only
    counts as successful if the result file was written by this attempt.
    """
    rc = 1
    for attempt in range(1 + retries):
        if attempt:
            print(f"  .. retry {attempt}/{retries}", flush=True)
        t0 = time.time()
        rc = sh(cmd, timeout_s)
        if rc == 0 and not (result.exists() and result.stat().st_mtime >= t0 - 1):
            print(f"  !! no fresh result file {result}", flush=True)
            rc = 1
        if rc == 0:
            break
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=1234)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--effective-dt", type=float, default=1.0 / 60.0)
    p.add_argument("--out-dir", default="outputs/adapter_fidelity")
    p.add_argument("--scenarios", nargs="*", default=list(DEFAULT_SCENARIOS))
    p.add_argument("--adapters", nargs="*", default=["analytic", "physx"],
                   choices=["analytic", "physx"])
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--init-yaw-clamp-deg", type=float, default=30.0,
                   help="clamp drawn initial heading to +/- this in emit-init (negative disables)")
    p.add_argument("--run-timeout-s", type=float, default=300.0,
                   help="kill a single run after this many seconds (a normal PhysX run takes ~15 s)")
    p.add_argument("--retries", type=int, default=1, help="extra attempts for a failed or hung run")
    p.add_argument("--skip-existing", action="store_true",
                   help="resume: keep existing init files and skip runs whose result JSON exists")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures = 0

    for r in range(args.repeats):
        seed = args.base_seed + r
        init = out / f"init__r{r:02d}.json"
        if args.skip_existing and init.exists():
            rc = 0
        else:
            rc = sh([PY, "-m", MOD, "--emit-init", str(init), "--seed", str(seed),
                 "--effective-dt", str(args.effective_dt), "--config-dir", args.config_dir,
                     "--init-yaw-clamp-deg", str(args.init_yaw_clamp_deg)],
                    args.run_timeout_s)
        failures += rc != 0

        actions_json = out / f"actions__r{r:02d}.json"
        if S6_NAME in args.scenarios:
            rc = sh([PY, "-m", MOD, "--emit-actions", str(actions_json),
                     "--init-json", str(init), "--seed", str(seed),
                     "--horizon", str(args.horizon),
                     "--effective-dt", str(args.effective_dt),
                     "--config-dir", args.config_dir])
            failures += rc != 0

        common = [
            "--seed", str(seed), "--repeat", str(r),
            "--horizon", str(args.horizon), "--warmup", str(args.warmup),
            "--effective-dt", str(args.effective_dt),
            "--init-json", str(init), "--out-dir", str(args.out_dir),
            "--config-dir", args.config_dir,
        ]
        for adapter in args.adapters:
            for scenario in args.scenarios:
                cmd = [PY, "-m", MOD, "--adapter", adapter, "--scenario", scenario, *common]
                if scenario in CLOSED_LOOP_SCENARIOS:
                    cmd += ["--closed-loop"]
                elif scenario == S6_NAME:
                    cmd += ["--actions-json", str(actions_json)]
                result = out / f"{scenario}__r{r:02d}__{adapter}.json"
                if args.skip_existing and result.exists():
                    continue
                rc = run_with_result(cmd, result, args.run_timeout_s, args.retries)
                failures += rc != 0

    print(f"\n[matrix] done, {failures} failed run(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
