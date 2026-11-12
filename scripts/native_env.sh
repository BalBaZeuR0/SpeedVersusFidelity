# Source this file (do not execute it): environment for running the study on a native
# Isaac Sim 6.0.1 + Isaac Lab install (pip / venv), without Docker.
#
#   source scripts/native_env.sh
#
# Overrides:
#   ISAAC_PY   python of the Isaac Lab venv (must import `isaacsim`)
#   SVF_DEPS   extra --target dir for packages the venv lacks (pyarrow, pytest)
#
# The extra packages are installed with --no-deps so they never shadow the numpy that
# ships with Isaac Sim.

SVF_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SVF_REPO
export ISAAC_PY="${ISAAC_PY:-/media/noron/DISK02/isaac-lab-native/env_isaaclab/bin/python}"
export SVF_DEPS="${SVF_DEPS:-$SVF_REPO/.deps}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SVF_REPO:$SVF_DEPS"

if [ ! -x "$ISAAC_PY" ]; then
  echo "[native_env] ISAAC_PY not found: $ISAAC_PY" >&2
  return 1 2>/dev/null || exit 1
fi

if [ ! -f "$SVF_DEPS/.installed" ]; then
  echo "[native_env] installing pyarrow + pytest into $SVF_DEPS"
  mkdir -p "$SVF_DEPS"
  PYTHONPATH= "$ISAAC_PY" -m pip install --quiet --target "$SVF_DEPS" --no-deps \
      pyarrow pytest iniconfig pluggy packaging pygments \
    && touch "$SVF_DEPS/.installed"
fi
