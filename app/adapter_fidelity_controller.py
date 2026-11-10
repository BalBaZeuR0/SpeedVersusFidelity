"""Shared closed-loop descent controller for the analytic-vs-PhysX fidelity study.

The study's original open-loop protocol (replay a fixed action sequence through both
adapters) breaks for this vehicle: with the repo's 6-DOF thrust model the attitude is
an undamped double integrator in vacuum, so any sustained gimbal command diverges
without bound and the comparison degenerates.

Instead we run **one identical stabilising guidance + attitude controller** in the loop
on both adapters, from an identical initial condition, and measure how far the two
closed-loop trajectories separate. That divergence is the quantity that actually
matters for RL: "if I train / evaluate a controller on the fast analytic adapter, how
wrong is it on the real physics?"

The controller is deliberately simple and adapter-agnostic - it reads the standard
``RocketKinematicState`` and returns ``[throttle, gimbal_x, gimbal_y]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.rl_types import RocketKinematicState


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    w, xyz = float(q[0]), q[1:4]
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _quat_rotate_inv_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return _quat_rotate_wxyz(np.array([q[0], -q[1], -q[2], -q[3]]), v)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _quat_from_z_axis(z_des: np.ndarray) -> np.ndarray:
    """Minimal-rotation wxyz quaternion taking world +Z onto the unit vector z_des."""
    z = np.asarray(z_des, dtype=np.float64)
    z = z / max(float(np.linalg.norm(z)), 1e-9)
    zc = np.array([0.0, 0.0, 1.0])
    d = float(np.clip(np.dot(zc, z), -1.0, 1.0))
    if d > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(zc, z)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    half = np.arccos(d) * 0.5
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


@dataclass
class ControllerConfig:
    # guidance (slow outer loop) -> a commanded thrust direction
    kp_pos: float = 0.12
    kd_pos: float = 0.60
    tilt_cmd_max_deg: float = 10.0   # cap on commanded thrust tilt from vertical
    k_alt: float = 0.22             # descent-rate schedule slope (1/s)
    vz_min: float = 0.35
    vz_max: float = 2.4
    kp_vz: float = 1.5
    az_thrust_floor_frac: float = 0.45   # keep >= this fraction of hover thrust
    # attitude: PD on the body-quaternion error toward the guidance-demanded attitude,
    # mapped to a gimbal deflection through the wrench model. Gains are the
    # critically-damped design for the PhysX double integrator (θ̈ ∝ gimbal); on the
    # algebraic analytic model the same law makes the thrust direction track the
    # demand. The rate term is fed a low-passed rate because the analytic model's
    # finite-difference body rate is impulsive at every gimbal step.
    kp_att: float = 0.30         # N.m per rad of attitude error
    kd_att: float = 0.24         # N.m per rad/s of (filtered) body rate  (over-damped for robustness)
    w_filter: float = 0.8        # low-pass on the observed body rate
    gimbal_lp: float = 0.3       # output low-pass on the gimbal command
    gimbal_rate: float = 0.09    # max change in gimbal command per step
    gimbal_max: float = 0.7      # fraction of max gimbal deflection
    # mission variant
    target_offset_xy: tuple[float, float] = (0.0, 0.0)
    vz_scale: float = 1.0           # >1 => more aggressive descent


class DescentController:
    """Cascaded controller: one shared **guidance** law (sink-rate schedule + a PD on
    the horizontal position error -> a desired thrust direction) and an adapter-matched
    **inner attitude law** that turns that direction into a gimbal command.

    The inner law must differ by adapter because the two models interpret a gimbal
    command differently: for the analytic model the gimbal *is* the thrust direction
    (and hence the attitude), so a feed-forward pointing law realises the demand
    directly; for the 6-DOF PhysX model the gimbal is a control *moment*, so a
    critically-damped PD on the attitude error is required to drive the (double
    integrator) rotation. Using each model's appropriate inner law is what a
    practitioner does; the guidance intent, initial condition, and scenario are
    identical, so the trajectory divergence still isolates the dynamics fidelity.

        inner_law = "pd"  (PhysX / any 6-DOF model)  |  "ff"  (analytic point mass)
    """

    def __init__(self, configs: dict[str, dict[str, Any]],
                 cfg: ControllerConfig | None = None, inner_law: str = "pd"):
        self.cfg = cfg or ControllerConfig()
        if inner_law not in ("pd", "ff"):
            raise ValueError(f"inner_law must be 'pd' or 'ff', got {inner_law!r}")
        self.inner_law = inner_law
        rc = configs["rocket"]
        self.mass = float(rc.get("mass_kg", 1.63))
        self.g = float(configs["sim"].get("gravity_mps2", 1.62))
        self.f_max = float(rc.get("thrust", {}).get("max_newtons", 30.0))
        self.max_gimbal_rad = np.deg2rad(float(rc.get("thrust", {}).get("max_gimbal_deg", 12.0)))
        self.lever = float(rc.get("thrust", {}).get("lever_arm_m", 0.85))
        self.reset()

    def reset(self) -> None:
        self._w_filt = np.zeros(3)
        self._gimbal_prev = np.zeros(2)

    def __call__(self, state: RocketKinematicState) -> np.ndarray:
        c = self.cfg
        p = np.asarray(state.position, dtype=np.float64)
        v = np.asarray(state.linear_velocity, dtype=np.float64)
        q = np.asarray(state.quaternion_wxyz, dtype=np.float64)
        w = np.asarray(state.angular_velocity, dtype=np.float64)
        tgt = np.asarray(state.target_position, dtype=np.float64).copy()
        tgt[0] += c.target_offset_xy[0]
        tgt[1] += c.target_offset_xy[1]
        alt = float(state.altitude_to_ground)

        # --- guidance: desired world-frame thrust acceleration ---
        # vertical: track an altitude-scheduled sink rate; thrust can only push up,
        # so floor the vertical thrust accel (a downward demand just means low thrust).
        vz_ref = -float(np.clip(c.vz_scale * (c.k_alt * alt) + c.vz_min, c.vz_min, c.vz_max))
        az_cmd = c.kp_vz * (vz_ref - v[2])
        az_thrust = max(az_cmd + self.g, c.az_thrust_floor_frac * self.g)
        # horizontal: PD toward the target, but capped so the commanded body tilt
        # (atan(|a_xy| / az_thrust)) never exceeds tilt_cmd_max.
        a_xy = c.kp_pos * (tgt[:2] - p[:2]) - c.kd_pos * v[:2]
        a_xy_max = az_thrust * np.tan(np.deg2rad(c.tilt_cmd_max_deg))
        norm_xy = float(np.linalg.norm(a_xy))
        if norm_xy > a_xy_max:
            a_xy = a_xy * (a_xy_max / norm_xy)
        thrust_acc = np.array([a_xy[0], a_xy[1], az_thrust])
        mag = float(np.linalg.norm(thrust_acc))
        throttle = float(np.clip(mag * self.mass / max(self.f_max, 1e-6), 0.0, 1.0))
        up_des = thrust_acc / max(mag, 1e-6)

        # --- inner attitude law: desired thrust direction -> gimbal command ---
        sd = np.sin(self.max_gimbal_rad) * 0.999
        if self.inner_law == "ff":
            # analytic point mass: the gimbal *is* the thrust direction. Point the
            # body-frame thrust straight along the guidance demand.
            up_des_body = _quat_rotate_inv_wxyz(q, up_des)
            g = np.array([
                np.arcsin(np.clip(up_des_body[0], -sd, sd)) / self.max_gimbal_rad,
                np.arcsin(np.clip(up_des_body[1], -sd, sd)) / self.max_gimbal_rad,
            ])
        else:
            # 6-DOF: gimbal is a moment. Critically-damped PD on the attitude error
            # toward the demanded attitude; rate term fed a low-passed body rate.
            q_err = _quat_mul(_quat_from_z_axis(up_des), _quat_conj(q))
            if q_err[0] < 0.0:
                q_err = -q_err
            err_body = _quat_rotate_inv_wxyz(q, 2.0 * q_err[1:4])
            self._w_filt = c.w_filter * self._w_filt + (1.0 - c.w_filter) * w
            w_body = _quat_rotate_inv_wxyz(q, self._w_filt)
            tau_des = c.kp_att * err_body - c.kd_att * w_body       # desired body torque
            lt = self.lever * max(self.f_max * throttle, 1e-3)
            # invert torque_body ~ [L T sin(gy θ), -L T sin(gx θ), 0]
            g = np.array([
                np.arcsin(np.clip(-tau_des[1] / lt, -sd, sd)) / self.max_gimbal_rad,
                np.arcsin(np.clip(tau_des[0] / lt, -sd, sd)) / self.max_gimbal_rad,
            ])

        g = np.clip(g, self._gimbal_prev - c.gimbal_rate, self._gimbal_prev + c.gimbal_rate)
        g = c.gimbal_lp * self._gimbal_prev + (1.0 - c.gimbal_lp) * g
        g = np.clip(g, -c.gimbal_max, c.gimbal_max)
        self._gimbal_prev = g
        return np.array([throttle, g[0], g[1]], dtype=np.float32)


# ---------------------------------------------------------------- mission variants

CLOSED_LOOP_SCENARIOS: dict[str, ControllerConfig] = {
    "c1_nominal": ControllerConfig(),
    "c2_divert": ControllerConfig(target_offset_xy=(6.0, -4.0)),
    "c3_fast_descent": ControllerConfig(vz_scale=1.8, vz_max=4.5),
    "c4_tight_guidance": ControllerConfig(kp_pos=0.7, kd_pos=1.6, gimbal_max=0.5),
}

# initial-attitude perturbation (deg, added to the reset euler) per scenario
CLOSED_LOOP_INIT_TILT_DEG: dict[str, tuple[float, float, float]] = {
    "c5_high_tilt": (7.0, -6.0, 0.0),
}
CLOSED_LOOP_SCENARIOS["c5_high_tilt"] = ControllerConfig()

ALL_CLOSED_LOOP: list[str] = sorted(CLOSED_LOOP_SCENARIOS)
