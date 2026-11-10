"""Static checks for the analytic-vs-PhysX fidelity tooling (no Isaac needed)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.adapter_fidelity_scenarios import (
    ALL_SCENARIOS,
    S6_NAME,
    SCENARIOS,
    build_action_sequence,
    hover_throttle,
    rollout_scripted_descent,
)

_CONFIGS = {
    "sim": {"gravity_mps2": 1.62, "seed": 1234},
    "rocket": {"mass_kg": 1.63, "thrust": {"max_newtons": 30.0}},
    "terrain": {},
}


class ScenarioTests(unittest.TestCase):
    def test_hover_throttle_cancels_gravity(self) -> None:
        thr = hover_throttle(_CONFIGS)
        self.assertAlmostEqual(thr, 1.63 * 1.62 / 30.0, places=6)
        self.assertGreater(thr, 0.0)
        self.assertLess(thr, 1.0)

    def test_all_scenarios_shape_and_bounds(self) -> None:
        for name in SCENARIOS:
            seq = build_action_sequence(name, 300, _CONFIGS, seed=0)
            self.assertEqual(seq.shape, (300, 3))
            self.assertEqual(seq.dtype, np.float32)
            self.assertTrue(np.all(seq[:, 0] >= 0.0) and np.all(seq[:, 0] <= 1.0))
            self.assertTrue(np.all(np.abs(seq[:, 1:]) <= 1.0))

    def test_scenarios_are_deterministic(self) -> None:
        a = build_action_sequence("s5_fixed_random", 200, _CONFIGS, seed=7)
        b = build_action_sequence("s5_fixed_random", 200, _CONFIGS, seed=7)
        c = build_action_sequence("s5_fixed_random", 200, _CONFIGS, seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_unknown_scenario_raises(self) -> None:
        with self.assertRaises(KeyError):
            build_action_sequence("nope", 10, _CONFIGS, seed=0)


class _FakeAnalytic:
    """Flat-ground point-mass mirror of the analytic adapter for hermetic S6 tests."""

    max_gimbal_deg = 12.0

    def __init__(self, pos, target_xy, dt=1.0 / 60.0, mass=1.63, g=1.62, f_max=30.0):
        self.state = type("S", (), {})()
        self.state.position = np.asarray(pos, dtype=np.float64)
        self.state.velocity = np.zeros(3)
        self.reset_sample = type(
            "R", (), {"target_position": (float(target_xy[0]), float(target_xy[1]), 0.0)}
        )()
        self.dt, self.mass, self.g, self.f_max = dt, mass, g, f_max

    def _altitude_to_ground(self, position):
        return max(0.0, float(position[2]))

    def step(self, action):
        thr, gx, gy = (float(action[0]), float(action[1]), float(action[2]))
        mgr = np.deg2rad(self.max_gimbal_deg)
        d = np.array([np.sin(gx * mgr), np.sin(gy * mgr), np.cos(gx * mgr) * np.cos(gy * mgr)])
        d /= max(float(np.linalg.norm(d)), 1e-6)
        acc = np.array([0.0, 0.0, -self.g]) + d * (self.f_max * thr / self.mass)
        self.state.velocity = self.state.velocity + acc * self.dt
        self.state.position = self.state.position + self.state.velocity * self.dt
        if self.state.position[2] < 0.0:
            self.state.position[2] = 0.0


_S6_CONFIGS = {
    "sim": {"gravity_mps2": 1.62, "seed": 1234},
    "rocket": {"mass_kg": 1.63, "thrust": {"max_newtons": 30.0}},
}


class ScriptedDescentTests(unittest.TestCase):
    def test_matrix_scenario_set(self) -> None:
        self.assertEqual(ALL_SCENARIOS, sorted(SCENARIOS) + [S6_NAME])
        self.assertNotIn(S6_NAME, SCENARIOS)

    def test_rollout_descends_and_steers(self) -> None:
        adapter = _FakeAnalytic(pos=[-14.7, -26.6, 20.0], target_xy=[-11.6, -24.4])
        actions = rollout_scripted_descent(adapter, _S6_CONFIGS, 700)
        self.assertEqual(actions.shape, (700, 3))
        self.assertEqual(actions.dtype, np.float32)
        self.assertTrue(np.all(actions[:, 0] >= 0.0) and np.all(actions[:, 0] <= 1.0))
        self.assertTrue(np.all(np.abs(actions[:, 1:]) <= 1.0))
        self.assertLess(adapter.state.position[2], 20.0)              # it went down
        self.assertGreater(np.abs(actions[:, 1:]).max(), 0.01)       # it used the gimbal
        # steered toward the target in the horizontal plane
        start_err = np.hypot(-14.7 - -11.6, -26.6 - -24.4)
        end_err = np.hypot(adapter.state.position[0] - -11.6, adapter.state.position[1] - -24.4)
        self.assertLess(end_err, start_err)

    def test_rollout_is_deterministic(self) -> None:
        a = rollout_scripted_descent(_FakeAnalytic([-14.7, -26.6, 18.0], [-11.6, -24.4]), _S6_CONFIGS, 400)
        b = rollout_scripted_descent(_FakeAnalytic([-14.7, -26.6, 18.0], [-11.6, -24.4]), _S6_CONFIGS, 400)
        np.testing.assert_array_equal(a, b)

    def test_s6_build_from_actions_json(self) -> None:
        seq_in = np.zeros((150, 3), dtype=np.float32)
        seq_in[:, 0] = 0.3
        seq_in[:, 1] = 0.2
        path = Path(tempfile.mkdtemp()) / "actions.json"
        path.write_text(json.dumps({"actions": seq_in.tolist()}))
        seq = build_action_sequence(S6_NAME, 150, _S6_CONFIGS, seed=0, actions_json=str(path))
        self.assertEqual(seq.shape, (150, 3))
        np.testing.assert_allclose(seq, seq_in, atol=1e-6)

    def test_s6_requires_actions_json(self) -> None:
        with self.assertRaises(ValueError):
            build_action_sequence(S6_NAME, 100, _S6_CONFIGS, seed=0)

    def test_s6_actions_padded_and_trimmed(self) -> None:
        path = Path(tempfile.mkdtemp()) / "short.json"
        path.write_text(json.dumps({"actions": np.full((80, 3), 0.1).tolist()}))
        padded = build_action_sequence(S6_NAME, 120, _S6_CONFIGS, seed=0, actions_json=str(path))
        self.assertEqual(padded.shape, (120, 3))
        np.testing.assert_array_equal(padded[80:], 0.0)
        trimmed = build_action_sequence(S6_NAME, 50, _S6_CONFIGS, seed=0, actions_json=str(path))
        self.assertEqual(trimmed.shape, (50, 3))


class ClosedLoopControllerTests(unittest.TestCase):
    CFG = {
        "rocket": {"mass_kg": 1.63, "thrust": {"max_newtons": 30.0, "max_gimbal_deg": 12.0, "lever_arm_m": 0.85}},
        "sim": {"gravity_mps2": 1.62},
    }

    def _state(self, pos, quat, vel=(0, 0, 0), omega=(0, 0, 0)):
        from app.rl_types import RocketKinematicState
        return RocketKinematicState(
            position=np.array(pos, float), target_position=np.zeros(3),
            linear_velocity=np.array(vel, float), quaternion_wxyz=np.array(quat, float),
            angular_velocity=np.array(omega, float), linear_acceleration=np.zeros(3),
            altitude_to_ground=max(float(pos[2]), 0.0), main_thrust=0, gimbal_x=0, gimbal_y=0,
            out_of_bounds=False,
        )

    def test_scenarios_and_bounds(self) -> None:
        from app.adapter_fidelity_controller import ALL_CLOSED_LOOP, CLOSED_LOOP_SCENARIOS, DescentController
        self.assertEqual(ALL_CLOSED_LOOP, sorted(CLOSED_LOOP_SCENARIOS))
        for name in CLOSED_LOOP_SCENARIOS:
            ctrl = DescentController(self.CFG, CLOSED_LOOP_SCENARIOS[name])
            a = ctrl(self._state([3.0, -2.0, 18.0], [1.0, 0, 0, 0]))
            self.assertEqual(a.shape, (3,))
            self.assertGreaterEqual(a[0], 0.0)
            self.assertLessEqual(a[0], 1.0)
            self.assertLessEqual(float(np.abs(a[1:]).max()), 1.0)

    def test_damps_rotation(self) -> None:
        from app.adapter_fidelity_controller import DescentController
        ctrl = DescentController(self.CFG)
        upright = self._state([0.0, 0.0, 15.0], [1.0, 0, 0, 0])
        spinning = self._state([0.0, 0.0, 15.0], [1.0, 0, 0, 0], omega=[0.0, 2.0, 0.0])
        # a positive body-Y rate should be opposed by a gimbal command
        self.assertGreater(float(np.abs(ctrl(spinning)[1:] - ctrl(upright)[1:]).max()), 0.02)


class MetricTests(unittest.TestCase):
    def test_quat_geodesic_zero_for_identical(self) -> None:
        from analysis.analyze_adapter_fidelity import quat_geodesic_deg

        q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))
        np.testing.assert_allclose(quat_geodesic_deg(q, q), 0.0, atol=1e-6)

    def test_usable_horizon_and_divergence(self) -> None:
        from analysis.analyze_adapter_fidelity import divergence_rate, usable_horizon

        t = np.linspace(0.0, 10.0, 101)
        err = 0.1 * t  # crosses 0.5 m just after t = 5 s, slope 0.1 m/s
        self.assertAlmostEqual(usable_horizon(t, err, 0.5), 5.1, delta=0.06)
        self.assertAlmostEqual(divergence_rate(t, err), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
