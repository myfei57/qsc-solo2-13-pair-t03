"""循环水：温度分档投切、先加机再调塔、水位联锁与调整留痕。"""

from __future__ import annotations

import unittest

from flashsmelter.application import Application
from flashsmelter.errors import GuardViolation, StateTransitionError

from .helpers import make_app, make_root


def start_cooling(app, pool_level: float = 0.6):
    return app.cooling.start("tester", pool_level=pool_level)


def update_cooling(app, temp: float, level: float = 0.6, outlet: float = 0.0):
    return app.cooling.update(
        "tester", inlet_temp_c=temp, pool_level=level, outlet_temp_c=outlet
    )


def evaluate(app):
    return app.cooling.evaluate("tester")


def evaluate_after_interval(app):
    app.clock.advance(app.settings.cooling_adjust_min_interval_seconds + 1)
    return evaluate(app)


class CoolingStartStopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = make_app()

    def test_start_requires_pool_level(self) -> None:
        with self.assertRaises(GuardViolation) as blocked:
            self.app.cooling.start("ops", pool_level=self.app.settings.cooling_pool_level_low - 0.05)
        self.assertIn("min", blocked.exception.details)
        status = start_cooling(self.app)
        self.assertEqual("circulating", status["state"])
        self.assertEqual(1, status["pumps_running"])
        self.assertEqual(0, status["fans_running"])
        self.assertEqual(0, status["level"])

    def test_start_twice_is_rejected(self) -> None:
        start_cooling(self.app)
        with self.assertRaises(StateTransitionError):
            start_cooling(self.app)

    def test_stop_returns_idle_and_blocks_evaluate(self) -> None:
        start_cooling(self.app)
        status = self.app.cooling.stop("ops")
        self.assertEqual("idle", status["state"])
        self.assertEqual(0, status["pumps_running"])
        self.assertEqual(0, status["fans_running"])
        with self.assertRaises(StateTransitionError):
            evaluate(self.app)
        with self.assertRaises(StateTransitionError):
            self.app.cooling.stop("ops")

    def test_update_records_measurements_and_alarms(self) -> None:
        start_cooling(self.app)
        status = update_cooling(
            self.app,
            self.app.settings.cooling_inlet_temp_max_c + 1.0,
            level=self.app.settings.cooling_pool_level_max + 0.05,
            outlet=38.5,
        )
        self.assertTrue(status["alarms"]["over_temperature"])
        self.assertTrue(status["alarms"]["overflow"])
        self.assertFalse(status["alarms"]["low_level"])
        self.assertEqual(38.5, status["outlet_temp_c"])

    def test_update_rejects_bad_measurements(self) -> None:
        start_cooling(self.app)
        with self.assertRaises(GuardViolation):
            self.app.cooling.update("ops", inlet_temp_c=-1.0, pool_level=0.6)
        with self.assertRaises(GuardViolation):
            self.app.cooling.update("ops", inlet_temp_c=30.0, pool_level=1.2)


class CoolingStagingTest(unittest.TestCase):
    """温度分档决定投几台：升温先加泵后加风机，降温先退风机后退泵。"""

    def setUp(self) -> None:
        self.app = make_app()
        start_cooling(self.app)

    def _units(self) -> tuple[int, int]:
        status = self.app.cooling.status()
        return status["pumps_running"], status["fans_running"]

    def test_stage_up_pumps_first_then_fans(self) -> None:
        seen: list[tuple[int, int]] = []
        for temp in (33.0, 34.0, 35.5, 37.0, 39.0):
            update_cooling(self.app, temp)
            result = evaluate_after_interval(self.app)
            self.assertEqual("stage-up", result["evaluation"]["adjustment"])
            seen.append((result["pumps_running"], result["fans_running"]))
        self.assertEqual(
            [(2, 0), (3, 0), (3, 1), (3, 2), (3, 3)],
            seen,
            "泵必须先于风机逐级投满",
        )
        status = self.app.cooling.status()
        self.assertEqual(status["max_level"], status["level"])
        self.assertEqual(self.app.settings.cooling_pump_count, status["pumps_running"])
        self.assertEqual(self.app.settings.cooling_fan_count, status["fans_running"])

    def test_stage_down_removes_fans_first(self) -> None:
        self.test_stage_up_pumps_first_then_fans()
        seen: list[tuple[int, int]] = []
        for temp in (36.0, 34.5, 33.0, 31.0, 30.0):
            update_cooling(self.app, temp)
            result = evaluate_after_interval(self.app)
            self.assertEqual("stage-down", result["evaluation"]["adjustment"])
            seen.append((result["pumps_running"], result["fans_running"]))
        self.assertEqual(
            [(3, 2), (3, 1), (3, 0), (2, 0), (1, 0)],
            seen,
            "退档必须先退风机再退泵",
        )
        # 循环期间始终保留一台泵，不会全停。
        self.assertEqual((1, 0), self._units())
        update_cooling(self.app, 20.0)
        result = evaluate_after_interval(self.app)
        self.assertEqual("hold", result["evaluation"]["adjustment"])
        self.assertEqual((1, 0), self._units())

    def test_hysteresis_prevents_chatter(self) -> None:
        update_cooling(self.app, 33.0)
        evaluate_after_interval(self.app)
        self.assertEqual((2, 0), self._units())
        # 刚跌破阈值但在回差内：保持，不退档。
        update_cooling(self.app, self.app.settings.cooling_inlet_temp_warn_c - 0.2)
        result = evaluate_after_interval(self.app)
        self.assertEqual("hold", result["evaluation"]["adjustment"])
        self.assertEqual((2, 0), self._units())
        # 跌破回差才退档。
        update_cooling(
            self.app,
            self.app.settings.cooling_inlet_temp_warn_c
            - self.app.settings.cooling_temp_hysteresis_c
            - 0.1,
        )
        result = evaluate_after_interval(self.app)
        self.assertEqual("stage-down", result["evaluation"]["adjustment"])
        self.assertEqual((1, 0), self._units())

    def test_min_interval_limits_adjustments(self) -> None:
        update_cooling(self.app, 33.0)
        result = evaluate(self.app)
        self.assertEqual("stage-up", result["evaluation"]["adjustment"])
        # 间隔未到，即使温度继续升高也保持。
        update_cooling(self.app, 36.0)
        result = evaluate(self.app)
        self.assertEqual("hold", result["evaluation"]["adjustment"])
        self.assertEqual("interval", result["evaluation"]["reason"])
        self.assertGreater(result["evaluation"]["remaining_seconds"], 0.0)
        self.assertEqual((2, 0), self._units())

    def test_stale_measurement_blocks_evaluate(self) -> None:
        update_cooling(self.app, 33.0)
        self.app.clock.advance(self.app.settings.cooling_measure_max_age_seconds + 1)
        with self.assertRaises(GuardViolation) as stale:
            evaluate(self.app)
        self.assertIn("age_seconds", stale.exception.details)

    def test_evaluate_requires_measurement(self) -> None:
        with self.assertRaises(GuardViolation):
            evaluate(self.app)

    def test_over_temperature_saturation_is_visible(self) -> None:
        self.test_stage_up_pumps_first_then_fans()
        update_cooling(self.app, self.app.settings.cooling_inlet_temp_max_c + 2.0)
        result = evaluate_after_interval(self.app)
        self.assertEqual("hold", result["evaluation"]["adjustment"])
        self.assertEqual("saturated", result["evaluation"]["reason"])
        self.assertTrue(result["saturated"])
        self.assertTrue(result["alarms"]["over_temperature"])


class CoolingLevelInterlockTest(unittest.TestCase):
    """水池水位：偏低拦住加泵，低低闩锁停泵。"""

    def setUp(self) -> None:
        self.app = make_app()
        start_cooling(self.app)

    def _low_band(self) -> float:
        settings = self.app.settings
        return (settings.cooling_pool_level_min + settings.cooling_pool_level_low) / 2

    def test_low_level_blocks_pump_addition_but_not_fans(self) -> None:
        update_cooling(self.app, 33.0)
        evaluate(self.app)
        self.assertEqual(2, self.app.cooling.status()["pumps_running"])
        # 水位落入偏低区间：加泵被拦，档位不动，尝试留痕。
        update_cooling(self.app, 34.0, level=self._low_band())
        result = evaluate_after_interval(self.app)
        self.assertEqual("blocked", result["evaluation"]["adjustment"])
        self.assertEqual("pool-level-low", result["evaluation"]["reason"])
        status = self.app.cooling.status()
        self.assertEqual(2, status["pumps_running"])
        self.assertTrue(status["alarms"]["low_level"])
        blocked = [item for item in self.app.cooling.adjustments() if item["kind"] == "blocked"]
        self.assertEqual(1, len(blocked))
        self.assertEqual("pool-level-low", blocked[0]["reason"])

    def test_low_level_still_allows_fan_steps(self) -> None:
        for temp in (33.0, 34.0, 35.5):
            update_cooling(self.app, temp)
            evaluate_after_interval(self.app)
        self.assertEqual((3, 1), (self.app.cooling.pumps_running, self.app.cooling.fans_running))
        # 泵已投满，水位偏低只拦泵：风机档位照常可加。
        update_cooling(self.app, 37.0, level=self._low_band())
        result = evaluate_after_interval(self.app)
        self.assertEqual("stage-up", result["evaluation"]["adjustment"])
        self.assertEqual((3, 2), (result["pumps_running"], result["fans_running"]))

    def test_low_low_level_latches_and_stops_pumps(self) -> None:
        update_cooling(self.app, 34.0)
        evaluate_after_interval(self.app)
        update_cooling(self.app, 35.5)
        evaluate_after_interval(self.app)
        self.assertEqual(3, self.app.cooling.pumps_running)
        status = update_cooling(self.app, 35.5, level=self.app.settings.cooling_pool_level_min - 0.05)
        self.assertEqual("latched", status["state"])
        self.assertEqual("pool-level-low", status["latch_reason"])
        self.assertEqual(0, status["pumps_running"])
        self.assertEqual(0, status["fans_running"])
        self.assertEqual(1, status["latch_count"])
        # 闩锁期间任何调整都被拦住。
        with self.assertRaises(StateTransitionError):
            evaluate(self.app)
        # 水位仍低时刷新闩锁，保持时长重新计。
        self.app.clock.advance(self.app.settings.cooling_latch_min_hold_seconds - 5)
        status = update_cooling(self.app, 35.5, level=self.app.settings.cooling_pool_level_min - 0.05)
        self.assertEqual(2, status["latch_count"])
        self.assertGreater(status["hold_remaining_seconds"], 0.0)

    def test_reset_requires_hold_note_and_recovered_level(self) -> None:
        update_cooling(self.app, 30.0, level=0.2)
        self.assertTrue(self.app.cooling.is_latched())
        with self.assertRaises(GuardViolation) as early:
            self.app.cooling.reset("ops", note="补水完成", pool_level=0.6)
        self.assertIn("remaining_seconds", early.exception.details)
        self.app.clock.advance(self.app.settings.cooling_latch_min_hold_seconds + 1)
        with self.assertRaises(GuardViolation):
            self.app.cooling.reset("ops", note="补水完成", pool_level=0.3)
        status = self.app.cooling.reset("ops", note="补水完成", pool_level=0.6)
        self.assertEqual("circulating", status["state"])
        self.assertEqual(1, status["pumps_running"])
        self.assertIsNone(status["latch_reason"])

    def test_stop_from_latched_clears_interlock(self) -> None:
        update_cooling(self.app, 30.0, level=0.2)
        status = self.app.cooling.stop("ops")
        self.assertEqual("idle", status["state"])
        self.assertIsNone(status["latch_reason"])
        self.assertEqual(1, status["latch_count"])


class CoolingDurabilityTest(unittest.TestCase):
    """每次调整都留得住：调整流水、审计与重启恢复。"""

    def test_adjustments_stream_and_audit_trail(self) -> None:
        app = make_app()
        start_cooling(app)
        update_cooling(app, 33.0)
        evaluate(app)
        update_cooling(app, 34.0)
        evaluate_after_interval(app)
        stream = app.cooling.adjustments()
        self.assertEqual(2, len(stream))
        self.assertEqual(["stage-up", "stage-up"], [item["kind"] for item in stream])
        self.assertEqual([(0, 1), (1, 2)], [(item["from_level"], item["to_level"]) for item in stream])
        events = app.audit_events(action="evaluate", limit=10)
        self.assertEqual(2, len(events))
        self.assertTrue(all(event["outcome"] == "ok" for event in events))
        kinds = [event["details"].get("adjustment") for event in events]
        self.assertEqual(["stage-up", "stage-up"], kinds)
        self.assertTrue(all(event["details"].get("key") for event in events))

    def test_state_survives_restart(self) -> None:
        root = make_root()
        app = make_app(root=root)
        start_cooling(app)
        update_cooling(app, 34.0)
        evaluate(app)
        update_cooling(app, 34.0, level=0.2)
        self.assertEqual("latched", app.cooling.state)
        restarted = Application(app.settings, clock=app.clock)
        status = restarted.cooling.status()
        self.assertEqual("latched", status["state"])
        self.assertEqual("pool-level-low", status["latch_reason"])
        self.assertEqual(1, status["latch_count"])
        self.assertEqual(0, status["pumps_running"])
        self.assertEqual(1, len(restarted.cooling.adjustments()))
        self.assertTrue(restarted.store.verify().ok)

    def test_circulating_state_survives_restart(self) -> None:
        root = make_root()
        app = make_app(root=root)
        start_cooling(app)
        update_cooling(app, 34.0)
        evaluate(app)
        restarted = Application(app.settings, clock=app.clock)
        status = restarted.cooling.status()
        self.assertEqual("circulating", status["state"])
        self.assertEqual(2, status["pumps_running"])
        self.assertEqual(0, status["fans_running"])
        self.assertAlmostEqual(34.0, status["inlet_temp_c"], places=3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
