"""循环水系统：温度分档投切、低水位联锁、复位门控与调整留痕。"""

from __future__ import annotations

import unittest

from flashsmelter.application import Application
from flashsmelter.errors import GuardViolation, StateTransitionError

from .helpers import make_app, make_root


class CoolingStagingTest(unittest.TestCase):
    """按进水温度决定泵/风机台数：升档先加泵再加塔，降档先退塔再退泵。"""

    def setUp(self) -> None:
        self.app = make_app()
        self.app.cooling.start("ops", pool_level=0.6)

    def test_start_runs_single_pump(self) -> None:
        status = self.app.cooling.status()
        self.assertEqual("circulating", status["state"])
        self.assertEqual(0, status["stage"])
        self.assertEqual(1, status["pumps_run"])
        self.assertEqual(0, status["fans_run"])

    def test_start_rejected_on_low_level(self) -> None:
        app = make_app()
        with self.assertRaises(GuardViolation) as blocked:
            app.cooling.start("ops", pool_level=app.settings.cooling_level_min - 0.01)
        self.assertIn("min", blocked.exception.details)

    def test_escalation_adds_pumps_before_fans(self) -> None:
        high = self.app.settings.cooling_temp_high_c
        step = self.app.settings.cooling_temp_step_c
        status = self.app.cooling.update("ops", inlet_temp_c=high + 0.5, pool_level=0.6)
        self.assertEqual((2, 0), (status["pumps_run"], status["fans_run"]))
        status = self.app.cooling.update("ops", inlet_temp_c=high + step + 0.5, pool_level=0.6)
        self.assertEqual((3, 0), (status["pumps_run"], status["fans_run"]))
        # 泵已加满，水温继续走高才轮到塔风机。
        status = self.app.cooling.update("ops", inlet_temp_c=high + 2 * step + 0.5, pool_level=0.6)
        self.assertEqual((3, 1), (status["pumps_run"], status["fans_run"]))
        status = self.app.cooling.update("ops", inlet_temp_c=high + 4 * step + 0.5, pool_level=0.6)
        self.assertEqual((3, 3), (status["pumps_run"], status["fans_run"]))
        self.assertEqual(status["max_stage"], status["stage"])
        # 温度再升也不会突破装机台数。
        status = self.app.cooling.update("ops", inlet_temp_c=high + 40.0, pool_level=0.6)
        self.assertEqual((3, 3), (status["pumps_run"], status["fans_run"]))

    def test_deescalation_removes_fans_before_pumps(self) -> None:
        high = self.app.settings.cooling_temp_high_c
        low = self.app.settings.cooling_temp_low_c
        step = self.app.settings.cooling_temp_step_c
        self.app.cooling.update("ops", inlet_temp_c=high + 2 * step + 0.5, pool_level=0.6)
        self.assertEqual(1, self.app.cooling.status()["fans_run"])
        # 温度回落：先退风机，泵保持满开。
        status = self.app.cooling.update("ops", inlet_temp_c=low + 2 * step - 0.5, pool_level=0.6)
        self.assertEqual((3, 0), (status["pumps_run"], status["fans_run"]))
        status = self.app.cooling.update("ops", inlet_temp_c=low + step - 0.5, pool_level=0.6)
        self.assertEqual((2, 0), (status["pumps_run"], status["fans_run"]))
        status = self.app.cooling.update("ops", inlet_temp_c=low - 0.5, pool_level=0.6)
        self.assertEqual((1, 0), (status["pumps_run"], status["fans_run"]))

    def test_deadband_holds_stage(self) -> None:
        high = self.app.settings.cooling_temp_high_c
        low = self.app.settings.cooling_temp_low_c
        self.app.cooling.update("ops", inlet_temp_c=high + 0.5, pool_level=0.6)
        self.assertEqual(1, self.app.cooling.status()["stage"])
        # 回差区间内温度来回，档位保持不动。
        for temp in (low + 0.3, high - 0.3, low + 1.0):
            status = self.app.cooling.update("ops", inlet_temp_c=temp, pool_level=0.6)
            self.assertEqual(1, status["stage"])
            self.assertEqual(2, status["pumps_run"])

    def test_update_while_idle_only_records_readings(self) -> None:
        app = make_app()
        status = app.cooling.update("ops", inlet_temp_c=40.0, pool_level=0.6)
        self.assertEqual("idle", status["state"])
        self.assertEqual(0, status["pumps_run"])
        self.assertAlmostEqual(40.0, status["inlet_temp_c"], places=3)

    def test_reading_out_of_range_rejected(self) -> None:
        with self.assertRaises(GuardViolation):
            self.app.cooling.update("ops", inlet_temp_c=-5.0, pool_level=0.6)
        with self.assertRaises(GuardViolation):
            self.app.cooling.update("ops", inlet_temp_c=30.0, pool_level=1.5)

    def test_stop_goes_idle_and_cuts_all(self) -> None:
        self.app.cooling.update("ops", inlet_temp_c=40.0, pool_level=0.6)
        status = self.app.cooling.stop("ops")
        self.assertEqual("idle", status["state"])
        self.assertEqual((0, 0), (status["pumps_run"], status["fans_run"]))
        with self.assertRaises(StateTransitionError):
            self.app.cooling.stop("ops")


class CoolingLatchTest(unittest.TestCase):
    """水池低水位联锁：立即切泵、保持闩锁、人工确认后复位。"""

    def setUp(self) -> None:
        self.app = make_app()
        self.app.cooling.start("ops", pool_level=0.6)

    def test_low_level_trips_pumps_and_latches(self) -> None:
        self.app.cooling.update("ops", inlet_temp_c=40.0, pool_level=0.6)
        self.assertGreater(self.app.cooling.status()["pumps_run"], 1)
        status = self.app.cooling.update(
            "ops", inlet_temp_c=40.0, pool_level=self.app.settings.cooling_level_min - 0.01
        )
        self.assertEqual("latched", status["state"])
        self.assertEqual("pool-level-low", status["latch_reason"])
        self.assertEqual((0, 0), (status["pumps_run"], status["fans_run"]))
        self.assertEqual(1, status["latch_count"])

    def test_latched_update_keeps_pumps_cut(self) -> None:
        self.app.cooling.update("ops", inlet_temp_c=40.0, pool_level=0.2)
        status = self.app.cooling.update("ops", inlet_temp_c=45.0, pool_level=0.6)
        self.assertEqual("latched", status["state"])
        self.assertEqual(0, status["pumps_run"])

    def test_reset_requires_hold_note_and_level(self) -> None:
        self.app.cooling.update("ops", inlet_temp_c=40.0, pool_level=0.2)
        with self.assertRaises(GuardViolation) as early:
            self.app.cooling.reset("ops", note="补水完成", pool_level=0.6)
        self.assertIn("remaining_seconds", early.exception.details)
        self.app.clock.advance(self.app.settings.cooling_latch_min_hold_seconds + 1)
        with self.assertRaises(GuardViolation):
            self.app.cooling.reset("ops", note="", pool_level=0.6)
        with self.assertRaises(GuardViolation):
            self.app.cooling.reset(
                "ops", note="补水完成", pool_level=self.app.settings.cooling_level_min - 0.01
            )
        status = self.app.cooling.reset("ops", note="补水完成", pool_level=0.6)
        self.assertEqual("circulating", status["state"])
        self.assertEqual(1, status["pumps_run"])
        self.assertEqual(1, status["latch_count"])
        # 复位后温度分档恢复自动投切。
        status = self.app.cooling.update("ops", inlet_temp_c=36.0, pool_level=0.6)
        self.assertEqual(2, status["pumps_run"])


class CoolingDurabilityTest(unittest.TestCase):
    """每次调整都留得住：调整记录随状态落盘，审计流水可查，重启不丢。"""

    def test_adjustments_recorded_and_audited(self) -> None:
        app = make_app()
        app.cooling.start("ops", pool_level=0.6)
        app.cooling.update("ops", inlet_temp_c=36.0, pool_level=0.6)
        app.cooling.update("ops", inlet_temp_c=40.0, pool_level=0.6)
        app.cooling.update("ops", inlet_temp_c=30.0, pool_level=0.6)
        adjustments = app.cooling.status()["adjustments"]
        # 投运 + 两次升档 + 一次降档，逐条留痕。
        self.assertEqual(4, len(adjustments))
        self.assertEqual(
            [(0, 0), (0, 1), (1, 3), (3, 0)],
            [(entry["from_stage"], entry["to_stage"]) for entry in adjustments],
        )
        last = adjustments[-1]
        self.assertEqual(1, last["pumps_run"])
        self.assertEqual(0, last["fans_run"])
        self.assertIn("at", last)
        self.assertIn("reason", last)
        events = app.audit_events(action="update", target="cooling")
        self.assertEqual(3, len(events))
        self.assertTrue(all(event["outcome"] == "ok" for event in events))

    def test_rejected_reset_is_audited(self) -> None:
        app = make_app()
        app.cooling.start("ops", pool_level=0.6)
        app.cooling.update("ops", inlet_temp_c=40.0, pool_level=0.2)
        with self.assertRaises(GuardViolation):
            app.cooling.reset("ops", note="补水完成", pool_level=0.6)
        events = app.audit_events(action="reset", target="cooling", outcome="rejected")
        self.assertEqual(1, len(events))

    def test_state_survives_restart(self) -> None:
        root = make_root()
        app = make_app(root=root)
        app.cooling.start("ops", pool_level=0.6)
        app.cooling.update("ops", inlet_temp_c=38.0, pool_level=0.6)
        app.cooling.update("ops", inlet_temp_c=38.0, pool_level=0.2)
        restarted = Application(app.settings, clock=app.clock)
        status = restarted.cooling.status()
        self.assertEqual("latched", status["state"])
        self.assertEqual("pool-level-low", status["latch_reason"])
        self.assertEqual(1, status["latch_count"])
        self.assertEqual(0, status["pumps_run"])
        self.assertEqual(3, len(status["adjustments"]))
        self.assertTrue(restarted.store.verify().ok)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
