"""炉体/水套循环水组件。

循环水带走炉体与水套的热量，经冷却塔降温后回到水池。过去泵常开、水温高了
加水、加多了溢流，泵与冷却塔风机全靠人盯。本组件把这套循环收进平台：

* 按进水温度分档决定泵与风机投几台；升温先加泵（机）后加风机（塔），降温
  按相反顺序退档，并带回差与最小调整间隔，避免边界抖动与频繁启停；
* 水池水位低低时闩锁并停掉全部泵机（拦住泵）；闩锁是保持型的，水位恢复后
  仍须人工在最短保持时长之后显式复位，复位说明先落盘；
* 水位偏低（未到低低）时禁止加泵，风机投退不受限；溢流与进水越限都记报警；
* 每一次档位调整（含被联锁拦下的尝试）都落盘并追加到调整流水，班后可追溯。
"""

from __future__ import annotations

from typing import Any, Mapping

from ..component import Component, ensure_actor
from ..errors import GuardViolation, StateTransitionError
from ..machine import StateMachine
from ..runtime import RuntimeContext

STATES = ("idle", "circulating", "latched")

TRANSITIONS: Mapping[str, tuple[str, ...]] = {
    "idle": ("circulating",),
    "circulating": ("idle", "latched"),
    "latched": ("circulating", "idle"),
}

ADJUST_STREAM = "cooling/adjustments"


class CoolingWaterSystem(Component):
    name = "cooling"

    def __init__(self, ctx: RuntimeContext) -> None:
        super().__init__(ctx)
        self._machine = StateMachine("cooling", "idle", TRANSITIONS, ctx.clock)
        self._level = 0
        self._inlet_temp_c = 0.0
        self._outlet_temp_c = 0.0
        self._pool_level = 0.0
        self._over_temperature = False
        self._low_level = False
        self._overflow = False
        self._latch_reason: str | None = None
        self._latched_at: float | None = None
        self._latch_count = 0
        self._last_update_at: str | None = None
        self._last_update_epoch: float | None = None
        self._last_adjust_epoch: float | None = None
        self._last_adjustment: dict[str, Any] | None = None
        restored = self.restore()
        if restored is not None:
            self._machine.restore(restored)
            self._level = int(restored.get("level", 0))
            self._inlet_temp_c = float(restored.get("inlet_temp_c", 0.0))
            self._outlet_temp_c = float(restored.get("outlet_temp_c", 0.0))
            self._pool_level = float(restored.get("pool_level", 0.0))
            self._over_temperature = bool(restored.get("over_temperature", False))
            self._low_level = bool(restored.get("low_level", False))
            self._overflow = bool(restored.get("overflow", False))
            self._latch_reason = restored.get("latch_reason")
            self._latched_at = restored.get("latched_at")
            self._latch_count = int(restored.get("latch_count", 0))
            self._last_update_at = restored.get("last_update_at")
            self._last_update_epoch = restored.get("last_update_epoch")
            self._last_adjust_epoch = restored.get("last_adjust_epoch")
            last = restored.get("last_adjustment")
            if isinstance(last, dict):
                self._last_adjustment = last
        self._clamp_level()
        self._refresh_gauges()

    # ------------------------------------------------------------------ 动作
    def start(
        self,
        actor: str,
        *,
        pool_level: float,
        correlation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> Mapping[str, Any]:
        actor = ensure_actor(actor)
        with self.action(
            "start",
            "cooling",
            actor,
            correlation_id=correlation_id,
            expected_generation=expected_generation,
        ) as trace:
            self._machine.require("idle", "投入循环水")
            if pool_level < self.settings.cooling_pool_level_low:
                raise GuardViolation(
                    "水池水位低于启动下限，禁止投泵",
                    details={"pool_level": pool_level, "min": self.settings.cooling_pool_level_low},
                )
            self._pool_level = pool_level
            self._low_level = False
            self._level = 0
            self._machine.to("circulating", actor, "首台循环泵投运")
            record = self._persist(reason="start")
            trace.attach(record).note("pool_level", pool_level).note("pumps_running", self.pumps_running)
            return self.status()

    def update(
        self,
        actor: str,
        *,
        inlet_temp_c: float,
        pool_level: float,
        outlet_temp_c: float = 0.0,
        correlation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> Mapping[str, Any]:
        actor = ensure_actor(actor)
        with self.action(
            "update",
            "cooling",
            actor,
            correlation_id=correlation_id,
            expected_generation=expected_generation,
        ) as trace:
            if inlet_temp_c < 0 or outlet_temp_c < 0 or pool_level < 0:
                raise GuardViolation(
                    "循环水测点不能为负",
                    details={
                        "inlet_temp_c": inlet_temp_c,
                        "outlet_temp_c": outlet_temp_c,
                        "pool_level": pool_level,
                    },
                )
            if pool_level > 1.0:
                raise GuardViolation(
                    "水池水位必须是 (0,1] 区间比例", details={"pool_level": pool_level}
                )
            self._inlet_temp_c = inlet_temp_c
            self._outlet_temp_c = outlet_temp_c
            self._pool_level = pool_level
            self._last_update_epoch = self.clock.timestamp()
            self._last_update_at = self.clock.timestamp_iso()
            self._over_temperature = inlet_temp_c >= self.settings.cooling_inlet_temp_max_c
            self._low_level = pool_level < self.settings.cooling_pool_level_low
            self._overflow = pool_level > self.settings.cooling_pool_level_max
            if pool_level < self.settings.cooling_pool_level_min and self._machine.state in (
                "circulating",
                "latched",
            ):
                intent = self.write_intent(
                    "latch",
                    {
                        "action": "latch",
                        "reason": "pool-level-low",
                        "pool_level": pool_level,
                        "min": self.settings.cooling_pool_level_min,
                        "at": self.clock.timestamp_iso(),
                        "actor": actor,
                    },
                )
                if self._machine.state != "latched":
                    self._machine.to("latched", actor, "水池水位低低，停掉全部泵机")
                self._level = 0
                self._latch_reason = "pool-level-low"
                self._latched_at = self.clock.timestamp()
                self._latch_count += 1
                record = self._persist(reason="latch")
                trace.attach(record).note("latch_reason", "pool-level-low").note(
                    "intent_version", intent.version
                )
                return self.status()
            record = self._persist(reason="update")
            trace.attach(record).note("over_temperature", self._over_temperature).note(
                "low_level", self._low_level
            ).note("overflow", self._overflow)
            return self.status()

    def evaluate(
        self,
        actor: str,
        *,
        correlation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> Mapping[str, Any]:
        """阶梯控制周期：按进水温度算目标档位，一次只调一档。

        升温先加泵后加风机，降温先退风机后退泵；水位偏低时加泵被拦下（风机不
        受限）；每次调用都把结论写进返回值与审计，档位有变还追加调整流水。
        """

        actor = ensure_actor(actor)
        with self.action(
            "evaluate",
            "cooling",
            actor,
            correlation_id=correlation_id,
            expected_generation=expected_generation,
        ) as trace:
            self._machine.require("circulating", "循环水阶梯调整")
            self._require_fresh_measurement()
            target = self._target_level(self._inlet_temp_c)
            target_down = self._target_level(
                self._inlet_temp_c + self.settings.cooling_temp_hysteresis_c
            )
            interval_remaining = self._interval_remaining()
            if interval_remaining > 0:
                evaluation = {"adjustment": "hold", "reason": "interval", "remaining_seconds": interval_remaining}
                trace.note("adjustment", "hold").note("reason", "interval").note(
                    "remaining_seconds", interval_remaining
                )
                return self._evaluation_result(evaluation)
            if target > self._level:
                next_level = self._level + 1
                if self._is_pump_step(next_level) and self._low_level:
                    self._last_adjust_epoch = self.clock.timestamp()
                    self._record_adjustment(
                        actor,
                        kind="blocked",
                        from_level=self._level,
                        to_level=self._level,
                        reason="pool-level-low",
                        target_level=target,
                    )
                    record = self._persist(reason="adjust-blocked")
                    evaluation = {
                        "adjustment": "blocked",
                        "reason": "pool-level-low",
                        "level": self._level,
                        "target_level": target,
                    }
                    trace.attach(record).note("adjustment", "blocked").note("reason", "pool-level-low")
                    return self._evaluation_result(evaluation)
                return self._apply_level(actor, next_level, "stage-up", target, trace)
            if target_down < self._level:
                return self._apply_level(actor, self._level - 1, "stage-down", target, trace)
            reason = "band"
            if self._over_temperature and self._level >= self._max_level():
                reason = "saturated"  # 泵与风机全投仍越限：水温压不住，需人工介入
            evaluation = {"adjustment": "hold", "reason": reason, "level": self._level, "target_level": target}
            trace.note("adjustment", "hold").note("reason", reason).note("target_level", target)
            return self._evaluation_result(evaluation)

    def stop(
        self,
        actor: str,
        *,
        correlation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> Mapping[str, Any]:
        actor = ensure_actor(actor)
        with self.action(
            "stop",
            "cooling",
            actor,
            correlation_id=correlation_id,
            expected_generation=expected_generation,
        ) as trace:
            self._machine.require_one_of(("circulating", "latched"), "停用循环水")
            self._level = 0
            self._machine.to("idle", actor, "停用循环，泵与风机全停")
            self._latch_reason = None
            self._latched_at = None
            record = self._persist(reason="stop")
            trace.attach(record)
            return self.status()

    def reset(
        self,
        actor: str,
        *,
        note: str,
        pool_level: float,
        correlation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> Mapping[str, Any]:
        actor = ensure_actor(actor)
        with self.action(
            "reset",
            "cooling",
            actor,
            correlation_id=correlation_id,
            expected_generation=expected_generation,
        ) as trace:
            self._machine.require("latched", "联锁复位")
            if not note:
                raise GuardViolation("复位必须填写处理说明")
            remaining = self._hold_remaining()
            if remaining > 0:
                raise GuardViolation(
                    "闩锁最短保持时长未到，禁止复位",
                    details={
                        "remaining_seconds": round(remaining, 3),
                        "min_hold_seconds": self.settings.cooling_latch_min_hold_seconds,
                    },
                )
            if pool_level < self.settings.cooling_pool_level_low:
                raise GuardViolation(
                    "水池水位未恢复，禁止复位",
                    details={"pool_level": pool_level, "min": self.settings.cooling_pool_level_low},
                )
            intent = self.write_intent(
                "reset",
                {
                    "action": "reset",
                    "note": note,
                    "latch_reason": self._latch_reason,
                    "at": self.clock.timestamp_iso(),
                    "actor": actor,
                },
            )
            self._pool_level = pool_level
            self._low_level = False
            self._level = 0
            self._machine.to("circulating", actor, f"闩锁复位：{note}")
            self._latch_reason = None
            self._latched_at = None
            record = self._persist(reason="reset")
            trace.attach(record).note("note", note).note("intent_version", intent.version)
            return self.status()

    # ------------------------------------------------------------------ 查询
    @property
    def state(self) -> str:
        return self._machine.state

    @property
    def pumps_running(self) -> int:
        if self._machine.state != "circulating":
            return 0
        return min(self._level + 1, self.settings.cooling_pump_count)

    @property
    def fans_running(self) -> int:
        if self._machine.state != "circulating":
            return 0
        return max(
            0,
            min(
                self._level - self.settings.cooling_pump_count + 1,
                self.settings.cooling_fan_count,
            ),
        )

    def is_latched(self) -> bool:
        return self._machine.state == "latched"

    def hold_remaining(self) -> float:
        return self._hold_remaining()

    def adjustments(self, *, limit: int = 20) -> list[Mapping[str, Any]]:
        """档位调整流水，含被联锁拦下的尝试，供班报与回溯。"""

        return [entry.payload for entry in self.store.read_stream(ADJUST_STREAM, limit=limit)]

    def status(self) -> Mapping[str, Any]:
        settings = self.settings
        return {
            "state": self._machine.state,
            "level": self._level,
            "max_level": self._max_level(),
            "pumps_running": self.pumps_running,
            "fans_running": self.fans_running,
            "pump_count": settings.cooling_pump_count,
            "fan_count": settings.cooling_fan_count,
            "inlet_temp_c": round(self._inlet_temp_c, 3),
            "outlet_temp_c": round(self._outlet_temp_c, 3),
            "pool_level": round(self._pool_level, 4),
            "target_level": (
                self._target_level(self._inlet_temp_c) if self._last_update_epoch is not None else None
            ),
            "saturated": bool(self._over_temperature and self._level >= self._max_level()),
            "alarms": {
                "over_temperature": self._over_temperature,
                "low_level": self._low_level,
                "overflow": self._overflow,
            },
            "latch_reason": self._latch_reason,
            "latched_at": self._latched_at,
            "latch_count": self._latch_count,
            "hold_remaining_seconds": round(self._hold_remaining(), 3),
            "last_update_at": self._last_update_at,
            "last_adjustment": self._last_adjustment,
            "thresholds": {
                "inlet_temp_warn_c": settings.cooling_inlet_temp_warn_c,
                "inlet_temp_step_c": settings.cooling_inlet_temp_step_c,
                "temp_hysteresis_c": settings.cooling_temp_hysteresis_c,
                "inlet_temp_max_c": settings.cooling_inlet_temp_max_c,
                "pool_level_low": settings.cooling_pool_level_low,
                "pool_level_min": settings.cooling_pool_level_min,
                "pool_level_max": settings.cooling_pool_level_max,
            },
            "history": list(self._machine.history),
        }

    # ------------------------------------------------------------------ 内部
    def _max_level(self) -> int:
        return self.settings.cooling_pump_count + self.settings.cooling_fan_count - 1

    def _clamp_level(self) -> None:
        self._level = max(0, min(self._level, self._max_level()))

    def _is_pump_step(self, level: int) -> bool:
        """升到该档是否多投一台泵；否则多投一台风机。"""

        return level < self.settings.cooling_pump_count

    def _target_level(self, temp_c: float) -> int:
        settings = self.settings
        if temp_c < settings.cooling_inlet_temp_warn_c:
            return 0
        steps = int((temp_c - settings.cooling_inlet_temp_warn_c) // settings.cooling_inlet_temp_step_c)
        return min(1 + steps, self._max_level())

    def _require_fresh_measurement(self) -> None:
        if self._last_update_epoch is None:
            raise GuardViolation("尚无循环水测点，禁止调整")
        age = self.clock.timestamp() - float(self._last_update_epoch)
        if age > self.settings.cooling_measure_max_age_seconds:
            raise GuardViolation(
                "循环水测点过期，禁止调整",
                details={
                    "age_seconds": round(age, 3),
                    "max_age_seconds": self.settings.cooling_measure_max_age_seconds,
                },
            )

    def _interval_remaining(self) -> float:
        if self._last_adjust_epoch is None:
            return 0.0
        elapsed = max(0.0, self.clock.timestamp() - float(self._last_adjust_epoch))
        return round(max(0.0, self.settings.cooling_adjust_min_interval_seconds - elapsed), 3)

    def _apply_level(
        self,
        actor: str,
        new_level: int,
        kind: str,
        target: int,
        trace: Any,
    ) -> Mapping[str, Any]:
        previous = self._level
        self._level = new_level
        self._last_adjust_epoch = self.clock.timestamp()
        self._record_adjustment(
            actor,
            kind=kind,
            from_level=previous,
            to_level=new_level,
            reason=kind,
            target_level=target,
        )
        record = self._persist(reason=kind)
        evaluation = {
            "adjustment": kind,
            "from_level": previous,
            "to_level": new_level,
            "pumps_running": self.pumps_running,
            "fans_running": self.fans_running,
            "target_level": target,
        }
        trace.attach(record).note("adjustment", kind).note("from_level", previous).note(
            "to_level", new_level
        ).note("pumps_running", self.pumps_running).note("fans_running", self.fans_running)
        return self._evaluation_result(evaluation)

    def _record_adjustment(
        self,
        actor: str,
        *,
        kind: str,
        from_level: int,
        to_level: int,
        reason: str,
        target_level: int,
    ) -> None:
        entry = {
            "at": self.clock.timestamp_iso(),
            "actor": actor,
            "kind": kind,
            "from_level": from_level,
            "to_level": to_level,
            "pumps_running": self.pumps_running,
            "fans_running": self.fans_running,
            "inlet_temp_c": round(self._inlet_temp_c, 3),
            "pool_level": round(self._pool_level, 4),
            "target_level": target_level,
            "reason": reason,
        }
        self._last_adjustment = entry
        self.store.append(ADJUST_STREAM, entry)

    def _evaluation_result(self, evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
        result = dict(self.status())
        result["evaluation"] = dict(evaluation)
        return result

    def _hold_remaining(self) -> float:
        if self._latched_at is None:
            return 0.0
        elapsed = max(0.0, self.clock.timestamp() - float(self._latched_at))
        return max(0.0, self.settings.cooling_latch_min_hold_seconds - elapsed)

    def _persist(self, *, reason: str) -> Any:
        payload = {
            "state": self._machine.state,
            "reason": reason,
            "written_epoch": self.clock.timestamp(),
            "written_at": self.clock.timestamp_iso(),
            "level": self._level,
            "inlet_temp_c": round(self._inlet_temp_c, 3),
            "outlet_temp_c": round(self._outlet_temp_c, 3),
            "pool_level": round(self._pool_level, 4),
            "over_temperature": self._over_temperature,
            "low_level": self._low_level,
            "overflow": self._overflow,
            "latch_reason": self._latch_reason,
            "latched_at": self._latched_at,
            "latch_count": self._latch_count,
            "last_update_at": self._last_update_at,
            "last_update_epoch": self._last_update_epoch,
            "last_adjust_epoch": self._last_adjust_epoch,
            "last_adjustment": self._last_adjustment,
            "history": list(self._machine.history),
        }
        record = self.persist_state(payload)
        self._refresh_gauges()
        return record

    def _refresh_gauges(self) -> None:
        self.metrics.observe("cooling.level", float(self._level))
        self.metrics.observe("cooling.pumps_running", float(self.pumps_running))
        self.metrics.observe("cooling.fans_running", float(self.fans_running))
        self.metrics.observe("cooling.inlet_temp_c", round(self._inlet_temp_c, 3))
        self.metrics.observe("cooling.pool_level", round(self._pool_level, 4))
        self.metrics.observe("cooling.latch_count", float(self._latch_count))


__all__ = ["CoolingWaterSystem", "STATES", "TRANSITIONS", "ADJUST_STREAM"]
