"""炉体/水套循环水组件。

循环泵与冷却塔风机的投运台数由进水温度分档决定：水温压不住时先加泵、泵
加满再加塔风机；温度回落时先退风机、再退泵。升/降档阈值之间留回差，避免
温度在阈值附近抖动时频繁投切。水池水位低于下限立即联锁切泵并保持闩锁，
必须由人工确认水位恢复、填写处理说明后显式复位。每一次台数调整都随状态
落盘并写入审计流，事后可以逐条对上。
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
    "circulating": ("latched", "idle"),
    "latched": ("circulating",),
}

# 进水温度测点量程（℃），超出即判为坏值，拒绝入库。
TEMP_RANGE_C = (0.0, 100.0)

# 调整记录留痕上限：超出后只保留最近若干条，与状态机历史口径一致。
ADJUSTMENT_LIMIT = 16


class CoolingLoop(Component):
    name = "cooling"

    def __init__(self, ctx: RuntimeContext) -> None:
        super().__init__(ctx)
        self._machine = StateMachine("cooling", "idle", TRANSITIONS, ctx.clock)
        self._inlet_temp_c = 0.0
        self._pool_level = 0.0
        self._stage = 0
        self._pumps_run = 0
        self._fans_run = 0
        self._latch_reason: str | None = None
        self._latched_at: float | None = None
        self._latch_count = 0
        self._adjustments: list[dict[str, Any]] = []
        self._last_update_at: str | None = None
        restored = self.restore()
        if restored is not None:
            self._machine.restore(restored)
            self._inlet_temp_c = float(restored.get("inlet_temp_c", 0.0))
            self._pool_level = float(restored.get("pool_level", 0.0))
            self._stage = int(restored.get("stage", 0))
            self._pumps_run = int(restored.get("pumps_run", 0))
            self._fans_run = int(restored.get("fans_run", 0))
            self._latch_reason = restored.get("latch_reason")
            self._latched_at = restored.get("latched_at")
            self._latch_count = int(restored.get("latch_count", 0))
            adjustments = restored.get("adjustments")
            if isinstance(adjustments, list):
                self._adjustments = [entry for entry in adjustments if isinstance(entry, dict)]
            self._last_update_at = restored.get("last_update_at")
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
            self._machine.require("idle", "循环水投运")
            self._validate_level(pool_level)
            if pool_level < self.settings.cooling_level_min:
                raise GuardViolation(
                    "水池水位低于投运下限，禁止启动循环泵",
                    details={"pool_level": pool_level, "min": self.settings.cooling_level_min},
                )
            self._pool_level = pool_level
            self._machine.to("circulating", actor, "循环水投运")
            self._apply_stage(actor, 0, "投运，先开一台循环泵")
            record = self._persist(reason="start")
            trace.attach(record).note("pool_level", pool_level)
            return self.status()

    def update(
        self,
        actor: str,
        *,
        inlet_temp_c: float,
        pool_level: float,
        correlation_id: str | None = None,
        expected_generation: int | None = None,
    ) -> Mapping[str, Any]:
        """扫描周期入口：录入测点 → 低水位联锁 → 按温度分档投切。"""

        actor = ensure_actor(actor)
        with self.action(
            "update",
            "cooling",
            actor,
            correlation_id=correlation_id,
            expected_generation=expected_generation,
        ) as trace:
            self._validate_temp(inlet_temp_c)
            self._validate_level(pool_level)
            self._inlet_temp_c = inlet_temp_c
            self._pool_level = pool_level
            self._last_update_at = self.clock.timestamp_iso()
            if self._machine.state == "circulating":
                if pool_level < self.settings.cooling_level_min:
                    self._trip_on_low_level(actor, pool_level)
                    record = self._persist(reason="latch")
                    trace.attach(record).note("latch_reason", self._latch_reason)
                    return self.status()
                target = self._target_stage(inlet_temp_c)
                if target != self._stage:
                    reason = "水温越上限，升档" if target > self._stage else "水温回差越下限，降档"
                    self._apply_stage(actor, target, reason)
                record = self._persist(reason="update")
                trace.attach(record).note("stage", self._stage)
                return self.status()
            # 未投运或已闩锁时只记录测点：闩锁状态下泵保持切除，等待人工复位。
            record = self._persist(reason="update")
            trace.attach(record).note("stage", self._stage)
            return self.status()

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
            self._machine.require("circulating", "循环水停运")
            self._machine.to("idle", actor, "循环水停运")
            self._apply_stage(actor, 0, "停运，切除全部泵与风机", cut=True)
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
            self._validate_level(pool_level)
            remaining = self._hold_remaining()
            if remaining > 0:
                raise GuardViolation(
                    "闩锁最短保持时长未到，禁止复位",
                    details={
                        "remaining_seconds": round(remaining, 3),
                        "min_hold_seconds": self.settings.cooling_latch_min_hold_seconds,
                    },
                )
            if pool_level < self.settings.cooling_level_min:
                raise GuardViolation(
                    "水池水位未恢复，禁止复位",
                    details={"pool_level": pool_level, "min": self.settings.cooling_level_min},
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
            self._machine.to("circulating", actor, f"闩锁复位：{note}")
            self._latch_reason = None
            self._latched_at = None
            self._apply_stage(actor, 0, "复位投运，先开一台循环泵")
            record = self._persist(reason="reset")
            trace.attach(record).note("note", note).note("intent_version", intent.version)
            return self.status()

    # ------------------------------------------------------------------ 查询
    @property
    def state(self) -> str:
        return self._machine.state

    def is_latched(self) -> bool:
        return self._machine.state == "latched"

    def max_stage(self) -> int:
        # 档位先排满泵（第 0 档已含一台泵），再排塔风机。
        return (self.settings.cooling_pump_total - 1) + self.settings.cooling_fan_total

    def hold_remaining(self) -> float:
        return self._hold_remaining()

    def status(self) -> Mapping[str, Any]:
        return {
            "state": self._machine.state,
            "inlet_temp_c": round(self._inlet_temp_c, 3),
            "pool_level": round(self._pool_level, 4),
            "pool_level_min": self.settings.cooling_level_min,
            "stage": self._stage,
            "max_stage": self.max_stage(),
            "pumps_run": self._pumps_run,
            "pump_total": self.settings.cooling_pump_total,
            "fans_run": self._fans_run,
            "fan_total": self.settings.cooling_fan_total,
            "temp_high_c": self.settings.cooling_temp_high_c,
            "temp_low_c": self.settings.cooling_temp_low_c,
            "temp_step_c": self.settings.cooling_temp_step_c,
            "latch_reason": self._latch_reason,
            "latched_at": self._latched_at,
            "latch_count": self._latch_count,
            "hold_remaining_seconds": round(self._hold_remaining(), 3),
            "adjustments": list(self._adjustments),
            "last_update_at": self._last_update_at,
            "history": list(self._machine.history),
        }

    # ------------------------------------------------------------------ 内部
    def _counts_for_stage(self, stage: int) -> tuple[int, int]:
        """档位 → 投运台数：先加泵，泵加满后再加塔风机。"""

        pump_total = self.settings.cooling_pump_total
        if stage <= pump_total - 1:
            return 1 + stage, 0
        return pump_total, stage - (pump_total - 1)

    def _target_stage(self, inlet_temp_c: float) -> int:
        """按进水温度计算目标档位，升/降档阈值之间留回差防抖。

        升到第 s 档要求温度高于 ``high + (s-1)*step``；从第 s 档退出要求温度
        低于 ``low + (s-1)*step``。两条阈值之间的温度维持当前档位不变。
        """

        stage = self._stage
        high = self.settings.cooling_temp_high_c
        low = self.settings.cooling_temp_low_c
        step = self.settings.cooling_temp_step_c
        while stage < self.max_stage() and inlet_temp_c > high + stage * step:
            stage += 1
        while stage > 0 and inlet_temp_c < low + (stage - 1) * step:
            stage -= 1
        return stage

    def _apply_stage(self, actor: str, target: int, reason: str, *, cut: bool = False) -> None:
        """落到目标档位并留痕；``cut=True`` 表示联锁/停运，泵与风机全部切除。"""

        pumps, fans = (0, 0) if cut else self._counts_for_stage(target)
        previous = self._stage
        self._stage = target
        self._pumps_run = pumps
        self._fans_run = fans
        self._adjustments.append(
            {
                "at": self.clock.timestamp_iso(),
                "actor": actor,
                "reason": reason,
                "from_stage": previous,
                "to_stage": target,
                "pumps_run": pumps,
                "fans_run": fans,
                "inlet_temp_c": round(self._inlet_temp_c, 3),
                "pool_level": round(self._pool_level, 4),
            }
        )
        self._adjustments = self._adjustments[-ADJUSTMENT_LIMIT:]

    def _trip_on_low_level(self, actor: str, pool_level: float) -> None:
        reason = "pool-level-low"
        self.write_intent(
            "latch",
            {
                "action": "latch",
                "reason": reason,
                "detail": {"pool_level": pool_level, "min": self.settings.cooling_level_min},
                "at": self.clock.timestamp_iso(),
                "actor": actor,
            },
        )
        self._machine.to("latched", actor, "水池水位低，联锁切除全部循环泵")
        self._latch_reason = reason
        self._latched_at = self.clock.timestamp()
        self._latch_count += 1
        self._apply_stage(actor, 0, "水位低联锁，切除全部泵与风机", cut=True)

    def _hold_remaining(self) -> float:
        if self._latched_at is None:
            return 0.0
        elapsed = max(0.0, self.clock.timestamp() - float(self._latched_at))
        return max(0.0, self.settings.cooling_latch_min_hold_seconds - elapsed)

    def _validate_temp(self, inlet_temp_c: float) -> None:
        low, high = TEMP_RANGE_C
        if not low < inlet_temp_c <= high:
            raise GuardViolation(
                "进水温度超出测点量程",
                details={"inlet_temp_c": inlet_temp_c, "range": "(0,100]"},
            )

    def _validate_level(self, pool_level: float) -> None:
        if not 0.0 <= pool_level <= 1.0:
            raise GuardViolation(
                "水池水位超出量程",
                details={"pool_level": pool_level, "range": "[0,1]"},
            )

    def _persist(self, *, reason: str) -> Any:
        payload = {
            "state": self._machine.state,
            "reason": reason,
            "written_epoch": self.clock.timestamp(),
            "written_at": self.clock.timestamp_iso(),
            "inlet_temp_c": round(self._inlet_temp_c, 3),
            "pool_level": round(self._pool_level, 4),
            "stage": self._stage,
            "pumps_run": self._pumps_run,
            "fans_run": self._fans_run,
            "latch_reason": self._latch_reason,
            "latched_at": self._latched_at,
            "latch_count": self._latch_count,
            "adjustments": list(self._adjustments),
            "last_update_at": self._last_update_at,
            "history": list(self._machine.history),
        }
        record = self.persist_state(payload)
        self._refresh_gauges()
        return record

    def _refresh_gauges(self) -> None:
        self.metrics.observe("cooling.inlet_temp_c", round(self._inlet_temp_c, 3))
        self.metrics.observe("cooling.pool_level", round(self._pool_level, 4))
        self.metrics.observe("cooling.stage", float(self._stage))
        self.metrics.observe("cooling.pumps_run", float(self._pumps_run))
        self.metrics.observe("cooling.fans_run", float(self._fans_run))
        self.metrics.observe("cooling.latch_count", float(self._latch_count))


__all__ = ["CoolingLoop", "STATES", "TRANSITIONS"]
