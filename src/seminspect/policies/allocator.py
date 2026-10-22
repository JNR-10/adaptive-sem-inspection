"""Deterministic sequential allocation under cumulative acquisition costs."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class CostSchedule:
    name: str
    cumulative_costs: tuple[float, float, float, float]

    @classmethod
    def from_values(cls, name: str, values: Iterable[float]) -> "CostSchedule":
        costs = tuple(float(item) for item in values)
        if len(costs) != 4 or any(next_cost <= current for current, next_cost in zip(costs, costs[1:])):
            raise ValueError("A cost schedule needs four strictly increasing cumulative state costs")
        return cls(name=name, cumulative_costs=costs)  # type: ignore[arg-type]

    @property
    def q0_cost(self) -> float:
        return self.cumulative_costs[0]

    @property
    def q3_cost(self) -> float:
        return self.cumulative_costs[3]

    def cumulative(self, quality: int) -> float:
        if quality not in {0, 1, 2, 3}:
            raise ValueError("Quality index must be in 0..3")
        return self.cumulative_costs[quality]

    def incremental(self, quality: int) -> float:
        if quality not in {0, 1, 2}:
            raise ValueError("Only Q0–Q2 have an upgrade action")
        return self.cumulative_costs[quality + 1] - self.cumulative_costs[quality]

    def validate_budget(self, budget: float) -> None:
        if float(budget) < self.q0_cost:
            raise ValueError(f"Budget {budget} is infeasible: Q0 survey cost is {self.q0_cost}")


@dataclass(frozen=True)
class Observation:
    """One current-quality observation. Labels and masks are deliberately absent."""

    image_id: str
    quality: int
    calibrated_probability: float
    detector_raw_score: float
    predicted_gain: float | None = None
    oracle_gain: float | None = None


class ObservationTable:
    """Immutable complete Q0–Q3 observations supplied equally to every policy."""

    def __init__(self, observations: Iterable[Observation]) -> None:
        table = {(item.image_id, item.quality): item for item in observations}
        if not table:
            raise ValueError("Observation table cannot be empty")
        image_ids = sorted({item.image_id for item in table.values()})
        for image_id in image_ids:
            if {(image_id, quality) for quality in range(4)} - set(table):
                raise ValueError(f"Every policy image needs all Q0–Q3 observations: {image_id}")
        self._table = table
        self.image_ids = tuple(image_ids)

    def get(self, image_id: str, quality: int) -> Observation:
        return self._table[(image_id, quality)]


Signal = Callable[[Observation, Observation | None], float]


@dataclass(frozen=True)
class Policy:
    name: str
    signal: Signal
    stop_on_nonpositive: bool = False
    non_deployable: bool = False


@dataclass
class RegionState:
    quality: int = 0
    version: int = 0
    history: list[dict[str, float | int | str | None]] = field(default_factory=list)


@dataclass
class AllocationResult:
    policy: str
    budget: float
    actual_cost: float
    final_qualities: dict[str, int]
    final_scores: dict[str, float]
    trajectories: dict[str, list[dict[str, float | int | str | None]]]

    @property
    def q_share(self) -> dict[str, float]:
        total = len(self.final_qualities)
        return {f"Q{quality}": sum(value == quality for value in self.final_qualities.values()) / total for quality in range(4)}

    @property
    def mean_upgrades(self) -> float:
        return float(np.mean(list(self.final_qualities.values())))


def entropy(probability: float) -> float:
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(-p * np.log(p) - (1.0 - p) * np.log(1.0 - p))


def _initial_states(table: ObservationTable) -> dict[str, RegionState]:
    states: dict[str, RegionState] = {}
    for image_id in table.image_ids:
        initial = table.get(image_id, 0)
        states[image_id] = RegionState(
            history=[
                {
                    "quality": 0,
                    "calibrated_probability": initial.calibrated_probability,
                    "predicted_gain": initial.predicted_gain,
                    "priority": None,
                    "action": "survey",
                }
            ]
        )
    return states


def _signal(policy: Policy, table: ObservationTable, states: Mapping[str, RegionState], image_id: str) -> float:
    state = states[image_id]
    current = table.get(image_id, state.quality)
    previous = table.get(image_id, state.quality - 1) if state.quality > 0 else None
    value = float(policy.signal(current, previous))
    if not np.isfinite(value):
        raise ValueError(f"Policy {policy.name} returned non-finite signal for {image_id}")
    return value


def _priority(policy: Policy, signal: float, incremental_cost: float) -> float:
    return max(0.0, signal) / incremental_cost


def allocate_deterministic(table: ObservationTable, policy: Policy, schedule: CostSchedule, budget: float) -> AllocationResult:
    """Use the locked heap / temporary-skipped-entry allocation algorithm."""

    schedule.validate_budget(budget)
    total = len(table.image_ids)
    remaining = np.float64(total * float(budget) - total * schedule.q0_cost)
    states = _initial_states(table)
    heap: list[tuple[float, str, int]] = []

    def push(image_id: str) -> None:
        state = states[image_id]
        if state.quality >= 3:
            return
        signal = _signal(policy, table, states, image_id)
        priority = _priority(policy, signal, schedule.incremental(state.quality))
        # Min-heap key encodes descending priority then ascending image_id.
        heapq.heappush(heap, (-priority, image_id, state.version))

    for image_id in table.image_ids:
        push(image_id)

    while heap:
        skipped: list[tuple[float, str, int]] = []
        selected: tuple[float, str, int, float, float] | None = None
        while heap:
            negative_priority, image_id, version = heapq.heappop(heap)
            state = states[image_id]
            if version != state.version or state.quality >= 3:
                continue  # stale state after a previous action
            signal = _signal(policy, table, states, image_id)
            priority = _priority(policy, signal, schedule.incremental(state.quality))
            # Current state was unchanged since push, so priority must match except a numerical no-op.
            if abs((-negative_priority) - priority) > EPSILON:
                heapq.heappush(heap, (-priority, image_id, state.version))
                continue
            if policy.stop_on_nonpositive and signal <= 0.0:
                # Heap is ordered by nonnegative priority, so no positive action remains.
                heap = []
                break
            incremental = schedule.incremental(state.quality)
            if incremental <= remaining + EPSILON:
                selected = (priority, image_id, version, signal, incremental)
                break
            skipped.append((negative_priority, image_id, version))
        if selected is None:
            break
        for item in skipped:
            heapq.heappush(heap, item)
        priority, image_id, _, signal, incremental = selected
        state = states[image_id]
        before = state.quality
        state.history[-1]["priority"] = priority
        state.history[-1]["signal"] = signal
        state.quality += 1
        state.version += 1
        remaining -= np.float64(incremental)
        observation = table.get(image_id, state.quality)
        state.history.append(
            {
                "quality": state.quality,
                "calibrated_probability": observation.calibrated_probability,
                "predicted_gain": observation.predicted_gain,
                "priority": None,
                "action": f"Q{before}->Q{state.quality}",
            }
        )
        push(image_id)

    final_qualities = {image_id: state.quality for image_id, state in states.items()}
    actual_cost = float(np.mean([schedule.cumulative(quality) for quality in final_qualities.values()]))
    if actual_cost > float(budget) + EPSILON:
        raise RuntimeError(f"Allocator exceeded budget: {actual_cost} > {budget}")
    return AllocationResult(
        policy=policy.name,
        budget=float(budget),
        actual_cost=actual_cost,
        final_qualities=final_qualities,
        final_scores={image_id: table.get(image_id, quality).calibrated_probability for image_id, quality in final_qualities.items()},
        trajectories={image_id: state.history for image_id, state in states.items()},
    )


def allocate_random(table: ObservationTable, schedule: CostSchedule, budget: float, seed: int) -> AllocationResult:
    """Uniformly choose among currently affordable actions in protocol-specified order."""

    schedule.validate_budget(budget)
    total = len(table.image_ids)
    remaining = np.float64(total * float(budget) - total * schedule.q0_cost)
    states = _initial_states(table)
    rng = np.random.Generator(np.random.PCG64(seed))
    while True:
        eligible = [
            (image_id, state.quality)
            for image_id, state in states.items()
            if state.quality < 3 and schedule.incremental(state.quality) <= remaining + EPSILON
        ]
        if not eligible:
            break
        eligible.sort(key=lambda item: (item[0], item[1]))
        image_id, quality = eligible[int(rng.integers(0, len(eligible)))]
        state = states[image_id]
        increment = schedule.incremental(quality)
        state.history[-1]["priority"] = None
        state.history[-1]["signal"] = None
        state.quality += 1
        state.version += 1
        remaining -= np.float64(increment)
        observation = table.get(image_id, state.quality)
        state.history.append(
            {
                "quality": state.quality,
                "calibrated_probability": observation.calibrated_probability,
                "predicted_gain": observation.predicted_gain,
                "priority": None,
                "action": f"Q{quality}->Q{state.quality}",
            }
        )
    final_qualities = {image_id: state.quality for image_id, state in states.items()}
    actual_cost = float(np.mean([schedule.cumulative(quality) for quality in final_qualities.values()]))
    if actual_cost > budget + EPSILON:
        raise RuntimeError("Random allocator exceeded cost cap")
    return AllocationResult(
        policy="random",
        budget=float(budget),
        actual_cost=actual_cost,
        final_qualities=final_qualities,
        final_scores={image_id: table.get(image_id, quality).calibrated_probability for image_id, quality in final_qualities.items()},
        trajectories={image_id: state.history for image_id, state in states.items()},
    )


def survey_only(table: ObservationTable, schedule: CostSchedule, budget: float) -> AllocationResult:
    schedule.validate_budget(budget)
    states = _initial_states(table)
    actual = schedule.q0_cost
    return AllocationResult(
        policy="survey_only",
        budget=float(budget),
        actual_cost=actual,
        final_qualities={image_id: 0 for image_id in table.image_ids},
        final_scores={image_id: table.get(image_id, 0).calibrated_probability for image_id in table.image_ids},
        trajectories={image_id: state.history for image_id, state in states.items()},
    )


def full_q3(table: ObservationTable, schedule: CostSchedule, budget: float | None = None) -> AllocationResult:
    cap = schedule.q3_cost if budget is None else float(budget)
    if cap + EPSILON < schedule.q3_cost:
        raise ValueError("Full-Q3 reference requires a cap at least equal to Q3 cumulative cost")
    trajectories: dict[str, list[dict[str, float | int | str | None]]] = {}
    for image_id in table.image_ids:
        trajectories[image_id] = [
            {
                "quality": quality,
                "calibrated_probability": table.get(image_id, quality).calibrated_probability,
                "predicted_gain": table.get(image_id, quality).predicted_gain,
                "priority": None,
                "action": "full_q3" if quality else "survey",
            }
            for quality in range(4)
        ]
    return AllocationResult(
        policy="full_q3",
        budget=cap,
        actual_cost=schedule.q3_cost,
        final_qualities={image_id: 3 for image_id in table.image_ids},
        final_scores={image_id: table.get(image_id, 3).calibrated_probability for image_id in table.image_ids},
        trajectories=trajectories,
    )
