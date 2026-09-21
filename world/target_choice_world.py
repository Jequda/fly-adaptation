from __future__ import annotations

from dataclasses import dataclass
import math
import random

from brain import BrainGenome

from .food_world import (
    EpisodeResult,
    FoodWorld,
    FoodWorldConfig,
    StepTrace,
    WorldTarget,
)
from .geometry import clamp, distance, wrap_angle


TARGET_SECTORS = (
    "front-upper",
    "back-upper",
    "back-lower",
    "front-lower",
)
WORST_CATEGORY_FITNESS_WEIGHT = 10.0
WORST_CATEGORY_SCORE_WEIGHT = 1.0
PAIR_CONTRAST_FITNESS_WEIGHT = 0.5
PAIR_ALIGNMENT_FITNESS_WEIGHT = 2.0
TARGET_SCENARIO_CODE_COUNT = 1_000_000


def target_sector_for_position(x: float, y: float) -> str:
    if x >= 0 and y >= 0:
        return "front-upper"
    if x < 0 <= y:
        return "back-upper"
    if x < 0 and y < 0:
        return "back-lower"
    return "front-lower"


@dataclass(frozen=True)
class TargetChoiceWorldConfig(FoodWorldConfig):
    """Two visible targets and a transient cue indicating the rewarded one."""

    max_steps: int = 80
    cue_steps: int = 6
    wrong_target_penalty: float = 25.0
    min_target_radius: float = 2.8
    max_target_radius: float = 4.0
    use_recurrence: bool = True

    def __post_init__(self) -> None:
        if self.cue_steps < 1:
            raise ValueError("cue_steps must be at least 1.")
        if self.cue_steps >= self.max_steps:
            raise ValueError("cue_steps must leave at least one movement step.")
        if self.wrong_target_penalty < 0:
            raise ValueError("wrong_target_penalty cannot be negative.")
        if self.min_target_radius <= self.eat_radius:
            raise ValueError("min_target_radius must exceed eat_radius.")
        if self.max_target_radius < self.min_target_radius:
            raise ValueError("max_target_radius must not be smaller than minimum.")
        if self.max_target_radius >= self.size / 2:
            raise ValueError("Targets must stay inside the world boundary.")


@dataclass(frozen=True)
class TargetChoiceSignature:
    correct_target: str
    target_a_sector: str

    @property
    def target_a_side(self) -> str:
        return "positive" if self.target_a_sector.endswith("upper") else "negative"


@dataclass(frozen=True)
class _TargetChoiceScenario:
    target_a_x: float
    target_a_y: float
    target_b_x: float
    target_b_y: float
    correct_target: str


class TargetChoiceWorld(FoodWorld):
    """Remember a transient rule and approach the matching visible target."""

    input_size = 9

    def __init__(self, config: TargetChoiceWorldConfig | None = None) -> None:
        self.config = config or TargetChoiceWorldConfig()

    def scenario_signature_for_seed(self, seed: int) -> TargetChoiceSignature:
        scenario = self._make_scenario(random.Random(seed))
        return TargetChoiceSignature(
            correct_target=scenario.correct_target,
            target_a_sector=target_sector_for_position(
                scenario.target_a_x,
                scenario.target_a_y,
            ),
        )

    def scenario_pair_key_for_seed(self, seed: int) -> int:
        return self._scenario_code(random.Random(seed)) // 2

    def aggregate_fitness(self, results: list[EpisodeResult]) -> float:
        """Reward overall skill while preventing one-cue specialization."""
        if not results:
            raise ValueError("Fitness requires at least one episode result.")

        cue_successes: dict[str, list[bool]] = {"A": [], "B": []}
        category_successes = {
            (cue, sector): []
            for cue in ("A", "B")
            for sector in TARGET_SECTORS
        }
        category_scores = {category: [] for category in category_successes}
        for result in results:
            correct_targets = [
                target.label
                for target in (result.targets or [])
                if target.is_correct
            ]
            if len(correct_targets) != 1 or correct_targets[0] not in cue_successes:
                raise ValueError("Target-choice result must identify one correct target.")
            target_a = next(
                (target for target in (result.targets or []) if target.label == "A"),
                None,
            )
            if target_a is None:
                raise ValueError("Target-choice result must include target A.")
            correct_target = correct_targets[0]
            sector = target_sector_for_position(target_a.x, target_a.y)
            cue_successes[correct_target].append(result.ate_food)
            category_successes[(correct_target, sector)].append(result.ate_food)
            category_scores[(correct_target, sector)].append(result.score)

        if not all(cue_successes.values()):
            raise ValueError("Target-choice fitness requires both cue A and cue B episodes.")
        if not all(category_successes.values()):
            raise ValueError(
                "Target-choice fitness requires every cue and target-sector category."
            )
        if len(results) % 2:
            raise ValueError("Target-choice fitness requires paired scenarios.")

        pair_contrasts = []
        pair_alignments = []
        for first, second in zip(results[::2], results[1::2]):
            first_targets = {target.label: target for target in first.targets or []}
            second_targets = {target.label: target for target in second.targets or []}
            if first_targets.keys() != {"A", "B"} or second_targets.keys() != {
                "A",
                "B",
            }:
                raise ValueError("Each target-choice pair must contain targets A and B.")
            same_geometry = all(
                (first_targets[label].x, first_targets[label].y)
                == (second_targets[label].x, second_targets[label].y)
                for label in ("A", "B")
            )
            correct_labels = {
                target.label
                for result in (first, second)
                for target in result.targets or []
                if target.is_correct
            }
            if not same_geometry or correct_labels != {"A", "B"}:
                raise ValueError(
                    "Each target-choice pair must reuse one geometry with opposite cues."
                )
            pair_contrasts.append({first.choice, second.choice} == {"A", "B"})
            by_cue = {
                next(
                    target.label
                    for target in result.targets or []
                    if target.is_correct
                ): result
                for result in (first, second)
            }
            if any(result.target_preference is None for result in by_cue.values()):
                raise ValueError(
                    "Target-choice results must include continuous target preference."
                )
            pair_alignments.append(
                (
                    float(by_cue["A"].target_preference)
                    - float(by_cue["B"].target_preference)
                )
                / 2.0
            )

        cue_rates = [
            sum(successes) / len(successes)
            for successes in cue_successes.values()
        ]
        overall_rate = sum(result.ate_food for result in results) / len(results)
        weakest_cue_rate = min(cue_rates)
        weakest_category_rate = min(
            sum(successes) / len(successes)
            for successes in category_successes.values()
        )
        weakest_category_score = min(
            sum(scores) / len(scores) for scores in category_scores.values()
        )
        pair_contrast_rate = sum(pair_contrasts) / len(pair_contrasts)
        pair_alignment = sum(pair_alignments) / len(pair_alignments)
        behavior_score = super().aggregate_fitness(results)
        return (
            self.config.eat_reward
            * (
                overall_rate
                + weakest_cue_rate
                + WORST_CATEGORY_FITNESS_WEIGHT * weakest_category_rate
            )
            + behavior_score * 0.1
            + weakest_category_score * WORST_CATEGORY_SCORE_WEIGHT
            + self.config.eat_reward
            * PAIR_CONTRAST_FITNESS_WEIGHT
            * pair_contrast_rate
            + self.config.eat_reward
            * PAIR_ALIGNMENT_FITNESS_WEIGHT
            * pair_alignment
        )

    def evaluate(
        self,
        genome: BrainGenome,
        rng: random.Random,
        record_trace: bool = False,
    ) -> EpisodeResult:
        scenario = self._make_scenario(rng)
        correct_x, correct_y = self._correct_target_position(scenario)
        wrong_label = "B" if scenario.correct_target == "A" else "A"
        wrong_x, wrong_y = self._target_position(scenario, wrong_label)
        targets = [
            WorldTarget(
                label="A",
                x=scenario.target_a_x,
                y=scenario.target_a_y,
                is_correct=scenario.correct_target == "A",
            ),
            WorldTarget(
                label="B",
                x=scenario.target_b_x,
                y=scenario.target_b_y,
                is_correct=scenario.correct_target == "B",
            ),
        ]

        agent_x = 0.0
        agent_y = 0.0
        heading = 0.0
        state = genome.initial_state()
        score = 0.0
        correct_distance = distance(agent_x, agent_y, correct_x, correct_y)
        previous_distance = correct_distance
        trace: list[StepTrace] | None = [] if record_trace else None

        if trace is not None:
            trace.append(
                self._trace_step(
                    step=0,
                    agent_x=agent_x,
                    agent_y=agent_y,
                    heading=heading,
                    scenario=scenario,
                    correct_x=correct_x,
                    correct_y=correct_y,
                    correct_distance=correct_distance,
                    score=score,
                    observation=self._observe_choice(
                        agent_x,
                        agent_y,
                        heading,
                        scenario,
                        cue_visible=True,
                    ),
                    output=[0.0 for _ in range(self.output_size)],
                    hidden=state.hidden.copy(),
                    cue_visible=True,
                    movement_enabled=False,
                )
            )

        for step in range(1, self.config.max_steps + 1):
            cue_visible = step <= self.config.cue_steps
            movement_enabled = not cue_visible
            observation = self._observe_choice(
                agent_x,
                agent_y,
                heading,
                scenario,
                cue_visible=cue_visible,
            )
            input_state = state if self.config.use_recurrence else genome.initial_state()
            output, state = genome.activate(observation, input_state)

            if movement_enabled:
                turn_signal = output[1] - output[0]
                move_signal = max(0.0, output[2])
                heading = wrap_angle(heading + turn_signal * self.config.turn_speed)
                agent_x = clamp(
                    agent_x + math.cos(heading) * move_signal * self.config.move_speed,
                    -self.config.size / 2,
                    self.config.size / 2,
                )
                agent_y = clamp(
                    agent_y + math.sin(heading) * move_signal * self.config.move_speed,
                    -self.config.size / 2,
                    self.config.size / 2,
                )

            correct_distance = distance(agent_x, agent_y, correct_x, correct_y)
            if movement_enabled:
                score += (
                    previous_distance - correct_distance
                ) * self.config.approach_reward
            score -= self.config.time_penalty
            previous_distance = correct_distance

            if trace is not None:
                trace.append(
                    self._trace_step(
                        step=step,
                        agent_x=agent_x,
                        agent_y=agent_y,
                        heading=heading,
                        scenario=scenario,
                        correct_x=correct_x,
                        correct_y=correct_y,
                        correct_distance=correct_distance,
                        score=score,
                        observation=observation,
                        output=output,
                        hidden=state.hidden.copy(),
                        cue_visible=cue_visible,
                        movement_enabled=movement_enabled,
                    )
                )

            if not movement_enabled:
                continue
            if correct_distance <= self.config.eat_radius:
                score += self.config.eat_reward
                if trace is not None:
                    trace[-1].score = score
                return EpisodeResult(
                    score=score,
                    ate_food=True,
                    steps=step,
                    final_distance=correct_distance,
                    trace=trace,
                    targets=targets,
                    choice=scenario.correct_target,
                    chose_correct=True,
                    target_preference=self._target_preference(
                        agent_x,
                        agent_y,
                        scenario,
                    ),
                )
            if distance(agent_x, agent_y, wrong_x, wrong_y) <= self.config.eat_radius:
                score -= self.config.wrong_target_penalty
                if trace is not None:
                    trace[-1].score = score
                return EpisodeResult(
                    score=score,
                    ate_food=False,
                    steps=step,
                    final_distance=correct_distance,
                    trace=trace,
                    targets=targets,
                    choice=wrong_label,
                    chose_correct=False,
                    target_preference=self._target_preference(
                        agent_x,
                        agent_y,
                        scenario,
                    ),
                )

        return EpisodeResult(
            score=score,
            ate_food=False,
            steps=self.config.max_steps,
            final_distance=correct_distance,
            trace=trace,
            targets=targets,
            choice=None,
            chose_correct=None,
            target_preference=self._target_preference(
                agent_x,
                agent_y,
                scenario,
            ),
        )

    def _make_scenario(self, rng: random.Random) -> _TargetChoiceScenario:
        scenario_code = self._scenario_code(rng)
        geometry_rng = random.Random(scenario_code // 2)
        angle = geometry_rng.uniform(-math.pi, math.pi)
        radius = geometry_rng.uniform(
            self.config.min_target_radius,
            self.config.max_target_radius,
        )
        target_a_x = math.cos(angle) * radius
        target_a_y = math.sin(angle) * radius
        correct_target = "A" if scenario_code % 2 == 0 else "B"
        return _TargetChoiceScenario(
            target_a_x=target_a_x,
            target_a_y=target_a_y,
            target_b_x=-target_a_x,
            target_b_y=-target_a_y,
            correct_target=correct_target,
        )

    @staticmethod
    def _scenario_code(rng: random.Random) -> int:
        return rng.randrange(TARGET_SCENARIO_CODE_COUNT)

    def _observe_choice(
        self,
        agent_x: float,
        agent_y: float,
        heading: float,
        scenario: _TargetChoiceScenario,
        *,
        cue_visible: bool,
    ) -> list[float]:
        target_a = self._target_observation(
            agent_x,
            agent_y,
            heading,
            scenario.target_a_x,
            scenario.target_a_y,
        )
        target_b = self._target_observation(
            agent_x,
            agent_y,
            heading,
            scenario.target_b_x,
            scenario.target_b_y,
        )
        cue_value = (
            1.0
            if cue_visible and scenario.correct_target == "A"
            else -1.0 if cue_visible else 0.0
        )
        cue_is_visible = 1.0 if cue_visible else 0.0
        return target_a + target_b + [cue_value, cue_is_visible, 1.0]

    def _target_observation(
        self,
        agent_x: float,
        agent_y: float,
        heading: float,
        target_x: float,
        target_y: float,
    ) -> list[float]:
        dx = target_x - agent_x
        dy = target_y - agent_y
        target_distance = math.hypot(dx, dy)
        relative_angle = wrap_angle(math.atan2(dy, dx) - heading)
        return [
            math.sin(relative_angle),
            math.cos(relative_angle),
            min(1.0, target_distance / self.config.size),
        ]

    @staticmethod
    def _target_preference(
        agent_x: float,
        agent_y: float,
        scenario: _TargetChoiceScenario,
    ) -> float:
        distance_a = distance(
            agent_x,
            agent_y,
            scenario.target_a_x,
            scenario.target_a_y,
        )
        distance_b = distance(
            agent_x,
            agent_y,
            scenario.target_b_x,
            scenario.target_b_y,
        )
        return (distance_b - distance_a) / (distance_a + distance_b)

    def _trace_step(
        self,
        *,
        step: int,
        agent_x: float,
        agent_y: float,
        heading: float,
        scenario: _TargetChoiceScenario,
        correct_x: float,
        correct_y: float,
        correct_distance: float,
        score: float,
        observation: list[float],
        output: list[float],
        hidden: list[float],
        cue_visible: bool,
        movement_enabled: bool,
    ) -> StepTrace:
        return StepTrace(
            step=step,
            agent_x=agent_x,
            agent_y=agent_y,
            heading=heading,
            food_x=correct_x,
            food_y=correct_y,
            distance=correct_distance,
            score=score,
            observation=observation,
            output=output,
            hidden=hidden,
            food_visible=True,
            movement_enabled=movement_enabled,
            cue_visible=cue_visible,
            cue_target=scenario.correct_target,
        )

    @staticmethod
    def _target_position(
        scenario: _TargetChoiceScenario,
        label: str,
    ) -> tuple[float, float]:
        if label == "A":
            return scenario.target_a_x, scenario.target_a_y
        return scenario.target_b_x, scenario.target_b_y

    def _correct_target_position(
        self,
        scenario: _TargetChoiceScenario,
    ) -> tuple[float, float]:
        return self._target_position(scenario, scenario.correct_target)
