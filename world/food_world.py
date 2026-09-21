from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean

from brain import BrainGenome

from .geometry import clamp, distance, wrap_angle


@dataclass(frozen=True)
class FoodWorldConfig:
    size: float = 10.0
    max_steps: int = 120
    eat_radius: float = 0.45
    move_speed: float = 0.34
    turn_speed: float = 0.55
    time_penalty: float = 0.01
    approach_reward: float = 5.0
    eat_reward: float = 100.0


@dataclass
class StepTrace:
    step: int
    agent_x: float
    agent_y: float
    heading: float
    food_x: float
    food_y: float
    distance: float
    score: float
    observation: list[float]
    output: list[float]
    hidden: list[float]
    food_visible: bool = True
    movement_enabled: bool = True
    collided: bool = False
    cue_visible: bool = False
    cue_target: str | None = None


@dataclass(frozen=True)
class WorldObstacle:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class WorldTarget:
    label: str
    x: float
    y: float
    is_correct: bool


@dataclass
class EpisodeResult:
    score: float
    ate_food: bool
    steps: int
    final_distance: float
    trace: list[StepTrace] | None = None
    obstacles: list[WorldObstacle] | None = None
    collisions: int = 0
    targets: list[WorldTarget] | None = None
    choice: str | None = None
    chose_correct: bool | None = None
    target_preference: float | None = None


class FoodWorld:
    input_size = 6
    output_size = 3

    def __init__(self, config: FoodWorldConfig | None = None) -> None:
        self.config = config or FoodWorldConfig()

    def aggregate_fitness(self, results: list[EpisodeResult]) -> float:
        """Combine episode scores into the value used for evolutionary selection."""
        if not results:
            raise ValueError("Fitness requires at least one episode result.")
        return mean(result.score for result in results)

    def evaluate(
        self,
        genome: BrainGenome,
        rng: random.Random,
        record_trace: bool = False,
    ) -> EpisodeResult:
        agent_x = rng.uniform(-self.config.size * 0.35, self.config.size * 0.35)
        agent_y = rng.uniform(-self.config.size * 0.35, self.config.size * 0.35)
        heading = rng.uniform(-math.pi, math.pi)
        food_x, food_y = self._place_food(rng, agent_x, agent_y)

        state = genome.initial_state()
        score = 0.0
        previous_distance = distance(agent_x, agent_y, food_x, food_y)
        trace: list[StepTrace] | None = [] if record_trace else None

        if trace is not None:
            trace.append(
                StepTrace(
                    step=0,
                    agent_x=agent_x,
                    agent_y=agent_y,
                    heading=heading,
                    food_x=food_x,
                    food_y=food_y,
                    distance=previous_distance,
                    score=score,
                    observation=self._observe(agent_x, agent_y, heading, food_x, food_y),
                    output=[0.0 for _ in range(self.output_size)],
                    hidden=state.hidden.copy(),
                )
            )

        for step in range(1, self.config.max_steps + 1):
            observation = self._observe(agent_x, agent_y, heading, food_x, food_y)
            output, state = genome.activate(observation, state)

            turn_signal = output[1] - output[0]
            move_signal = max(0.0, output[2])
            heading = wrap_angle(heading + turn_signal * self.config.turn_speed)

            agent_x += math.cos(heading) * move_signal * self.config.move_speed
            agent_y += math.sin(heading) * move_signal * self.config.move_speed
            agent_x = clamp(agent_x, -self.config.size / 2, self.config.size / 2)
            agent_y = clamp(agent_y, -self.config.size / 2, self.config.size / 2)

            food_distance = distance(agent_x, agent_y, food_x, food_y)
            score += (previous_distance - food_distance) * self.config.approach_reward
            score -= self.config.time_penalty
            previous_distance = food_distance

            if trace is not None:
                trace.append(
                    StepTrace(
                        step=step,
                        agent_x=agent_x,
                        agent_y=agent_y,
                        heading=heading,
                        food_x=food_x,
                        food_y=food_y,
                        distance=food_distance,
                        score=score,
                        observation=observation,
                        output=output,
                        hidden=state.hidden.copy(),
                    )
                )

            if food_distance <= self.config.eat_radius:
                score += self.config.eat_reward
                if trace is not None:
                    trace[-1].score = score
                return EpisodeResult(
                    score=score,
                    ate_food=True,
                    steps=step,
                    final_distance=food_distance,
                    trace=trace,
                )

        return EpisodeResult(
            score=score,
            ate_food=False,
            steps=self.config.max_steps,
            final_distance=previous_distance,
            trace=trace,
        )

    def _place_food(self, rng: random.Random, agent_x: float, agent_y: float) -> tuple[float, float]:
        for _ in range(100):
            food_x = rng.uniform(-self.config.size * 0.45, self.config.size * 0.45)
            food_y = rng.uniform(-self.config.size * 0.45, self.config.size * 0.45)
            if distance(agent_x, agent_y, food_x, food_y) >= self.config.size * 0.25:
                return food_x, food_y
        return -agent_x, -agent_y

    def _observe(
        self,
        agent_x: float,
        agent_y: float,
        heading: float,
        food_x: float,
        food_y: float,
    ) -> list[float]:
        dx = food_x - agent_x
        dy = food_y - agent_y
        distance = math.hypot(dx, dy)
        angle_to_food = math.atan2(dy, dx)
        relative_angle = wrap_angle(angle_to_food - heading)
        normalized_distance = min(1.0, distance / self.config.size)

        return [
            math.sin(relative_angle),
            math.cos(relative_angle),
            normalized_distance,
            agent_x / (self.config.size / 2),
            agent_y / (self.config.size / 2),
            1.0,
        ]
