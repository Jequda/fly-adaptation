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
)
from .geometry import clamp, distance, wrap_angle


@dataclass(frozen=True)
class MemoryFoodWorldConfig(FoodWorldConfig):
    """A cue phase followed by navigation with the food sensors switched off."""

    visible_steps: int = 8
    use_recurrence: bool = True

    def __post_init__(self) -> None:
        if self.visible_steps < 1:
            raise ValueError("visible_steps must be at least 1.")
        if self.visible_steps >= self.max_steps:
            raise ValueError("visible_steps must leave at least one hidden navigation step.")


class MemoryFoodWorld(FoodWorld):
    """A food task where only recurrent state can carry the target cue forward."""

    def __init__(self, config: MemoryFoodWorldConfig | None = None) -> None:
        self.config = config or MemoryFoodWorldConfig()

    def evaluate(
        self,
        genome: BrainGenome,
        rng: random.Random,
        record_trace: bool = False,
    ) -> EpisodeResult:
        # A stable start frame prevents body orientation from becoming an external memory.
        agent_x = 0.0
        agent_y = 0.0
        heading = 0.0
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
                    observation=self._observe(
                        agent_x,
                        agent_y,
                        heading,
                        food_x,
                        food_y,
                        food_visible=True,
                    ),
                    output=[0.0 for _ in range(self.output_size)],
                    hidden=state.hidden.copy(),
                    food_visible=True,
                    movement_enabled=False,
                )
            )

        for step in range(1, self.config.max_steps + 1):
            food_visible = step <= self.config.visible_steps
            movement_enabled = not food_visible
            observation = self._observe(
                agent_x,
                agent_y,
                heading,
                food_x,
                food_y,
                food_visible=food_visible,
            )
            input_state = state if self.config.use_recurrence else genome.initial_state()
            output, state = genome.activate(observation, input_state)

            if movement_enabled:
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
                        food_visible=food_visible,
                        movement_enabled=movement_enabled,
                    )
                )

            if movement_enabled and food_distance <= self.config.eat_radius:
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

    def _observe(
        self,
        agent_x: float,
        agent_y: float,
        heading: float,
        food_x: float,
        food_y: float,
        food_visible: bool,
    ) -> list[float]:
        if not food_visible:
            return [
                0.0,
                0.0,
                0.0,
                agent_x / (self.config.size / 2),
                agent_y / (self.config.size / 2),
                1.0,
            ]
        return super()._observe(agent_x, agent_y, heading, food_x, food_y)
