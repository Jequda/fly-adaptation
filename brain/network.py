from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Any


def _rand_matrix(rng: random.Random, rows: int, cols: int, scale: float) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _rand_vector(rng: random.Random, size: int, scale: float) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(size)]


def _dot(weights: list[float], values: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, values))


@dataclass
class BrainState:
    hidden: list[float]

    @classmethod
    def zeros(cls, hidden_size: int) -> BrainState:
        return cls(hidden=[0.0 for _ in range(hidden_size)])


@dataclass
class BrainGenome:
    input_size: int
    hidden_size: int
    output_size: int
    input_hidden: list[list[float]]
    recurrent_hidden: list[list[float]]
    hidden_output: list[list[float]]
    hidden_bias: list[float]
    output_bias: list[float]

    @classmethod
    def random(
        cls,
        rng: random.Random,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_scale: float = 1.0,
    ) -> BrainGenome:
        recurrent_scale = weight_scale / math.sqrt(max(hidden_size, 1))
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            input_hidden=_rand_matrix(rng, hidden_size, input_size, weight_scale),
            recurrent_hidden=_rand_matrix(rng, hidden_size, hidden_size, recurrent_scale),
            hidden_output=_rand_matrix(rng, output_size, hidden_size, weight_scale),
            hidden_bias=_rand_vector(rng, hidden_size, weight_scale),
            output_bias=_rand_vector(rng, output_size, weight_scale),
        )

    def initial_state(self) -> BrainState:
        return BrainState.zeros(self.hidden_size)

    def activate(self, observation: list[float], state: BrainState) -> tuple[list[float], BrainState]:
        if len(observation) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(observation)}.")
        if len(state.hidden) != self.hidden_size:
            raise ValueError(f"Expected hidden state {self.hidden_size}, got {len(state.hidden)}.")

        next_hidden: list[float] = []
        for row, recurrent_row, bias in zip(
            self.input_hidden,
            self.recurrent_hidden,
            self.hidden_bias,
        ):
            activation = bias + _dot(row, observation) + _dot(recurrent_row, state.hidden)
            next_hidden.append(math.tanh(activation))

        output: list[float] = []
        for row, bias in zip(self.hidden_output, self.output_bias):
            output.append(math.tanh(bias + _dot(row, next_hidden)))

        return output, BrainState(hidden=next_hidden)

    def clone(self) -> BrainGenome:
        return BrainGenome.from_dict(self.to_dict())

    def mutated(
        self,
        rng: random.Random,
        weight_mutation_rate: float,
        weight_mutation_power: float,
        structural_mutation_rate: float,
        min_hidden_size: int,
        max_hidden_size: int,
    ) -> BrainGenome:
        child = self.clone()
        child._mutate_weights(rng, weight_mutation_rate, weight_mutation_power)

        if rng.random() < structural_mutation_rate:
            can_add = child.hidden_size < max_hidden_size
            can_remove = child.hidden_size > min_hidden_size
            if can_add and (not can_remove or rng.random() < 0.65):
                child._add_hidden_neuron(rng)
            elif can_remove:
                child._remove_hidden_neuron(rng.randrange(child.hidden_size))

        return child

    def _mutate_weights(
        self,
        rng: random.Random,
        mutation_rate: float,
        mutation_power: float,
    ) -> None:
        def mutate_value(value: float) -> float:
            if rng.random() >= mutation_rate:
                return value
            return _clamp(value + rng.gauss(0.0, mutation_power), -5.0, 5.0)

        self.input_hidden = [
            [mutate_value(value) for value in row] for row in self.input_hidden
        ]
        self.recurrent_hidden = [
            [mutate_value(value) for value in row] for row in self.recurrent_hidden
        ]
        self.hidden_output = [
            [mutate_value(value) for value in row] for row in self.hidden_output
        ]
        self.hidden_bias = [mutate_value(value) for value in self.hidden_bias]
        self.output_bias = [mutate_value(value) for value in self.output_bias]

    def _add_hidden_neuron(self, rng: random.Random) -> None:
        scale = 0.5
        self.hidden_size += 1
        self.input_hidden.append(_rand_vector(rng, self.input_size, scale))
        self.hidden_bias.append(rng.uniform(-scale, scale))

        for row in self.recurrent_hidden:
            row.append(rng.uniform(-scale, scale))
        self.recurrent_hidden.append(_rand_vector(rng, self.hidden_size, scale))

        for row in self.hidden_output:
            row.append(rng.uniform(-scale, scale))

    def _remove_hidden_neuron(self, index: int) -> None:
        if self.hidden_size <= 1:
            return

        self.hidden_size -= 1
        del self.input_hidden[index]
        del self.hidden_bias[index]
        del self.recurrent_hidden[index]

        for row in self.recurrent_hidden:
            del row[index]
        for row in self.hidden_output:
            del row[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "input_hidden": self.input_hidden,
            "recurrent_hidden": self.recurrent_hidden,
            "hidden_output": self.hidden_output,
            "hidden_bias": self.hidden_bias,
            "output_bias": self.output_bias,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrainGenome:
        return cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            output_size=int(data["output_size"]),
            input_hidden=[list(map(float, row)) for row in data["input_hidden"]],
            recurrent_hidden=[list(map(float, row)) for row in data["recurrent_hidden"]],
            hidden_output=[list(map(float, row)) for row in data["hidden_output"]],
            hidden_bias=list(map(float, data["hidden_bias"])),
            output_bias=list(map(float, data["output_bias"])),
        )

    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> BrainGenome:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
