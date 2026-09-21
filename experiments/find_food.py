from __future__ import annotations

import argparse
from pathlib import Path

from experiments.runner import (
    add_evolution_arguments,
    run_experiment,
    validate_evolution_arguments,
)
from world import FoodWorld, FoodWorldConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve small recurrent brains that learn to find food.",
    )
    add_evolution_arguments(parser, output=Path("results/find_food"))
    args = parser.parse_args()
    validate_evolution_arguments(parser, args)

    run_experiment(
        args,
        experiment_name="find_food",
        world_class=FoodWorld,
        world_config=FoodWorldConfig(max_steps=args.max_steps),
        next_question="Does structural mutation improve the result compared with weight-only mutation?",
    )


if __name__ == "__main__":
    main()
