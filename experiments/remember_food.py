from __future__ import annotations

import argparse
from pathlib import Path

from experiments.runner import (
    add_evolution_arguments,
    run_experiment,
    validate_evolution_arguments,
)
from world import MemoryFoodWorld, MemoryFoodWorldConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve recurrent brains that must remember a food cue.",
    )
    add_evolution_arguments(parser, output=Path("results/remember_food"))
    parser.set_defaults(episodes=10)
    parser.add_argument(
        "--visible-steps",
        type=int,
        default=8,
        help="Cue steps before food direction and distance sensors are hidden.",
    )
    parser.add_argument(
        "--no-recurrence",
        action="store_true",
        help="Reset hidden state every step for a memory ablation.",
    )
    args = parser.parse_args()
    validate_evolution_arguments(parser, args)
    if args.visible_steps < 1:
        parser.error("--visible-steps must be at least 1")
    if args.visible_steps >= args.max_steps:
        parser.error("--visible-steps must be smaller than --max-steps")

    world_config = MemoryFoodWorldConfig(
        max_steps=args.max_steps,
        visible_steps=args.visible_steps,
        use_recurrence=not args.no_recurrence,
    )
    run_experiment(
        args,
        experiment_name="remember_food",
        world_class=MemoryFoodWorld,
        world_config=world_config,
        next_question=(
            "Does recurrence improve hidden-food success compared with --no-recurrence?"
        ),
    )


if __name__ == "__main__":
    main()
