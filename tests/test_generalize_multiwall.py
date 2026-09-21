from __future__ import annotations

import random
import unittest
from collections import Counter

from brain import BrainGenome
from experiments.generalize_multiwall import (
    _balanced_multiwall_scenarios,
    _count_crossed_walls,
    _evaluate_validation,
    _representative_validation_replays,
)
from world import (
    EpisodeResult,
    MultiWallFoodWorld,
    MultiWallFoodWorldConfig,
    StepTrace,
    WorldObstacle,
)


class GeneralizeMultiWallTest(unittest.TestCase):
    def test_balanced_sets_cover_all_sixteen_combinations(self) -> None:
        scenarios = _balanced_multiwall_scenarios(
            MultiWallFoodWorldConfig(max_steps=6),
            count=32,
            first_seed=101,
        )
        signatures = [
            (
                item.orientation,
                item.start_side,
                item.first_door_side,
                item.door_pattern,
            )
            for item in scenarios
        ]

        self.assertEqual(len(scenarios), 32)
        self.assertEqual(len(set(signatures)), 16)
        self.assertEqual(set(Counter(signatures).values()), {2})
        self.assertEqual(len({item.seed for item in scenarios}), 32)

    def test_validation_reports_route_categories_and_representatives(self) -> None:
        config = MultiWallFoodWorldConfig(max_steps=6)
        scenarios = _balanced_multiwall_scenarios(
            config,
            count=16,
            first_seed=201,
        )
        world = MultiWallFoodWorld(config)
        genome = BrainGenome.random(
            random.Random(202),
            input_size=world.input_size,
            hidden_size=6,
            output_size=world.output_size,
        )

        validation, replays, records = _evaluate_validation(
            genome,
            config,
            scenarios,
            generation=3,
        )

        self.assertEqual(validation.generation, 3)
        self.assertEqual(len(replays), 16)
        self.assertEqual(len(records), 16)
        self.assertEqual(
            len(_representative_validation_replays(replays, scenarios)),
            8,
        )
        self.assertTrue(all(0 <= item.walls_crossed <= 2 for item in records))
        self.assertTrue(all(replay.episode.trace for replay in replays))

    def test_crossed_wall_count_uses_both_dividers(self) -> None:
        obstacles = [
            WorldObstacle(-2.1, -5.0, -1.9, -1.0),
            WorldObstacle(-2.1, 1.0, -1.9, 5.0),
            WorldObstacle(1.9, -5.0, 2.1, -1.0),
            WorldObstacle(1.9, 1.0, 2.1, 5.0),
        ]
        trace = [self._trace(step, x) for step, x in enumerate((-4.0, 0.0, 4.0))]
        episode = EpisodeResult(
            score=0.0,
            ate_food=False,
            steps=2,
            final_distance=1.0,
            trace=trace,
            obstacles=obstacles,
        )

        self.assertEqual(_count_crossed_walls(episode), 2)

    @staticmethod
    def _trace(step: int, x: float) -> StepTrace:
        return StepTrace(
            step=step,
            agent_x=x,
            agent_y=0.0,
            heading=0.0,
            food_x=4.5,
            food_y=0.0,
            distance=4.5 - x,
            score=0.0,
            observation=[0.0] * 9,
            output=[0.0] * 3,
            hidden=[0.0] * 4,
        )


if __name__ == "__main__":
    unittest.main()
