from __future__ import annotations

import random
import unittest
from collections import Counter

from brain import BrainGenome
from experiments.generalize_rooms import (
    ValidationGenerationRecord,
    _balanced_room_scenarios,
    _evaluate_scenarios,
    _evaluate_validation,
    _live_trace_episode_indices,
    _representative_validation_replays,
    _validation_selection_key,
)
from world import RoomFoodWorld, RoomFoodWorldConfig


class GeneralizeRoomsTest(unittest.TestCase):
    def test_balanced_sets_cover_all_room_combinations_equally(self) -> None:
        world_config = RoomFoodWorldConfig(max_steps=6)
        scenarios = _balanced_room_scenarios(
            world_config,
            count=24,
            first_seed=2_000_010,
        )
        signatures = [
            (scenario.orientation, scenario.start_side, scenario.door_side)
            for scenario in scenarios
        ]
        expected = {
            (orientation, start_side, door_side)
            for orientation in ("vertical", "horizontal")
            for start_side in ("negative", "positive")
            for door_side in ("negative", "positive")
        }

        self.assertEqual(len(scenarios), 24)
        self.assertEqual(len({scenario.seed for scenario in scenarios}), 24)
        self.assertEqual(set(signatures[:8]), expected)
        self.assertEqual(set(Counter(signatures).values()), {3})

    def test_validation_records_all_rooms_and_selects_by_eat_rate(self) -> None:
        world_config = RoomFoodWorldConfig(max_steps=6)
        scenarios = _balanced_room_scenarios(
            world_config,
            count=8,
            first_seed=2_000_010,
        )

        world = RoomFoodWorld(world_config)
        genome = BrainGenome.random(
            random.Random(53),
            input_size=world.input_size,
            hidden_size=6,
            output_size=world.output_size,
        )
        validation, replays, records = _evaluate_validation(
            genome,
            world_config,
            scenarios,
            generation=3,
        )

        self.assertEqual(validation.generation, 3)
        self.assertEqual(len(replays), 8)
        self.assertEqual(len(records), 8)
        self.assertTrue(all(replay.metric == "diagnostic" for replay in replays))
        self.assertTrue(all(replay.episode.trace for replay in replays))
        self.assertEqual(
            len(_representative_validation_replays(replays, scenarios)),
            8,
        )

        lower_eat_rate = ValidationGenerationRecord(
            generation=1,
            average_score=1_000.0,
            overall_eat_rate=0.5,
            vertical_eat_rate=0.5,
            horizontal_eat_rate=0.5,
            wall_cross_rate=1.0,
            stuck_rate=0.0,
            average_collisions=0.0,
        )
        higher_eat_rate = ValidationGenerationRecord(
            generation=2,
            average_score=-1_000.0,
            overall_eat_rate=0.75,
            vertical_eat_rate=0.75,
            horizontal_eat_rate=0.75,
            wall_cross_rate=0.75,
            stuck_rate=0.25,
            average_collisions=10.0,
        )
        self.assertGreater(
            _validation_selection_key(higher_eat_rate),
            _validation_selection_key(lower_eat_rate),
        )

    def test_live_samples_include_both_wall_orientations(self) -> None:
        scenarios = _balanced_room_scenarios(
            RoomFoodWorldConfig(),
            count=8,
            first_seed=7,
        )
        seeds = tuple(scenario.seed for scenario in scenarios)
        indices = _live_trace_episode_indices(seeds, sample_count=4)

        self.assertEqual(
            [RoomFoodWorld.layout_orientation_for_seed(seeds[index]) for index in indices],
            ["vertical", "horizontal", "vertical", "horizontal"],
        )

    def test_evaluation_records_unseen_room_replays(self) -> None:
        world_config = RoomFoodWorldConfig(max_steps=8)
        world = RoomFoodWorld(world_config)
        genome = BrainGenome.random(
            random.Random(51),
            input_size=world.input_size,
            hidden_size=6,
            output_size=world.output_size,
        )

        records, replays = _evaluate_scenarios(
            genome,
            world_config,
            (101, 102, 103),
            split="holdout",
            record_replays=True,
        )

        self.assertEqual([record.seed for record in records], [101, 102, 103])
        self.assertTrue(all(record.split == "holdout" for record in records))
        self.assertEqual([replay.seed for replay in replays], [101, 102, 103])
        self.assertTrue(all(len(replay.episode.obstacles or []) == 2 for replay in replays))


if __name__ == "__main__":
    unittest.main()
