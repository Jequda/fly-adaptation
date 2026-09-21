from __future__ import annotations

from dataclasses import replace
import unittest

from evolution import EvolutionConfig
from experiments.compare_wall_sensors import (
    FULL,
    NO_POSITION,
    POSITION_KNOCKOUT,
    POSITION_SENSOR_SPEC,
    _paired_showcase_replays,
    _run_ablation_pair,
)
from world import RoomFoodWorldConfig


class ComparePositionSensorsTest(unittest.TestCase):
    def test_position_ablation_retrains_and_knocks_out_only_agent_xy(self) -> None:
        seed = 29
        full_world_config = RoomFoodWorldConfig(
            max_steps=6,
            use_position_sensors=True,
        )
        no_position_world_config = replace(
            full_world_config,
            use_position_sensors=False,
        )
        result = _run_ablation_pair(
            replicate=0,
            seed=seed,
            evolution_config=EvolutionConfig(
                population_size=6,
                generations=2,
                episodes_per_genome=8,
                hidden_size=6,
                min_hidden_size=4,
                max_hidden_size=10,
                seed=seed,
            ),
            full_world_config=full_world_config,
            ablated_world_config=no_position_world_config,
            validation_episodes=8,
            holdout_episodes=8,
            spec=POSITION_SENSOR_SPEC,
        )

        self.assertEqual(
            result.full.training_scenarios,
            result.ablated.training_scenarios,
        )
        self.assertEqual(
            {item.condition for item in result.category_records},
            {FULL, NO_POSITION, POSITION_KNOCKOUT},
        )
        self.assertEqual(result.ablated.training.best_validation_genome.input_size, 9)
        self.assertAlmostEqual(
            result.record.trained_ablation_delta,
            result.record.full_holdout_eat_rate
            - result.record.ablated_holdout_eat_rate,
        )

        ablated_replays = result.ablated.holdout_replays + result.knockout_replays
        observations = [
            frame.observation
            for replay in ablated_replays
            for frame in replay.episode.trace or []
        ]
        self.assertTrue(observations)
        self.assertTrue(all(values[3:5] == [0.0, 0.0] for values in observations))

        showcase = _paired_showcase_replays(
            result,
            full_world_config,
            no_position_world_config,
            visual_seed=1007,
            spec=POSITION_SENSOR_SPEC,
        )
        self.assertEqual(
            [item.label for item in showcase],
            [
                "full | seed 29",
                "retrained without position | seed 29",
                "full brain, position off | seed 29",
            ],
        )


if __name__ == "__main__":
    unittest.main()
