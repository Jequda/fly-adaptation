from __future__ import annotations

from dataclasses import replace
import unittest

from evolution import EvolutionConfig
from experiments.compare_wall_sensors import (
    FULL,
    KNOCKOUT,
    NO_WALL,
    _ablation_conclusion,
    _paired_showcase_replays,
    _run_ablation_pair,
)
from world import RoomFoodWorldConfig


class CompareWallSensorsTest(unittest.TestCase):
    def test_paired_ablation_reuses_rooms_and_records_real_sensor_knockout(self) -> None:
        seed = 23
        full_world_config = RoomFoodWorldConfig(
            max_steps=6,
            use_wall_sensors=True,
        )
        no_wall_world_config = replace(
            full_world_config,
            use_wall_sensors=False,
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
            ablated_world_config=no_wall_world_config,
            validation_episodes=8,
            holdout_episodes=8,
        )

        self.assertEqual(
            result.full.training_scenarios,
            result.ablated.training_scenarios,
        )
        self.assertEqual(
            result.full.validation_scenarios,
            result.ablated.validation_scenarios,
        )
        self.assertEqual(
            result.full.holdout_scenarios,
            result.ablated.holdout_scenarios,
        )
        self.assertEqual(result.full.training.best_validation_genome.input_size, 9)
        self.assertEqual(result.ablated.training.best_validation_genome.input_size, 9)
        self.assertEqual(len(result.knockout_records), 8)
        self.assertEqual(len(result.knockout_replays), 8)
        self.assertEqual(len(result.category_records), 24)
        self.assertEqual(
            {item.condition for item in result.category_records},
            {FULL, NO_WALL, KNOCKOUT},
        )
        self.assertAlmostEqual(
            result.record.trained_ablation_delta,
            result.record.full_holdout_eat_rate
            - result.record.ablated_holdout_eat_rate,
        )
        self.assertAlmostEqual(
            result.record.knockout_delta,
            result.record.full_holdout_eat_rate
            - result.record.knockout_holdout_eat_rate,
        )

        no_wall_replays = result.ablated.holdout_replays + result.knockout_replays
        observations = [
            frame.observation[6:]
            for replay in no_wall_replays
            for frame in replay.episode.trace or []
        ]
        self.assertTrue(observations)
        self.assertTrue(all(values == [0.0, 0.0, 0.0] for values in observations))

        showcase = _paired_showcase_replays(
            result,
            full_world_config,
            no_wall_world_config,
            visual_seed=1007,
        )
        self.assertEqual(len(showcase), 3)
        self.assertTrue(all(item.metric == "ablation" for item in showcase))
        self.assertEqual(
            [item.label for item in showcase],
            [
                "full | seed 23",
                "retrained without wall sensors | seed 23",
                "full brain, wall sensors off | seed 23",
            ],
        )

    def test_conclusion_distinguishes_learning_and_current_reliance(self) -> None:
        self.assertIn(
            "неинтерпретируем",
            _ablation_conclusion(False, False, baseline_ready=False),
        )
        self.assertIn("помогают обучению", _ablation_conclusion(True, True))
        self.assertIn("альтернативу", _ablation_conclusion(False, True))
        self.assertIn("эволюционному поиску", _ablation_conclusion(True, False))
        self.assertIn("не обнаружена", _ablation_conclusion(False, False))


if __name__ == "__main__":
    unittest.main()
