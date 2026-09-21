from __future__ import annotations

import unittest

from evolution import EvolutionConfig
from experiments.compare_generalization import (
    SCENARIO_NAMESPACE_SIZE,
    _resolve_seeds,
    _run_replicate,
    _scenario_namespace,
)
from world import RoomFoodWorldConfig


class CompareGeneralizationTest(unittest.TestCase):
    def test_resolve_seeds_supports_generated_and_explicit_sequences(self) -> None:
        self.assertEqual(_resolve_seeds(7, 3, 10, None), (7, 17, 27))
        self.assertEqual(_resolve_seeds(7, 99, 10, [3, 11]), (3, 11))

    def test_scenario_namespaces_are_disjoint_for_signed_seeds(self) -> None:
        namespaces = [_scenario_namespace(seed) for seed in (-2, -1, 0, 1, 2)]

        self.assertEqual(len(set(namespaces)), len(namespaces))
        for left in namespaces:
            for right in namespaces:
                if left != right:
                    self.assertGreaterEqual(abs(left - right), SCENARIO_NAMESPACE_SIZE)

    def test_replicate_uses_separate_balanced_splits_and_holdout(self) -> None:
        seed = 13
        result = _run_replicate(
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
            world_config=RoomFoodWorldConfig(max_steps=6),
            validation_episodes=8,
            holdout_episodes=8,
        )

        training_seeds = {item.seed for item in result.training_scenarios}
        validation_seeds = {item.seed for item in result.validation_scenarios}
        holdout_seeds = {item.seed for item in result.holdout_scenarios}
        namespace = _scenario_namespace(seed)

        self.assertEqual(len(training_seeds), 8)
        self.assertEqual(len(validation_seeds), 8)
        self.assertEqual(len(holdout_seeds), 8)
        self.assertTrue(training_seeds.isdisjoint(validation_seeds))
        self.assertTrue(training_seeds.isdisjoint(holdout_seeds))
        self.assertTrue(validation_seeds.isdisjoint(holdout_seeds))
        self.assertGreaterEqual(min(training_seeds), namespace)
        self.assertLess(max(holdout_seeds), namespace + SCENARIO_NAMESPACE_SIZE)
        self.assertEqual(len(result.training.history), 2)
        self.assertEqual(len(result.training.validation_history), 2)
        self.assertEqual(len(result.training.validation_episode_records), 16)
        self.assertIn(result.record.selected_generation, (0, 1))
        self.assertGreaterEqual(result.record.holdout_eat_rate, 0.0)
        self.assertLessEqual(result.record.holdout_eat_rate, 1.0)
        self.assertEqual(len(result.holdout_records), 8)
        self.assertEqual(len(result.holdout_replays), 8)
        self.assertTrue(all(replay.episode.trace for replay in result.holdout_replays))
        self.assertEqual(len(result.category_records), 8)
        self.assertTrue(all(item.episodes == 1 for item in result.category_records))


if __name__ == "__main__":
    unittest.main()
