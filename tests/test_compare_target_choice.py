from __future__ import annotations

import unittest

from evolution import EvolutionConfig
from experiments.compare_target_choice import (
    SCENARIO_NAMESPACE_SIZE,
    _resolve_seeds,
    _run_replicate,
    _scenario_namespace,
)
from world import TargetChoiceWorldConfig


class CompareTargetChoiceTest(unittest.TestCase):
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

    def test_replicate_pairs_conditions_and_keeps_splits_disjoint(self) -> None:
        seed = 13
        result = _run_replicate(
            replicate=0,
            seed=seed,
            evolution_config=EvolutionConfig(
                population_size=6,
                generations=2,
                episodes_per_genome=16,
                hidden_size=6,
                min_hidden_size=4,
                max_hidden_size=10,
                seed=seed,
            ),
            base_world_config=TargetChoiceWorldConfig(
                max_steps=6,
                cue_steps=2,
            ),
            validation_episodes=16,
            holdout_episodes=16,
        )

        training_seeds = {item.seed for item in result.training_scenarios}
        validation_seeds = {item.seed for item in result.validation_scenarios}
        holdout_seeds = {item.seed for item in result.holdout_scenarios}
        namespace = _scenario_namespace(seed)

        self.assertEqual(len(training_seeds), 256)
        self.assertEqual(len(result.training_scenario_sets), 16)
        self.assertTrue(
            all(len(scenarios) == 16 for scenarios in result.training_scenario_sets)
        )
        self.assertEqual(len(validation_seeds), 16)
        self.assertEqual(len(holdout_seeds), 16)
        self.assertTrue(training_seeds.isdisjoint(validation_seeds))
        self.assertTrue(training_seeds.isdisjoint(holdout_seeds))
        self.assertTrue(validation_seeds.isdisjoint(holdout_seeds))
        self.assertGreaterEqual(min(training_seeds), namespace)
        self.assertLess(max(holdout_seeds), namespace + SCENARIO_NAMESPACE_SIZE)

        self.assertTrue(result.recurrent.world_config.use_recurrence)
        self.assertFalse(result.stateless.world_config.use_recurrence)
        self.assertEqual(len(result.recurrent.training.history), 2)
        self.assertEqual(len(result.stateless.training.history), 2)
        self.assertEqual(len(result.recurrent.holdout_records), 16)
        self.assertEqual(len(result.stateless.holdout_records), 16)
        self.assertEqual(len(result.knockout_records), 16)
        self.assertEqual(len(result.recurrent.category_records), 8)
        self.assertEqual(len(result.stateless.category_records), 8)
        self.assertEqual(
            {item.seed for item in result.recurrent.holdout_records},
            {item.seed for item in result.stateless.holdout_records},
        )
        self.assertEqual(
            {item.seed for item in result.recurrent.holdout_records},
            {item.seed for item in result.knockout_records},
        )
        self.assertGreaterEqual(
            result.pair_record.recurrent_holdout_success_rate,
            0.0,
        )
        self.assertLessEqual(
            result.pair_record.recurrent_holdout_success_rate,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
