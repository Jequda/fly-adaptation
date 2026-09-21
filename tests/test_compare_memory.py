from __future__ import annotations

import unittest

from evolution import EvolutionConfig
from experiments.compare_memory import _run_condition
from world import MemoryFoodWorldConfig


class CompareMemoryTest(unittest.TestCase):
    def test_condition_uses_holdout_episodes(self) -> None:
        record, best_genome, history, holdout_replays = _run_condition(
            condition="recurrent",
            replicate=0,
            seed=53,
            evolution_config=EvolutionConfig(
                population_size=6,
                generations=2,
                episodes_per_genome=1,
                hidden_size=6,
                min_hidden_size=4,
                max_hidden_size=10,
                seed=53,
            ),
            world_config=MemoryFoodWorldConfig(max_steps=8, visible_steps=2),
            holdout_episodes=3,
        )

        self.assertEqual(record.condition, "recurrent")
        self.assertEqual(record.replicate, 0)
        self.assertGreaterEqual(record.holdout_eat_rate, 0.0)
        self.assertLessEqual(record.holdout_eat_rate, 1.0)
        self.assertEqual(best_genome.input_size, 6)
        self.assertEqual(len(history), 2)
        self.assertEqual([replay.seed for replay in holdout_replays], [50053, 50054, 50055])
        self.assertTrue(all(replay.episode.trace for replay in holdout_replays))


if __name__ == "__main__":
    unittest.main()
