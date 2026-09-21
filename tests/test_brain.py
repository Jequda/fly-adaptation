from __future__ import annotations

import random
import unittest

from brain import BrainGenome


class BrainGenomeTest(unittest.TestCase):
    def test_activate_keeps_expected_output_shape(self) -> None:
        rng = random.Random(1)
        genome = BrainGenome.random(rng, input_size=6, hidden_size=12, output_size=3)

        output, state = genome.activate([0.0, 1.0, 0.5, 0.0, 0.0, 1.0], genome.initial_state())

        self.assertEqual(len(output), 3)
        self.assertEqual(len(state.hidden), 12)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in output))

    def test_structural_mutation_preserves_network_shapes(self) -> None:
        rng = random.Random(2)
        genome = BrainGenome.random(rng, input_size=6, hidden_size=12, output_size=3)

        child = genome.mutated(
            rng,
            weight_mutation_rate=0.0,
            weight_mutation_power=0.0,
            structural_mutation_rate=1.0,
            min_hidden_size=8,
            max_hidden_size=80,
        )

        self.assertEqual(len(child.input_hidden), child.hidden_size)
        self.assertEqual(len(child.recurrent_hidden), child.hidden_size)
        self.assertEqual(len(child.hidden_output), child.output_size)
        self.assertTrue(all(len(row) == child.hidden_size for row in child.recurrent_hidden))
        self.assertTrue(all(len(row) == child.hidden_size for row in child.hidden_output))


if __name__ == "__main__":
    unittest.main()
