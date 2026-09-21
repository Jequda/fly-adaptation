from __future__ import annotations

import random
import unittest

from brain import BrainGenome
from world import (
    FoodWorld,
    FoodWorldConfig,
    MemoryFoodWorld,
    MemoryFoodWorldConfig,
    MultiWallFoodWorld,
    MultiWallFoodWorldConfig,
    RoomFoodWorld,
    RoomFoodWorldConfig,
    TargetChoiceWorld,
    TargetChoiceWorldConfig,
)


class FoodWorldTest(unittest.TestCase):
    def test_world_evaluates_a_brain(self) -> None:
        rng = random.Random(3)
        world = FoodWorld(FoodWorldConfig(max_steps=10))
        genome = BrainGenome.random(
            rng,
            input_size=world.input_size,
            hidden_size=8,
            output_size=world.output_size,
        )

        result = world.evaluate(genome, rng)

        self.assertEqual(result.steps <= 10, True)
        self.assertIsInstance(result.score, float)
        self.assertGreaterEqual(result.final_distance, 0.0)

    def test_world_can_record_episode_trace(self) -> None:
        rng = random.Random(4)
        world = FoodWorld(FoodWorldConfig(max_steps=5))
        genome = BrainGenome.random(
            rng,
            input_size=world.input_size,
            hidden_size=8,
            output_size=world.output_size,
        )

        result = world.evaluate(genome, rng, record_trace=True)

        self.assertIsNotNone(result.trace)
        self.assertGreaterEqual(len(result.trace or []), 1)
        self.assertEqual(len((result.trace or [])[0].observation), world.input_size)
        self.assertEqual(len((result.trace or [])[0].output), world.output_size)

    def test_memory_world_hides_food_sensors_after_cue(self) -> None:
        rng = random.Random(17)
        world = MemoryFoodWorld(
            MemoryFoodWorldConfig(max_steps=6, visible_steps=2),
        )
        genome = BrainGenome.random(
            rng,
            input_size=world.input_size,
            hidden_size=8,
            output_size=world.output_size,
        )

        result = world.evaluate(genome, rng, record_trace=True)
        trace = result.trace or []

        self.assertGreaterEqual(len(trace), 4)
        self.assertTrue(trace[1].food_visible)
        self.assertFalse(trace[1].movement_enabled)
        self.assertFalse(trace[3].food_visible)
        self.assertTrue(trace[3].movement_enabled)
        self.assertEqual(trace[3].observation[:3], [0.0, 0.0, 0.0])

    def test_room_world_adds_walls_and_range_sensors(self) -> None:
        world = RoomFoodWorld(RoomFoodWorldConfig(max_steps=8))
        genome = BrainGenome.random(
            random.Random(21),
            input_size=world.input_size,
            hidden_size=8,
            output_size=world.output_size,
        )

        result = world.evaluate(genome, random.Random(22), record_trace=True)
        trace = result.trace or []

        self.assertEqual(world.input_size, 9)
        self.assertEqual(len(result.obstacles or []), 2)
        self.assertEqual(len(trace[0].observation), 9)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in trace[0].observation[6:]))

    def test_room_layout_is_reproducible_for_a_seed(self) -> None:
        world = RoomFoodWorld(RoomFoodWorldConfig(max_steps=4))
        genome = BrainGenome.random(
            random.Random(31),
            input_size=world.input_size,
            hidden_size=8,
            output_size=world.output_size,
        )

        first = world.evaluate(genome, random.Random(99), record_trace=True)
        second = world.evaluate(genome, random.Random(99), record_trace=True)

        self.assertEqual(first.obstacles, second.obstacles)
        self.assertEqual(first.trace, second.trace)

    def test_room_world_can_ablate_wall_sensors_without_changing_input_size(self) -> None:
        world = RoomFoodWorld(
            RoomFoodWorldConfig(max_steps=6, use_wall_sensors=False)
        )
        genome = BrainGenome.random(
            random.Random(41),
            input_size=world.input_size,
            hidden_size=8,
            output_size=world.output_size,
        )

        result = world.evaluate(genome, random.Random(42), record_trace=True)
        trace = result.trace or []

        self.assertEqual(world.input_size, 9)
        self.assertTrue(trace)
        self.assertTrue(all(frame.observation[6:] == [0.0, 0.0, 0.0] for frame in trace))

    def test_room_world_can_ablate_position_without_changing_other_inputs(self) -> None:
        full_world = RoomFoodWorld(RoomFoodWorldConfig(max_steps=6))
        no_position_world = RoomFoodWorld(
            RoomFoodWorldConfig(max_steps=6, use_position_sensors=False)
        )
        genome = BrainGenome.random(
            random.Random(43),
            input_size=full_world.input_size,
            hidden_size=8,
            output_size=full_world.output_size,
        )

        full_result = full_world.evaluate(
            genome,
            random.Random(44),
            record_trace=True,
        )
        no_position_result = no_position_world.evaluate(
            genome,
            random.Random(44),
            record_trace=True,
        )
        full_observation = (full_result.trace or [])[0].observation
        no_position_observation = (no_position_result.trace or [])[0].observation

        self.assertEqual(no_position_world.input_size, 9)
        self.assertEqual(no_position_observation[3:5], [0.0, 0.0])
        self.assertEqual(no_position_observation[:3], full_observation[:3])
        self.assertEqual(no_position_observation[5:], full_observation[5:])

    def test_multi_wall_world_has_two_doors_and_no_position_signal(self) -> None:
        world = MultiWallFoodWorld(MultiWallFoodWorldConfig(max_steps=8))
        genome = BrainGenome.random(
            random.Random(61),
            input_size=world.input_size,
            hidden_size=8,
            output_size=world.output_size,
        )

        result = world.evaluate(genome, random.Random(62), record_trace=True)
        observation = (result.trace or [])[0].observation

        self.assertEqual(world.input_size, 9)
        self.assertEqual(len(result.obstacles or []), 4)
        self.assertEqual(observation[3:5], [0.0, 0.0])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in observation[6:]))

    def test_multi_wall_layout_and_signature_are_reproducible(self) -> None:
        world = MultiWallFoodWorld(MultiWallFoodWorldConfig(max_steps=4))
        genome = BrainGenome.random(
            random.Random(63),
            input_size=world.input_size,
            hidden_size=8,
            output_size=world.output_size,
        )

        first = world.evaluate(genome, random.Random(64), record_trace=True)
        second = world.evaluate(genome, random.Random(64), record_trace=True)

        self.assertEqual(first.obstacles, second.obstacles)
        self.assertEqual(first.trace, second.trace)
        self.assertEqual(
            world.scenario_signature_for_seed(64),
            world.scenario_signature_for_seed(64),
        )

    def test_target_choice_world_hides_rule_but_keeps_both_targets_visible(self) -> None:
        world = TargetChoiceWorld(
            TargetChoiceWorldConfig(max_steps=8, cue_steps=2)
        )
        genome = BrainGenome.random(
            random.Random(71),
            input_size=world.input_size,
            hidden_size=8,
            output_size=world.output_size,
        )

        result = world.evaluate(genome, random.Random(72), record_trace=True)
        trace = result.trace or []

        self.assertEqual(world.input_size, 9)
        self.assertEqual({target.label for target in result.targets or []}, {"A", "B"})
        self.assertGreaterEqual(len(trace), 4)
        self.assertTrue(trace[1].cue_visible)
        self.assertFalse(trace[1].movement_enabled)
        self.assertFalse(trace[3].cue_visible)
        self.assertTrue(trace[3].movement_enabled)
        self.assertIn(trace[1].observation[6], (-1.0, 1.0))
        self.assertEqual(trace[1].observation[7], 1.0)
        self.assertEqual(trace[3].observation[6:8], [0.0, 0.0])
        self.assertNotEqual(trace[3].observation[:6], [0.0] * 6)
        self.assertIsNotNone(result.target_preference)
        self.assertGreaterEqual(float(result.target_preference), -1.0)
        self.assertLessEqual(float(result.target_preference), 1.0)


if __name__ == "__main__":
    unittest.main()
