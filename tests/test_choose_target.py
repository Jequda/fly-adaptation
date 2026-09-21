from __future__ import annotations

import random
import unittest
from collections import Counter

from brain import BrainGenome
from experiments.choose_target import (
    TargetValidationRecord,
    _balanced_target_scenarios,
    _evaluate_validation,
    _representative_validation_replays,
    _validation_selection_key,
)
from world import (
    EpisodeResult,
    TARGET_SECTORS,
    TargetChoiceWorld,
    TargetChoiceWorldConfig,
    WorldTarget,
)


class ChooseTargetTest(unittest.TestCase):
    def test_fitness_rewards_improvement_on_the_weaker_cue(self) -> None:
        world = TargetChoiceWorld(TargetChoiceWorldConfig(max_steps=10, cue_steps=2))

        specialized = [
            _target_result(cue, sector, success=cue == "A")
            for sector in TARGET_SECTORS
            for cue in ("A", "B")
        ]
        balanced = [
            _target_result(
                cue,
                sector,
                success=index % 2 == (0 if cue == "A" else 1),
            )
            for index, sector in enumerate(TARGET_SECTORS)
            for cue in ("A", "B")
        ]

        self.assertGreater(
            world.aggregate_fitness(balanced),
            world.aggregate_fitness(specialized),
        )

    def test_fitness_prefers_success_spread_across_sectors(self) -> None:
        world = TargetChoiceWorld(TargetChoiceWorldConfig(max_steps=10, cue_steps=2))
        concentrated = [
            _target_result(cue, sector, success=sector_index < 2)
            for sector_index, sector in enumerate(TARGET_SECTORS)
            for _ in range(2)
            for cue in ("A", "B")
        ]
        spread = [
            _target_result(cue, sector, success=episode_index == 0)
            for sector in TARGET_SECTORS
            for episode_index in range(2)
            for cue in ("A", "B")
        ]

        self.assertEqual(
            sum(result.ate_food for result in concentrated),
            sum(result.ate_food for result in spread),
        )
        self.assertGreater(
            world.aggregate_fitness(spread),
            world.aggregate_fitness(concentrated),
        )

    def test_fitness_rewards_cue_dependent_choices_before_success(self) -> None:
        world = TargetChoiceWorld(TargetChoiceWorldConfig(max_steps=10, cue_steps=2))
        insensitive = [
            _target_result(cue, sector, success=False)
            for sector in TARGET_SECTORS
            for cue in ("A", "B")
        ]
        contrasting = [
            _target_result(cue, sector, success=False)
            for sector in TARGET_SECTORS
            for cue in ("A", "B")
        ]
        for result in insensitive:
            result.choice = "A"
            result.chose_correct = None
        for index, result in enumerate(contrasting):
            result.choice = "A" if index % 2 else "B"
            result.chose_correct = False

        self.assertGreater(
            world.aggregate_fitness(contrasting),
            world.aggregate_fitness(insensitive),
        )

    def test_fitness_rewards_small_correct_pair_alignment(self) -> None:
        world = TargetChoiceWorld(TargetChoiceWorldConfig(max_steps=10, cue_steps=2))
        insensitive = [
            _target_result(cue, sector, success=False)
            for sector in TARGET_SECTORS
            for cue in ("A", "B")
        ]
        aligned = [
            _target_result(cue, sector, success=False)
            for sector in TARGET_SECTORS
            for cue in ("A", "B")
        ]
        for result in aligned:
            correct_target = next(
                target.label for target in result.targets or [] if target.is_correct
            )
            result.target_preference = 0.1 if correct_target == "A" else -0.1

        self.assertGreater(
            world.aggregate_fitness(aligned),
            world.aggregate_fitness(insensitive),
        )

    def test_balanced_sets_cover_cue_and_target_sector_equally(self) -> None:
        scenarios = _balanced_target_scenarios(
            TargetChoiceWorldConfig(max_steps=10, cue_steps=2),
            count=32,
            first_seed=101,
        )
        signatures = [
            (item.correct_target, item.target_a_sector) for item in scenarios
        ]

        self.assertEqual(len(scenarios), 32)
        self.assertEqual(len(set(signatures)), 8)
        self.assertEqual(set(Counter(signatures).values()), {4})
        self.assertEqual(len({item.seed for item in scenarios}), 32)
        geometry_groups = {}
        world = TargetChoiceWorld(TargetChoiceWorldConfig(max_steps=10, cue_steps=2))
        for scenario in scenarios:
            geometry_groups.setdefault(scenario.geometry_key, []).append(scenario)
        self.assertEqual(len(geometry_groups), 16)
        for pair in geometry_groups.values():
            self.assertEqual({item.correct_target for item in pair}, {"A", "B"})
            first = world._make_scenario(random.Random(pair[0].seed))
            second = world._make_scenario(random.Random(pair[1].seed))
            self.assertEqual(
                (first.target_a_x, first.target_a_y),
                (second.target_a_x, second.target_a_y),
            )

        stateless_world = TargetChoiceWorld(
            TargetChoiceWorldConfig(
                max_steps=10,
                cue_steps=2,
                use_recurrence=False,
            )
        )
        genome = BrainGenome.random(
            random.Random(102),
            input_size=stateless_world.input_size,
            hidden_size=6,
            output_size=stateless_world.output_size,
        )
        for pair in geometry_groups.values():
            outcomes = [
                stateless_world.evaluate(genome, random.Random(item.seed))
                for item in pair
            ]
            self.assertEqual(outcomes[0].choice, outcomes[1].choice)
            self.assertAlmostEqual(
                float(outcomes[0].target_preference),
                float(outcomes[1].target_preference),
            )
            self.assertLessEqual(sum(item.ate_food for item in outcomes), 1)

    def test_validation_records_choices_and_selects_by_success(self) -> None:
        config = TargetChoiceWorldConfig(max_steps=10, cue_steps=2)
        scenarios = _balanced_target_scenarios(
            config,
            count=16,
            first_seed=201,
        )
        world = TargetChoiceWorld(config)
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
        self.assertTrue(all(replay.episode.trace for replay in replays))
        self.assertAlmostEqual(
            validation.overall_success_rate
            + validation.wrong_choice_rate
            + validation.no_choice_rate,
            1.0,
        )

        lower_success = TargetValidationRecord(
            generation=1,
            average_score=1_000.0,
            overall_success_rate=0.5,
            cue_a_success_rate=1.0,
            cue_b_success_rate=0.0,
            cue_gap=1.0,
            pair_alignment=0.0,
            pair_contrast_rate=0.0,
            target_a_negative_success_rate=0.5,
            target_a_positive_success_rate=0.5,
            worst_category="Cue B|A back-lower",
            worst_category_success_rate=0.0,
            wrong_choice_rate=0.5,
            no_choice_rate=0.0,
            average_success_steps=20.0,
        )
        higher_success = TargetValidationRecord(
            generation=2,
            average_score=-1_000.0,
            overall_success_rate=0.75,
            cue_a_success_rate=0.75,
            cue_b_success_rate=0.75,
            cue_gap=0.0,
            pair_alignment=0.5,
            pair_contrast_rate=0.5,
            target_a_negative_success_rate=0.75,
            target_a_positive_success_rate=0.75,
            worst_category="Cue A|A front-upper",
            worst_category_success_rate=0.5,
            wrong_choice_rate=0.25,
            no_choice_rate=0.0,
            average_success_steps=30.0,
        )
        self.assertGreater(
            _validation_selection_key(higher_success),
            _validation_selection_key(lower_success),
        )


def _target_result(
    correct_target: str,
    sector: str,
    *,
    success: bool,
) -> EpisodeResult:
    positions = {
        "front-upper": (1.0, 1.0),
        "back-upper": (-1.0, 1.0),
        "back-lower": (-1.0, -1.0),
        "front-lower": (1.0, -1.0),
    }
    target_a_x, target_a_y = positions[sector]
    return EpisodeResult(
        score=100.0 if success else 0.0,
        ate_food=success,
        steps=1,
        final_distance=0.0 if success else 1.0,
        targets=[
            WorldTarget("A", target_a_x, target_a_y, correct_target == "A"),
            WorldTarget("B", -target_a_x, -target_a_y, correct_target == "B"),
        ],
        choice=correct_target if success else None,
        chose_correct=True if success else None,
        target_preference=(
            1.0 if success and correct_target == "A"
            else -1.0 if success
            else 0.0
        ),
    )


if __name__ == "__main__":
    unittest.main()
