from __future__ import annotations

import random
import unittest

from brain import BrainGenome
from evolution import EvolutionConfig, run_evolution, training_episode_seeds
from world import EpisodeResult, FoodWorld, FoodWorldConfig, MemoryFoodWorld, MemoryFoodWorldConfig


class _ScenarioProbeWorld(FoodWorld):
    def evaluate(
        self,
        genome: BrainGenome,
        rng,
        record_trace: bool = False,
    ) -> EpisodeResult:
        del genome, record_trace
        marker = rng.random()
        return EpisodeResult(
            score=marker,
            ate_food=False,
            steps=1,
            final_distance=marker,
        )


class _GenomeScoreWorld(FoodWorld):
    def evaluate(
        self,
        genome: BrainGenome,
        rng,
        record_trace: bool = False,
    ) -> EpisodeResult:
        del rng, record_trace
        score = genome.hidden_bias[0]
        return EpisodeResult(
            score=score,
            ate_food=score > 0,
            steps=1,
            final_distance=0.0,
        )


class EvolutionTest(unittest.TestCase):
    def test_small_evolution_run_returns_history(self) -> None:
        config = EvolutionConfig(
            population_size=8,
            generations=3,
            episodes_per_genome=2,
            hidden_size=6,
            min_hidden_size=4,
            max_hidden_size=12,
            seed=11,
        )

        best, history = run_evolution(config, FoodWorldConfig(max_steps=8))

        self.assertEqual(len(history), 3)
        self.assertGreaterEqual(best.hidden_size, 4)
        self.assertLessEqual(best.hidden_size, 12)

    def test_progress_callback_includes_sample_traces(self) -> None:
        config = EvolutionConfig(
            population_size=5,
            generations=2,
            episodes_per_genome=1,
            hidden_size=6,
            min_hidden_size=4,
            max_hidden_size=12,
            seed=29,
            workers=2,
        )
        progress = []

        run_evolution(
            config,
            FoodWorldConfig(max_steps=4),
            on_evaluation=progress.append,
            trace_sample_size=2,
        )

        self.assertEqual(len(progress), 10)
        self.assertEqual(
            [item.completed for item in progress[:5]],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [item.completed for item in progress[5:]],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(sum(item.replay is not None for item in progress), 4)

    def test_trace_samples_can_show_different_training_episodes(self) -> None:
        config = EvolutionConfig(
            population_size=4,
            generations=1,
            episodes_per_genome=4,
            hidden_size=6,
            min_hidden_size=4,
            max_hidden_size=12,
            seed=29,
            workers=2,
        )
        progress = []

        run_evolution(
            config,
            FoodWorldConfig(max_steps=2),
            on_evaluation=progress.append,
            trace_sample_size=4,
            trace_episode_indices=(0, 1, 2, 3),
            world_class=_ScenarioProbeWorld,
        )

        samples = sorted(
            (item for item in progress if item.replay),
            key=lambda item: item.candidate_index,
        )
        observed = [item.replay.final_distance for item in samples]
        expected = [
            random.Random(seed).random()
            for seed in training_episode_seeds(config)
        ]
        self.assertEqual(observed, expected)

    def test_evolution_supports_memory_world(self) -> None:
        config = EvolutionConfig(
            population_size=6,
            generations=2,
            episodes_per_genome=1,
            hidden_size=6,
            min_hidden_size=4,
            max_hidden_size=12,
            seed=31,
        )

        best, history = run_evolution(
            config,
            MemoryFoodWorldConfig(max_steps=8, visible_steps=2),
            world_class=MemoryFoodWorld,
        )

        self.assertEqual(len(history), 2)
        self.assertGreaterEqual(best.hidden_size, 4)

    def test_ranked_callback_receives_top_candidates_in_order(self) -> None:
        config = EvolutionConfig(
            population_size=6,
            generations=1,
            episodes_per_genome=1,
            hidden_size=6,
            min_hidden_size=4,
            max_hidden_size=12,
            seed=37,
        )
        ranked_generations = []

        run_evolution(
            config,
            FoodWorldConfig(max_steps=2),
            on_generation_ranked=lambda _stats, candidates: ranked_generations.append(
                candidates
            ),
            ranked_candidate_count=4,
            world_class=_GenomeScoreWorld,
        )

        self.assertEqual(len(ranked_generations), 1)
        leaders = ranked_generations[0]
        self.assertEqual([candidate.rank for candidate in leaders], [1, 2, 3, 4])
        self.assertEqual(
            [candidate.score for candidate in leaders],
            sorted((candidate.score for candidate in leaders), reverse=True),
        )

    def test_candidates_and_generations_share_training_scenarios(self) -> None:
        config = EvolutionConfig(
            population_size=5,
            generations=2,
            episodes_per_genome=3,
            hidden_size=6,
            min_hidden_size=4,
            max_hidden_size=12,
            seed=41,
        )
        progress = []

        run_evolution(
            config,
            FoodWorldConfig(max_steps=2),
            on_evaluation=progress.append,
            world_class=_ScenarioProbeWorld,
        )

        first_generation_scores = {
            round(item.score, 12) for item in progress if item.generation == 0
        }
        second_generation_scores = {
            round(item.score, 12) for item in progress if item.generation == 1
        }
        self.assertEqual(len(first_generation_scores), 1)
        self.assertEqual(len(second_generation_scores), 1)
        self.assertEqual(first_generation_scores, second_generation_scores)

    def test_explicit_episode_seeds_override_default_training_scenarios(self) -> None:
        config = EvolutionConfig(
            population_size=4,
            generations=1,
            episodes_per_genome=4,
            hidden_size=6,
            seed=43,
        )
        explicit_seeds = (101, 202, 303, 404)
        progress = []

        run_evolution(
            config,
            FoodWorldConfig(max_steps=2),
            on_evaluation=progress.append,
            trace_sample_size=4,
            trace_episode_indices=(0, 1, 2, 3),
            world_class=_ScenarioProbeWorld,
            episode_seeds=explicit_seeds,
        )

        samples = sorted(progress, key=lambda item: item.candidate_index)
        self.assertEqual(
            [item.replay.final_distance for item in samples],
            [random.Random(seed).random() for seed in explicit_seeds],
        )

        with self.assertRaisesRegex(ValueError, "episodes_per_genome"):
            run_evolution(
                config,
                FoodWorldConfig(max_steps=2),
                world_class=_ScenarioProbeWorld,
                episode_seeds=(101, 202),
            )

    def test_episode_seed_sets_rotate_between_generations(self) -> None:
        config = EvolutionConfig(
            population_size=4,
            generations=3,
            episodes_per_genome=2,
            hidden_size=6,
            seed=47,
        )
        seed_sets = ((101, 202), (303, 404))
        progress = []

        run_evolution(
            config,
            FoodWorldConfig(max_steps=2),
            on_evaluation=progress.append,
            world_class=_ScenarioProbeWorld,
            episode_seed_sets=seed_sets,
        )

        scores_by_generation = {
            generation: {
                round(item.score, 12)
                for item in progress
                if item.generation == generation
            }
            for generation in range(config.generations)
        }
        expected = [
            round(
                sum(random.Random(seed).random() for seed in seed_set)
                / len(seed_set),
                12,
            )
            for seed_set in seed_sets
        ]
        self.assertEqual(scores_by_generation[0], {expected[0]})
        self.assertEqual(scores_by_generation[1], {expected[1]})
        self.assertEqual(scores_by_generation[2], {expected[0]})

        with self.assertRaisesRegex(ValueError, "either"):
            run_evolution(
                config,
                FoodWorldConfig(max_steps=2),
                world_class=_ScenarioProbeWorld,
                episode_seeds=seed_sets[0],
                episode_seed_sets=seed_sets,
            )

    def test_training_episode_seeds_are_fixed_and_distinct(self) -> None:
        config = EvolutionConfig(episodes_per_genome=4, seed=7)

        self.assertEqual(
            training_episode_seeds(config),
            (7, 10_014, 20_021, 30_028),
        )


if __name__ == "__main__":
    unittest.main()
