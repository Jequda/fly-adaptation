from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
import unittest

from brain import BrainGenome
from evolution import EvolutionConfig, GenerationStats
from visualization import (
    RankedReplay,
    ReplayEpisode,
    write_find_food_visualization,
    write_ranked_visualization,
    write_visualization,
)
from world import (
    FoodWorldConfig,
    MemoryFoodWorld,
    MemoryFoodWorldConfig,
    RoomFoodWorld,
    RoomFoodWorldConfig,
    TargetChoiceWorld,
    TargetChoiceWorldConfig,
)


class VisualizationTest(unittest.TestCase):
    def test_write_visualization_embeds_replay_data(self) -> None:
        rng = random.Random(5)
        config = EvolutionConfig(population_size=2, generations=1, hidden_size=6)
        world_config = FoodWorldConfig(max_steps=3)
        genome = BrainGenome.random(rng, input_size=6, hidden_size=6, output_size=3)
        history = [
            GenerationStats(
                generation=0,
                best_score=1.0,
                average_score=0.5,
                best_eat_rate=0.0,
                average_hidden_size=6.0,
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            viewer_path = write_find_food_visualization(
                run_dir=Path(temp_dir),
                best_genome=genome,
                history=history,
                evolution_config=config,
                world_config=world_config,
                seed=42,
            )

            html = viewer_path.read_text(encoding="utf-8")
            self.assertIn("const RUN_DATA =", html)
            self.assertNotIn("__RUN_DATA_JSON__", html)
            self.assertTrue((Path(temp_dir) / "replay.json").exists())

    def test_memory_visualization_exports_food_visibility(self) -> None:
        rng = random.Random(9)
        config = EvolutionConfig(population_size=2, generations=1, hidden_size=6)
        world_config = MemoryFoodWorldConfig(max_steps=5, visible_steps=2)
        genome = BrainGenome.random(rng, input_size=6, hidden_size=6, output_size=3)
        history = [
            GenerationStats(
                generation=0,
                best_score=1.0,
                average_score=0.5,
                best_eat_rate=0.0,
                average_hidden_size=6.0,
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            write_visualization(
                run_dir=Path(temp_dir),
                best_genome=genome,
                history=history,
                evolution_config=config,
                world_config=world_config,
                seed=42,
                world_class=MemoryFoodWorld,
                experiment="remember_food",
            )

            replay = json.loads((Path(temp_dir) / "replay.json").read_text(encoding="utf-8"))
            self.assertEqual(replay["experiment"], "remember_food")
            self.assertTrue(replay["episode"]["trace"][1]["food_visible"])
            self.assertFalse(replay["episode"]["trace"][3]["food_visible"])

    def test_visualization_embeds_multiple_episode_traces(self) -> None:
        rng = random.Random(11)
        config = EvolutionConfig(population_size=2, generations=1, hidden_size=6)
        world_config = MemoryFoodWorldConfig(max_steps=5, visible_steps=2)
        world = MemoryFoodWorld(world_config)
        genome = BrainGenome.random(rng, input_size=6, hidden_size=6, output_size=3)
        history = [GenerationStats(0, 1.0, 0.5, 0.0, 6.0)]
        replay_episodes = [
            ReplayEpisode(
                index=index,
                seed=seed,
                episode=world.evaluate(
                    genome,
                    random.Random(seed),
                    record_trace=True,
                ),
            )
            for index, seed in enumerate((101, 102, 103))
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            viewer_path = write_visualization(
                run_dir=Path(temp_dir),
                best_genome=genome,
                history=history,
                evolution_config=config,
                world_config=world_config,
                seed=101,
                world_class=MemoryFoodWorld,
                experiment="remember_food",
                replay_episodes=replay_episodes,
            )

            replay = json.loads((Path(temp_dir) / "replay.json").read_text(encoding="utf-8"))
            html = viewer_path.read_text(encoding="utf-8")
            self.assertEqual(len(replay["episodes"]), 3)
            self.assertEqual([item["seed"] for item in replay["episodes"]], [101, 102, 103])
            self.assertIn("data-episode-filter", html)

    def test_ranked_visualization_is_standalone(self) -> None:
        config = EvolutionConfig(population_size=6, generations=2, hidden_size=6)
        world_config = MemoryFoodWorldConfig(max_steps=5, visible_steps=2)
        world = MemoryFoodWorld(world_config)
        history = [
            GenerationStats(0, 1.0, 0.5, 0.0, 6.0),
            GenerationStats(1, 2.0, 1.0, 0.5, 6.0),
        ]
        ranked_replays = []
        for rank in range(1, 5):
            genome = BrainGenome.random(
                random.Random(rank),
                input_size=6,
                hidden_size=6,
                output_size=3,
            )
            ranked_replays.append(
                RankedReplay(
                    rank=rank,
                    score=10.0 - rank,
                    eat_rate=0.5,
                    episode=world.evaluate(
                        genome,
                        random.Random(100 + rank),
                        record_trace=True,
                    ),
                    label=f"run {rank}",
                    metric="holdout",
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics = [
                {
                    "generation": 1,
                    "vertical_eat_rate": 0.75,
                    "horizontal_eat_rate": 0.5,
                    "wall_cross_rate": 0.875,
                    "stuck_rate": 0.125,
                }
            ]
            viewer_path = write_ranked_visualization(
                run_dir=Path(temp_dir),
                ranked_replays=ranked_replays,
                history=history,
                evolution_config=config,
                world_config=world_config,
                diagnostics=diagnostics,
                viewer_filename="validation-viewer.html",
                data_filename="validation-replay.json",
                selected_stats=history[0],
                status_message="Shared holdout summary",
            )

            html = viewer_path.read_text(encoding="utf-8")
            state = json.loads(
                (Path(temp_dir) / "validation-replay.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("__STATIC_STATE_JSON__", html)
            self.assertIn("const STATIC_STATE = {", html)
            self.assertEqual(len(state["final"]["leaders"]), 4)
            self.assertEqual(state["final"]["leaders"][0]["label"], "run 1")
            self.assertEqual(state["diagnostics"], diagnostics)
            self.assertEqual(state["selected_generation"], 0)
            self.assertEqual(state["final"]["generation"], 0)
            self.assertEqual(state["message"], "Shared holdout summary")
            self.assertEqual(viewer_path.name, "validation-viewer.html")

    def test_room_visualization_exports_obstacles_and_collisions(self) -> None:
        config = EvolutionConfig(population_size=2, generations=1, hidden_size=6)
        world_config = RoomFoodWorldConfig(max_steps=8)
        world = RoomFoodWorld(world_config)
        genome = BrainGenome.random(
            random.Random(41),
            input_size=world.input_size,
            hidden_size=6,
            output_size=world.output_size,
        )
        history = [GenerationStats(0, 1.0, 0.5, 0.0, 6.0)]

        with tempfile.TemporaryDirectory() as temp_dir:
            write_visualization(
                run_dir=Path(temp_dir),
                best_genome=genome,
                history=history,
                evolution_config=config,
                world_config=world_config,
                seed=42,
                world_class=RoomFoodWorld,
                experiment="generalize_rooms",
                selected_generation=0,
            )

            replay = json.loads(
                (Path(temp_dir) / "replay.json").read_text(encoding="utf-8")
            )
            html = (Path(temp_dir) / "viewer.html").read_text(encoding="utf-8")
            self.assertEqual(len(replay["episode"]["obstacles"]), 2)
            self.assertIn("collided", replay["episode"]["trace"][0])
            self.assertEqual(replay["selected_generation"], 0)
            self.assertIn("wall front", html)

    def test_target_choice_visualization_exports_two_targets_and_cue(self) -> None:
        config = EvolutionConfig(population_size=2, generations=1, hidden_size=6)
        world_config = TargetChoiceWorldConfig(max_steps=8, cue_steps=2)
        world = TargetChoiceWorld(world_config)
        genome = BrainGenome.random(
            random.Random(51),
            input_size=world.input_size,
            hidden_size=6,
            output_size=world.output_size,
        )
        history = [GenerationStats(0, 1.0, 0.5, 0.0, 6.0)]

        with tempfile.TemporaryDirectory() as temp_dir:
            write_visualization(
                run_dir=Path(temp_dir),
                best_genome=genome,
                history=history,
                evolution_config=config,
                world_config=world_config,
                seed=52,
                world_class=TargetChoiceWorld,
                experiment="choose_target",
            )

            replay = json.loads(
                (Path(temp_dir) / "replay.json").read_text(encoding="utf-8")
            )
            html = (Path(temp_dir) / "viewer.html").read_text(encoding="utf-8")
            self.assertEqual({item["label"] for item in replay["episode"]["targets"]}, {"A", "B"})
            self.assertTrue(replay["episode"]["trace"][1]["cue_visible"])
            self.assertFalse(replay["episode"]["trace"][3]["cue_visible"])
            self.assertIn("A sin angle", html)
            self.assertIn("CUE HIDDEN", html)


if __name__ == "__main__":
    unittest.main()
