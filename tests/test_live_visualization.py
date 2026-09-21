from __future__ import annotations

import json
import random
import unittest
from urllib.request import ProxyHandler, build_opener

from brain import BrainGenome
from evolution import EvaluationProgress, EvolutionConfig, GenerationStats
from visualization import LiveRunViewer, RankedReplay
from world import FoodWorld, FoodWorldConfig


class LiveVisualizationTest(unittest.TestCase):
    def test_local_server_exposes_live_progress(self) -> None:
        evolution_config = EvolutionConfig(population_size=3, generations=2, hidden_size=6)
        world_config = FoodWorldConfig(max_steps=3)
        genome = BrainGenome.random(
            random.Random(3),
            input_size=6,
            hidden_size=6,
            output_size=3,
        )
        episode = FoodWorld(world_config).evaluate(
            genome,
            random.Random(4),
            record_trace=True,
        )
        viewer = LiveRunViewer(evolution_config, world_config, sample_limit=4)

        try:
            url = viewer.start()
            html = _read_text(url)
            self.assertIn("const STATIC_STATE = null", html)
            self.assertNotIn("__STATIC_STATE_JSON__", html)
            initial = _read_state(url)
            self.assertEqual(initial["phase"], "starting")

            viewer.record_evaluation(
                EvaluationProgress(
                    generation=0,
                    completed=1,
                    total=3,
                    candidate_index=0,
                    worker_pid=1234,
                    score=2.5,
                    eat_rate=0.0,
                    genome=genome,
                    replay=episode,
                )
            )
            updated = _read_state(url)
            self.assertEqual(updated["progress"]["completed"], 1)
            self.assertEqual(updated["samples"][0]["worker_pid"], 1234)
            self.assertEqual(len(updated["samples"][0]["episode"]["trace"]), len(episode.trace or []))

            stats = GenerationStats(0, 2.5, 1.0, 0.0, 6.0)
            leaders = [
                RankedReplay(
                    rank=rank,
                    score=3.0 - rank,
                    eat_rate=0.5,
                    episode=episode,
                )
                for rank in range(1, 5)
            ]
            viewer.record_generation_leaders(stats, leaders)
            selected = _read_state(url)
            self.assertEqual(selected["phase"], "selecting")
            self.assertEqual(
                [item["rank"] for item in selected["generation_leaders"]],
                [1, 2, 3, 4],
            )
            diagnostics = {
                "generation": 0,
                "overall_eat_rate": 0.5,
                "vertical_eat_rate": 0.75,
                "horizontal_eat_rate": 0.25,
                "wall_cross_rate": 0.625,
                "stuck_rate": 0.125,
                "average_collisions": 3.5,
            }
            diagnostic_replays = [
                RankedReplay(
                    rank=rank,
                    score=3.0 - rank,
                    eat_rate=0.5,
                    episode=episode,
                    label=f"room {rank}",
                    metric="diagnostic",
                )
                for rank in range(1, 5)
            ]
            viewer.record_generation_diagnostics(
                stats,
                diagnostic_replays,
                diagnostics,
                selected_generation=0,
            )
            diagnosed = _read_state(url)
            self.assertEqual(diagnosed["phase"], "diagnostic")
            self.assertEqual(diagnosed["diagnostics"][-1], diagnostics)
            self.assertEqual(diagnosed["selected_generation"], 0)
            self.assertEqual(
                [item["label"] for item in diagnosed["generation_leaders"]],
                ["room 1", "room 2", "room 3", "room 4"],
            )
            viewer.begin_run(
                evolution_config,
                world_config,
                experiment="remember_food",
                label="stateless, repeat 1/1",
            )
            reset = _read_state(url)
            self.assertEqual(reset["phase"], "starting")
            self.assertEqual(reset["label"], "stateless, repeat 1/1")
            self.assertEqual(reset["samples"], [])
            viewer.finish(stats, episode, None, ranked_replays=leaders)
            finished = _read_state(url)
            self.assertEqual(finished["phase"], "complete")
            self.assertEqual(finished["final"]["generation"], 0)
            self.assertEqual(len(finished["final"]["leaders"]), 4)
            self.assertTrue(viewer.wait_until_final_served(timeout=0.1))
        finally:
            viewer.close()


def _read_state(url: str) -> dict[str, object]:
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"{url}api/state", timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_text(url: str) -> str:
    opener = build_opener(ProxyHandler({}))
    with opener.open(url, timeout=3) as response:
        return response.read().decode("utf-8")


if __name__ == "__main__":
    unittest.main()
