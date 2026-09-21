from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path
from typing import Any, Sequence

from brain import BrainGenome
from evolution import EvolutionConfig, GenerationStats
from world import EpisodeResult, FoodWorld, FoodWorldConfig, StepTrace


@dataclass(frozen=True)
class ReplayEpisode:
    index: int
    seed: int
    episode: EpisodeResult


@dataclass(frozen=True)
class RankedReplay:
    """A ranked brain paired with a trace used only for visualization."""

    rank: int
    score: float
    eat_rate: float
    episode: EpisodeResult
    label: str | None = None
    metric: str = "training"


def write_find_food_visualization(
    run_dir: Path,
    best_genome: BrainGenome,
    history: list[GenerationStats],
    evolution_config: EvolutionConfig,
    world_config: FoodWorldConfig,
    seed: int,
) -> Path:
    return write_visualization(
        run_dir=run_dir,
        best_genome=best_genome,
        history=history,
        evolution_config=evolution_config,
        world_config=world_config,
        seed=seed,
    )


def write_visualization(
    run_dir: Path,
    best_genome: BrainGenome,
    history: list[GenerationStats],
    evolution_config: EvolutionConfig,
    world_config: FoodWorldConfig,
    seed: int,
    world_class: type[FoodWorld] = FoodWorld,
    experiment: str = "find_food",
    replay_episodes: list[ReplayEpisode] | None = None,
    diagnostics: Sequence[dict[str, Any]] | None = None,
    selected_generation: int | None = None,
) -> Path:
    if replay_episodes is None:
        world = world_class(world_config)
        episode = world.evaluate(best_genome, random.Random(seed), record_trace=True)
        replay_episodes = [ReplayEpisode(index=0, seed=seed, episode=episode)]
    if not replay_episodes:
        raise ValueError("Visualization requires at least one replay episode.")

    episode_payloads = []
    for replay in replay_episodes:
        if replay.episode.trace is None:
            raise RuntimeError("Visualization requires recorded episode traces.")
        episode_payloads.append(
            {
                "index": replay.index,
                "seed": replay.seed,
                **episode_to_dict(replay.episode),
            }
        )

    payload = {
        "experiment": experiment,
        "config": {
            "evolution": asdict(evolution_config),
            "world": asdict(world_config),
        },
        "brain": {
            "input_size": best_genome.input_size,
            "hidden_size": best_genome.hidden_size,
            "output_size": best_genome.output_size,
        },
        "metrics": [asdict(stats) for stats in history],
        "diagnostics": list(diagnostics or []),
        "selected_generation": selected_generation,
        "episode": episode_payloads[0],
        "episodes": episode_payloads,
    }

    replay_path = run_dir / "replay.json"
    replay_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    template_path = Path(__file__).with_name("viewer_template.html")
    template = template_path.read_text(encoding="utf-8")
    embedded_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__RUN_DATA_JSON__", embedded_json)

    viewer_path = run_dir / "viewer.html"
    viewer_path.write_text(html, encoding="utf-8")
    return viewer_path


def write_ranked_visualization(
    run_dir: Path,
    ranked_replays: Sequence[RankedReplay],
    history: list[GenerationStats],
    evolution_config: EvolutionConfig,
    world_config: FoodWorldConfig,
    *,
    experiment: str = "compare_memory",
    label: str = "comparison summary",
    diagnostics: Sequence[dict[str, Any]] | None = None,
    viewer_filename: str = "comparison-viewer.html",
    data_filename: str = "comparison-replay.json",
    selected_stats: GenerationStats | None = None,
    status_message: str | None = None,
) -> Path:
    """Write a standalone version of the final ranked live-viewer screen."""
    if not ranked_replays:
        raise ValueError("Ranked visualization requires at least one replay.")
    if not history:
        raise ValueError("Ranked visualization requires generation history.")

    leaders = _ranked_replays_to_dicts(ranked_replays)
    latest = history[-1]
    snapshot = selected_stats or latest
    winner = leaders[0]
    if status_message is not None:
        message = status_message
    elif ranked_replays[0].metric == "diagnostic":
        message = f"One brain in {len(leaders)} balanced rooms"
    else:
        message = (
            f"{label}: top {len(leaders)} runs ranked by "
            f"{ranked_replays[0].metric} score"
        )
    state = {
        "experiment": experiment,
        "label": label,
        "phase": "complete",
        "message": message,
        "config": {
            "evolution": asdict(evolution_config),
            "world": asdict(world_config),
        },
        "progress": {
            "generation": snapshot.generation,
            "completed": evolution_config.population_size,
            "total": evolution_config.population_size,
            "workers": evolution_config.workers,
        },
        "current": {
            "best_score": _round_float(snapshot.best_score),
            "average_score": _round_float(snapshot.average_score),
            "best_eat_rate": _round_float(snapshot.best_eat_rate),
        },
        "metrics": [asdict(stats) for stats in history],
        "diagnostics": list(diagnostics or []),
        "selected_generation": snapshot.generation,
        "samples": [],
        "generation_leaders": leaders,
        "generation_best": None,
        "final": {
            "generation": snapshot.generation,
            "score": winner["score"],
            "eat_rate": winner["eat_rate"],
            "episode": winner["episode"],
            "leaders": leaders,
            "viewer_path": None,
        },
    }

    data_path = run_dir / data_filename
    data_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    template_path = Path(__file__).with_name("live_viewer_template.html")
    template = template_path.read_text(encoding="utf-8")
    embedded_json = json.dumps(state, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("__STATIC_STATE_JSON__", embedded_json)

    viewer_path = run_dir / viewer_filename
    viewer_path.write_text(html, encoding="utf-8")
    return viewer_path


def episode_to_dict(episode: EpisodeResult) -> dict[str, Any]:
    """Convert a recorded world episode into browser-safe JSON data."""
    if episode.trace is None:
        raise ValueError("The episode must be recorded with record_trace=True.")

    return {
        "score": _round_float(episode.score),
        "ate_food": episode.ate_food,
        "steps": episode.steps,
        "final_distance": _round_float(episode.final_distance),
        "collisions": episode.collisions,
        "targets": [
            {
                "label": target.label,
                "x": _round_float(target.x),
                "y": _round_float(target.y),
                "is_correct": target.is_correct,
            }
            for target in episode.targets or []
        ],
        "choice": episode.choice,
        "chose_correct": episode.chose_correct,
        "obstacles": [
            {
                "x_min": _round_float(obstacle.x_min),
                "y_min": _round_float(obstacle.y_min),
                "x_max": _round_float(obstacle.x_max),
                "y_max": _round_float(obstacle.y_max),
            }
            for obstacle in episode.obstacles or []
        ],
        "trace": [_trace_step_to_dict(step) for step in episode.trace],
    }


def _ranked_replays_to_dicts(
    replays: Sequence[RankedReplay],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_index": f"rank-{replay.rank}",
            "rank": replay.rank,
            "label": replay.label,
            "metric": replay.metric,
            "worker_pid": None,
            "score": _round_float(replay.score),
            "eat_rate": _round_float(replay.eat_rate),
            "episode": episode_to_dict(replay.episode),
        }
        for replay in replays
    ]


def _trace_step_to_dict(step: StepTrace) -> dict[str, Any]:
    return {
        "step": step.step,
        "agent_x": _round_float(step.agent_x),
        "agent_y": _round_float(step.agent_y),
        "heading": _round_float(step.heading),
        "food_x": _round_float(step.food_x),
        "food_y": _round_float(step.food_y),
        "distance": _round_float(step.distance),
        "score": _round_float(step.score),
        "observation": [_round_float(value) for value in step.observation],
        "output": [_round_float(value) for value in step.output],
        "hidden": [_round_float(value) for value in step.hidden],
        "food_visible": step.food_visible,
        "movement_enabled": step.movement_enabled,
        "collided": step.collided,
        "cue_visible": step.cue_visible,
        "cue_target": step.cue_target,
    }


def _round_float(value: float) -> float:
    return round(value, 5)
