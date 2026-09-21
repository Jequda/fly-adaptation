from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from statistics import mean
from typing import Callable

from brain import BrainGenome
from evolution import (
    EvaluationProgress,
    EvolutionConfig,
    GenerationStats,
    RankedCandidate,
    run_evolution,
)
from experiments.generalize_rooms import _has_collision_streak
from experiments.runner import (
    _create_run_dir,
    _format_stats,
    _open_in_browser,
    _write_json,
    _write_metrics,
    add_evolution_arguments,
    evolution_config_from_args,
    validate_evolution_arguments,
)
from visualization import (
    LiveRunViewer,
    RankedReplay,
    ReplayEpisode,
    write_ranked_visualization,
    write_visualization,
)
from world import (
    EpisodeResult,
    MultiWallFoodWorld,
    MultiWallFoodWorldConfig,
    MultiWallScenarioSignature,
)


STUCK_STEPS = 8
WALL_COUNT = 2


@dataclass(frozen=True)
class BalancedMultiWallScenario:
    index: int
    seed: int
    orientation: str
    start_side: str
    first_door_side: str
    door_pattern: str

    @property
    def label(self) -> str:
        orientation = "V" if self.orientation == "vertical" else "H"
        start = "+" if self.start_side == "positive" else "-"
        first_door = "+" if self.first_door_side == "positive" else "-"
        route = "same side" if self.door_pattern == "aligned" else "zigzag"
        return f"{orientation} | start {start} | first door {first_door} | {route}"


@dataclass(frozen=True)
class MultiWallEvaluationRecord:
    split: str
    episode: int
    seed: int
    orientation: str
    start_side: str
    first_door_side: str
    door_pattern: str
    score: float
    ate_food: bool
    walls_crossed: int
    stuck: bool
    steps: int
    final_distance: float
    collisions: int


@dataclass(frozen=True)
class MultiWallValidationRecord:
    generation: int
    average_score: float
    overall_eat_rate: float
    vertical_eat_rate: float
    horizontal_eat_rate: float
    aligned_eat_rate: float
    alternating_eat_rate: float
    all_walls_crossed_rate: float
    average_walls_crossed: float
    stuck_rate: float
    average_collisions: float


@dataclass(frozen=True)
class MultiWallValidationEpisodeRecord:
    generation: int
    scenario: int
    seed: int
    orientation: str
    start_side: str
    first_door_side: str
    door_pattern: str
    score: float
    ate_food: bool
    walls_crossed: int
    stuck: bool
    collisions: int
    steps: int


@dataclass(frozen=True)
class MultiWallTrainingResult:
    best_training_genome: BrainGenome
    best_validation_genome: BrainGenome
    history: list[GenerationStats]
    validation_history: list[MultiWallValidationRecord]
    validation_episode_records: list[MultiWallValidationEpisodeRecord]
    selected_validation: MultiWallValidationRecord
    selected_stats: GenerationStats
    selected_replays: list[RankedReplay]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train without absolute coordinates in rooms with two walls.",
    )
    add_evolution_arguments(parser, output=Path("results/generalize_multiwall"))
    parser.set_defaults(population=80, generations=50, episodes=32, max_steps=160)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--holdout-episodes", type=int, default=64)
    parser.add_argument("--door-width", type=float, default=2.0)
    parser.add_argument("--wall-spacing", type=float, default=3.2)
    parser.add_argument("--door-offset", type=float, default=2.4)
    parser.add_argument("--door-jitter", type=float, default=0.45)
    parser.add_argument("--wall-sensor-range", type=float, default=2.8)
    parser.add_argument("--collision-penalty", type=float, default=0.25)
    args = parser.parse_args()
    validate_evolution_arguments(parser, args)
    for name in ("episodes", "validation_episodes", "holdout_episodes"):
        value = getattr(args, name)
        if value < 16 or value % 16 != 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive multiple of 16")

    evolution_config = evolution_config_from_args(args)
    world_config = MultiWallFoodWorldConfig(
        max_steps=args.max_steps,
        door_width=args.door_width,
        wall_spacing=args.wall_spacing,
        door_offset=args.door_offset,
        door_jitter=args.door_jitter,
        wall_sensor_range=args.wall_sensor_range,
        collision_penalty=args.collision_penalty,
        use_position_sensors=False,
    )
    run_dir = _create_run_dir(args.output)
    training_scenarios = _balanced_multiwall_scenarios(
        world_config,
        count=evolution_config.episodes_per_genome,
        first_seed=evolution_config.seed,
    )
    validation_scenarios = _balanced_multiwall_scenarios(
        world_config,
        count=args.validation_episodes,
        first_seed=evolution_config.seed + 2_000_003,
    )
    holdout_scenarios = _balanced_multiwall_scenarios(
        world_config,
        count=args.holdout_episodes,
        first_seed=evolution_config.seed + 4_000_003,
    )
    all_seeds = tuple(
        scenario.seed
        for scenarios in (training_scenarios, validation_scenarios, holdout_scenarios)
        for scenario in scenarios
    )
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("Training, validation, and holdout seeds must be disjoint.")

    _write_json(
        run_dir / "config.json",
        {
            "experiment": "generalize_multiwall",
            "evolution": asdict(evolution_config),
            "world": asdict(world_config),
            "selection": {
                "primary": "validation_eat_rate",
                "tie_breaker": "validation_average_score",
                "holdout_used": False,
            },
            "training_scenarios": [asdict(item) for item in training_scenarios],
            "validation_scenarios": [asdict(item) for item in validation_scenarios],
            "holdout_scenarios": [asdict(item) for item in holdout_scenarios],
        },
    )

    print(f"Run: {run_dir}")
    live_viewer = _start_live_viewer(args, evolution_config, world_config)
    print("generation,best_score,avg_score,best_eat_rate,avg_hidden", flush=True)

    def on_generation(stats: GenerationStats) -> None:
        print(_format_stats(stats), flush=True)

    def on_validation(
        stats: GenerationStats,
        validation: MultiWallValidationRecord,
        replays: list[RankedReplay],
        selected: bool,
        selected_generation: int,
    ) -> None:
        print(
            "  validation: "
            f"eat={validation.overall_eat_rate:.0%}, "
            f"V={validation.vertical_eat_rate:.0%}, "
            f"H={validation.horizontal_eat_rate:.0%}, "
            f"same={validation.aligned_eat_rate:.0%}, "
            f"zigzag={validation.alternating_eat_rate:.0%}, "
            f"crossed2={validation.all_walls_crossed_rate:.0%}, "
            f"stuck={validation.stuck_rate:.0%}"
            f"{'  <- checkpoint' if selected else ''}",
            flush=True,
        )
        if live_viewer is not None:
            live_viewer.record_generation_diagnostics(
                stats,
                replays,
                asdict(validation),
                selected_generation=selected_generation,
            )

    training_result = _run_training_with_validation(
        evolution_config=evolution_config,
        world_config=world_config,
        training_scenarios=training_scenarios,
        validation_scenarios=validation_scenarios,
        on_generation=on_generation,
        on_validation=on_validation,
        on_evaluation=live_viewer.record_evaluation if live_viewer else None,
        trace_sample_size=args.live_samples if live_viewer else 0,
        trace_episode_indices=(
            _live_trace_episode_indices(training_scenarios, args.live_samples)
            if live_viewer
            else None
        ),
    )

    training_records, _ = _evaluate_scenarios(
        training_result.best_validation_genome,
        world_config,
        training_scenarios,
        split="training",
        record_replays=False,
    )
    holdout_records, holdout_replays = _evaluate_scenarios(
        training_result.best_validation_genome,
        world_config,
        holdout_scenarios,
        split="holdout",
        record_replays=True,
    )

    _write_metrics(run_dir / "metrics.csv", training_result.history)
    _write_records(run_dir / "evaluation.csv", MultiWallEvaluationRecord, training_records + holdout_records)
    _write_records(run_dir / "validation.csv", MultiWallValidationRecord, training_result.validation_history)
    _write_records(
        run_dir / "validation_episodes.csv",
        MultiWallValidationEpisodeRecord,
        training_result.validation_episode_records,
    )
    training_result.best_validation_genome.save_json(run_dir / "best_genome.json")
    training_result.best_validation_genome.save_json(
        run_dir / "best_validation_genome.json"
    )
    training_result.best_training_genome.save_json(
        run_dir / "best_training_genome.json"
    )
    _write_json(
        run_dir / "selection.json",
        {
            "selected_generation": training_result.selected_stats.generation,
            "training_metrics_at_selection": asdict(training_result.selected_stats),
            "validation_metrics_at_selection": asdict(
                training_result.selected_validation
            ),
            "rule": ["maximum validation eat rate", "maximum validation score"],
            "holdout_used_for_selection": False,
        },
    )

    viewer_path = None
    validation_viewer_path = None
    diagnostics = [asdict(item) for item in training_result.validation_history]
    if not args.no_visualization:
        viewer_path = write_visualization(
            run_dir=run_dir,
            best_genome=training_result.best_validation_genome,
            history=training_result.history,
            evolution_config=evolution_config,
            world_config=world_config,
            seed=holdout_scenarios[0].seed,
            world_class=MultiWallFoodWorld,
            experiment="generalize_multiwall",
            replay_episodes=holdout_replays,
            diagnostics=diagnostics,
            selected_generation=training_result.selected_stats.generation,
        )
        validation_viewer_path = write_ranked_visualization(
            run_dir=run_dir,
            ranked_replays=training_result.selected_replays,
            history=training_result.history,
            evolution_config=evolution_config,
            world_config=world_config,
            experiment="generalize_multiwall",
            label=(
                "two walls | validation checkpoint | generation "
                f"{training_result.selected_stats.generation + 1}/"
                f"{evolution_config.generations}"
            ),
            diagnostics=diagnostics,
            viewer_filename="validation-viewer.html",
            data_filename="validation-replay.json",
            selected_stats=training_result.selected_stats,
        )

    _write_report(
        run_dir / "report.md",
        evolution_config,
        training_records,
        holdout_records,
        training_result.selected_validation,
        training_result.validation_history[-1],
        training_result.selected_stats,
        len(validation_scenarios),
        viewer_path,
        validation_viewer_path,
    )

    if live_viewer is not None:
        live_viewer.finish(
            training_result.selected_stats,
            training_result.selected_replays[0].episode,
            validation_viewer_path or viewer_path,
            ranked_replays=training_result.selected_replays,
            message=(
                "Training complete: two-wall checkpoint from generation "
                f"{training_result.selected_stats.generation + 1}/"
                f"{evolution_config.generations}"
            ),
        )
        live_viewer.wait_until_final_served()

    print()
    print(
        "Selected validation generation: "
        f"{training_result.selected_stats.generation + 1}/"
        f"{evolution_config.generations}"
    )
    print(
        "Validation eat rate: "
        f"{training_result.selected_validation.overall_eat_rate:.2%}"
    )
    print(f"Holdout eat rate: {_eat_rate(holdout_records):.2%}")
    print(
        "Holdout crossed both walls: "
        f"{mean(item.walls_crossed == WALL_COUNT for item in holdout_records):.2%}"
    )
    print(f"Saved results to: {run_dir}")
    if viewer_path is not None:
        print(f"Open visualization: {viewer_path}")
        print(f"Open validation: {validation_viewer_path}")
        if args.open_viewer and live_viewer is None:
            _open_in_browser(viewer_path.resolve().as_uri())


def _run_training_with_validation(
    *,
    evolution_config: EvolutionConfig,
    world_config: MultiWallFoodWorldConfig,
    training_scenarios: tuple[BalancedMultiWallScenario, ...],
    validation_scenarios: tuple[BalancedMultiWallScenario, ...],
    on_generation: Callable[[GenerationStats], None] | None = None,
    on_validation: (
        Callable[
            [GenerationStats, MultiWallValidationRecord, list[RankedReplay], bool, int],
            None,
        ]
        | None
    ) = None,
    on_evaluation: Callable[[EvaluationProgress], None] | None = None,
    trace_sample_size: int = 0,
    trace_episode_indices: tuple[int, ...] | None = None,
) -> MultiWallTrainingResult:
    validation_history: list[MultiWallValidationRecord] = []
    validation_episode_records: list[MultiWallValidationEpisodeRecord] = []
    selected_genome: BrainGenome | None = None
    selected_validation: MultiWallValidationRecord | None = None
    selected_stats: GenerationStats | None = None
    selected_replays: list[RankedReplay] = []
    selected_key: tuple[float, float] | None = None

    def on_generation_ranked(
        stats: GenerationStats,
        candidates: list[RankedCandidate],
    ) -> None:
        nonlocal selected_genome, selected_validation, selected_stats, selected_key
        validation, replays, episode_records = _evaluate_validation(
            candidates[0].genome,
            world_config,
            validation_scenarios,
            generation=stats.generation,
        )
        validation_history.append(validation)
        validation_episode_records.extend(episode_records)
        key = _validation_selection_key(validation)
        selected = selected_key is None or key > selected_key
        if selected:
            selected_key = key
            selected_genome = candidates[0].genome.clone()
            selected_validation = validation
            selected_stats = stats
            selected_replays.clear()
            selected_replays.extend(
                _representative_validation_replays(replays, validation_scenarios)
            )
        if on_validation is not None:
            if selected_stats is None:
                raise RuntimeError("Validation ran before checkpoint selection.")
            on_validation(
                stats,
                validation,
                _representative_validation_replays(replays, validation_scenarios),
                selected,
                selected_stats.generation,
            )

    best_training_genome, history = run_evolution(
        evolution_config,
        world_config,
        on_generation=on_generation,
        on_generation_ranked=on_generation_ranked,
        on_evaluation=on_evaluation,
        trace_sample_size=trace_sample_size,
        ranked_candidate_count=1,
        world_class=MultiWallFoodWorld,
        trace_episode_indices=trace_episode_indices,
        episode_seeds=tuple(item.seed for item in training_scenarios),
    )
    if selected_genome is None or selected_validation is None or selected_stats is None:
        raise RuntimeError("Evolution completed without a validation checkpoint.")
    return MultiWallTrainingResult(
        best_training_genome=best_training_genome,
        best_validation_genome=selected_genome,
        history=history,
        validation_history=validation_history,
        validation_episode_records=validation_episode_records,
        selected_validation=selected_validation,
        selected_stats=selected_stats,
        selected_replays=selected_replays,
    )


def _balanced_multiwall_scenarios(
    world_config: MultiWallFoodWorldConfig,
    *,
    count: int,
    first_seed: int,
) -> tuple[BalancedMultiWallScenario, ...]:
    if count < 1:
        raise ValueError("A balanced multi-wall set requires at least one scenario.")
    wanted = tuple(
        MultiWallScenarioSignature(orientation, start, first_door, pattern)
        for orientation in ("vertical", "horizontal")
        for pattern in ("aligned", "alternating")
        for start in ("negative", "positive")
        for first_door in ("negative", "positive")
    )
    base_count, remainder = divmod(count, len(wanted))
    quotas = {
        signature: base_count + (index < remainder)
        for index, signature in enumerate(wanted)
    }
    found = {signature: [] for signature in wanted}
    world = MultiWallFoodWorld(world_config)
    seed = first_seed
    while sum(len(seeds) for seeds in found.values()) < count:
        signature = world.scenario_signature_for_seed(seed)
        if signature in found and len(found[signature]) < quotas[signature]:
            found[signature].append(seed)
        seed += 1
        if seed - first_seed > max(200_000, count * 20_000):
            raise RuntimeError("Could not build a balanced multi-wall scenario set.")

    scenarios = []
    for round_index in range(max(quotas.values())):
        for signature in wanted:
            if round_index >= quotas[signature]:
                continue
            scenarios.append(
                BalancedMultiWallScenario(
                    index=len(scenarios),
                    seed=found[signature][round_index],
                    orientation=signature.orientation,
                    start_side=signature.start_side,
                    first_door_side=signature.first_door_side,
                    door_pattern=signature.door_pattern,
                )
            )
    return tuple(scenarios)


def _evaluate_validation(
    genome: BrainGenome,
    world_config: MultiWallFoodWorldConfig,
    scenarios: tuple[BalancedMultiWallScenario, ...],
    *,
    generation: int,
) -> tuple[
    MultiWallValidationRecord,
    list[RankedReplay],
    list[MultiWallValidationEpisodeRecord],
]:
    world = MultiWallFoodWorld(world_config)
    outcomes: list[tuple[BalancedMultiWallScenario, EpisodeResult, int, bool]] = []
    replays = []
    episode_records = []
    for scenario in scenarios:
        episode = world.evaluate(genome, random.Random(scenario.seed), record_trace=True)
        walls_crossed = _count_crossed_walls(episode)
        stuck = _has_collision_streak(episode, STUCK_STEPS)
        outcomes.append((scenario, episode, walls_crossed, stuck))
        replays.append(
            RankedReplay(
                rank=scenario.index + 1,
                score=episode.score,
                eat_rate=1.0 if episode.ate_food else 0.0,
                episode=episode,
                label=scenario.label,
                metric="diagnostic",
            )
        )
        episode_records.append(
            MultiWallValidationEpisodeRecord(
                generation=generation,
                scenario=scenario.index,
                seed=scenario.seed,
                orientation=scenario.orientation,
                start_side=scenario.start_side,
                first_door_side=scenario.first_door_side,
                door_pattern=scenario.door_pattern,
                score=episode.score,
                ate_food=episode.ate_food,
                walls_crossed=walls_crossed,
                stuck=stuck,
                collisions=episode.collisions,
                steps=episode.steps,
            )
        )

    vertical = [item for item in outcomes if item[0].orientation == "vertical"]
    horizontal = [item for item in outcomes if item[0].orientation == "horizontal"]
    aligned = [item for item in outcomes if item[0].door_pattern == "aligned"]
    alternating = [
        item for item in outcomes if item[0].door_pattern == "alternating"
    ]
    validation = MultiWallValidationRecord(
        generation=generation,
        average_score=mean(item[1].score for item in outcomes),
        overall_eat_rate=mean(item[1].ate_food for item in outcomes),
        vertical_eat_rate=mean(item[1].ate_food for item in vertical),
        horizontal_eat_rate=mean(item[1].ate_food for item in horizontal),
        aligned_eat_rate=mean(item[1].ate_food for item in aligned),
        alternating_eat_rate=mean(item[1].ate_food for item in alternating),
        all_walls_crossed_rate=mean(item[2] == WALL_COUNT for item in outcomes),
        average_walls_crossed=mean(item[2] for item in outcomes),
        stuck_rate=mean(item[3] for item in outcomes),
        average_collisions=mean(item[1].collisions for item in outcomes),
    )
    return validation, replays, episode_records


def _evaluate_scenarios(
    genome: BrainGenome,
    world_config: MultiWallFoodWorldConfig,
    scenarios: tuple[BalancedMultiWallScenario, ...],
    *,
    split: str,
    record_replays: bool,
) -> tuple[list[MultiWallEvaluationRecord], list[ReplayEpisode]]:
    world = MultiWallFoodWorld(world_config)
    records = []
    replays = []
    for scenario in scenarios:
        result = world.evaluate(genome, random.Random(scenario.seed), record_trace=True)
        records.append(
            MultiWallEvaluationRecord(
                split=split,
                episode=scenario.index,
                seed=scenario.seed,
                orientation=scenario.orientation,
                start_side=scenario.start_side,
                first_door_side=scenario.first_door_side,
                door_pattern=scenario.door_pattern,
                score=result.score,
                ate_food=result.ate_food,
                walls_crossed=_count_crossed_walls(result),
                stuck=_has_collision_streak(result, STUCK_STEPS),
                steps=result.steps,
                final_distance=result.final_distance,
                collisions=result.collisions,
            )
        )
        if record_replays:
            replays.append(
                ReplayEpisode(index=scenario.index, seed=scenario.seed, episode=result)
            )
    return records, replays


def _count_crossed_walls(episode: EpisodeResult) -> int:
    if not episode.trace or not episode.obstacles:
        return 0
    first_obstacle = episode.obstacles[0]
    vertical = (
        first_obstacle.y_max - first_obstacle.y_min
        > first_obstacle.x_max - first_obstacle.x_min
    )
    wall_positions = sorted(
        {
            round(
                (obstacle.x_min + obstacle.x_max) / 2
                if vertical
                else (obstacle.y_min + obstacle.y_max) / 2,
                9,
            )
            for obstacle in episode.obstacles
        }
    )
    start = episode.trace[0]
    start_axis = start.agent_x if vertical else start.agent_y
    crossed = 0
    for wall_position in wall_positions:
        start_delta = start_axis - wall_position
        if any(
            ((step.agent_x if vertical else step.agent_y) - wall_position)
            * start_delta
            < 0
            for step in episode.trace
        ):
            crossed += 1
    return crossed


def _representative_validation_replays(
    replays: list[RankedReplay],
    scenarios: tuple[BalancedMultiWallScenario, ...],
) -> list[RankedReplay]:
    if len(replays) != len(scenarios):
        raise ValueError("Validation replays and scenarios must have equal lengths.")
    selected = []
    seen: set[tuple[str, str, str]] = set()
    for replay, scenario in zip(replays, scenarios):
        signature = (scenario.orientation, scenario.start_side, scenario.door_pattern)
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(replay)
    return selected


def _live_trace_episode_indices(
    scenarios: tuple[BalancedMultiWallScenario, ...],
    sample_count: int,
) -> tuple[int, ...]:
    representatives = []
    seen: set[tuple[str, str]] = set()
    for index, scenario in enumerate(scenarios):
        key = (scenario.orientation, scenario.door_pattern)
        if key not in seen:
            seen.add(key)
            representatives.append(index)
    if not representatives:
        return (0,)
    return tuple(
        representatives[index % len(representatives)] for index in range(sample_count)
    )


def _validation_selection_key(
    record: MultiWallValidationRecord,
) -> tuple[float, float]:
    return record.overall_eat_rate, record.average_score


def _start_live_viewer(
    args: argparse.Namespace,
    evolution_config: EvolutionConfig,
    world_config: MultiWallFoodWorldConfig,
) -> LiveRunViewer | None:
    if not args.open_viewer:
        return None
    try:
        viewer = LiveRunViewer(
            evolution_config=evolution_config,
            world_config=world_config,
            sample_limit=max(args.live_samples, 8),
            experiment="generalize_multiwall",
            label="two walls | no absolute position",
        )
        live_url = viewer.start()
        print(f"Live visualization: {live_url}", flush=True)
        _open_in_browser(live_url)
        return viewer
    except OSError as error:
        print(f"Could not start live visualization: {error}", flush=True)
        return None


def _write_records(path: Path, record_type: type, records: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(record_type.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_report(
    path: Path,
    evolution_config: EvolutionConfig,
    training_records: list[MultiWallEvaluationRecord],
    holdout_records: list[MultiWallEvaluationRecord],
    selected_validation: MultiWallValidationRecord,
    final_validation: MultiWallValidationRecord,
    selected_stats: GenerationStats,
    validation_count: int,
    viewer_path: Path | None,
    validation_viewer_path: Path | None,
) -> None:
    training_eat_rate = _eat_rate(training_records)
    holdout_eat_rate = _eat_rate(holdout_records)
    lines = [
        "# Обобщение в комнатах с двумя стенами",
        "",
        "Абсолютные координаты `agent x/y` во всех сценах равны нулю. Агент",
        "получает направление и расстояние до еды, рекуррентное состояние и три",
        "локальных wall-сенсора.",
        "",
        "| Split | Episodes | Average score | Eat rate | Crossed both | Stuck | Avg collisions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _summary_row("training", training_records),
        (
            f"| validation | {validation_count} | "
            f"{selected_validation.average_score:.3f} | "
            f"{selected_validation.overall_eat_rate:.2%} | "
            f"{selected_validation.all_walls_crossed_rate:.2%} | "
            f"{selected_validation.stuck_rate:.2%} | "
            f"{selected_validation.average_collisions:.2f} |"
        ),
        _summary_row("holdout", holdout_records),
        "",
        (
            f"- Selected generation: {selected_stats.generation + 1}/"
            f"{evolution_config.generations} (index {selected_stats.generation})"
        ),
        f"- Generalization gap: {training_eat_rate - holdout_eat_rate:+.2%}",
        "- Holdout did not participate in checkpoint selection.",
        "",
        "## Validation checkpoint",
        "",
        "| Generation | Overall | Vertical | Horizontal | Same side | Zigzag | Crossed both | Stuck | Avg collisions |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _validation_row(selected_validation, "selected"),
        _validation_row(final_validation, "final"),
        "",
        "## Holdout categories",
        "",
        "| Orientation | Route | Episodes | Eat rate | Crossed both | Stuck | Avg collisions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for orientation in ("vertical", "horizontal"):
        for pattern in ("aligned", "alternating"):
            category = [
                item
                for item in holdout_records
                if item.orientation == orientation and item.door_pattern == pattern
            ]
            lines.append(_category_row(orientation, pattern, category))
    lines.extend(
        [
            "",
            "`Same side` означает две двери с одной стороны комнаты. `Zigzag`",
            "означает, что вторая дверь находится с противоположной стороны.",
            "Высокий `Crossed both` при низком eat rate показывает, что агент умеет",
            "проходить препятствия, но пока не завершает маршрут вовремя.",
            "",
        ]
    )
    if viewer_path is not None:
        lines.append(f"[Открыть holdout viewer]({viewer_path.name})")
        lines.append("")
    if validation_viewer_path is not None:
        lines.append(f"[Открыть validation viewer]({validation_viewer_path.name})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_row(label: str, records: list[MultiWallEvaluationRecord]) -> str:
    return (
        f"| {label} | {len(records)} | {mean(item.score for item in records):.3f} | "
        f"{_eat_rate(records):.2%} | "
        f"{mean(item.walls_crossed == WALL_COUNT for item in records):.2%} | "
        f"{mean(item.stuck for item in records):.2%} | "
        f"{mean(item.collisions for item in records):.2f} |"
    )


def _validation_row(record: MultiWallValidationRecord, label: str) -> str:
    return (
        f"| {record.generation + 1} ({label}) | "
        f"{record.overall_eat_rate:.2%} | {record.vertical_eat_rate:.2%} | "
        f"{record.horizontal_eat_rate:.2%} | {record.aligned_eat_rate:.2%} | "
        f"{record.alternating_eat_rate:.2%} | "
        f"{record.all_walls_crossed_rate:.2%} | {record.stuck_rate:.2%} | "
        f"{record.average_collisions:.2f} |"
    )


def _category_row(
    orientation: str,
    pattern: str,
    records: list[MultiWallEvaluationRecord],
) -> str:
    route = "same side" if pattern == "aligned" else "zigzag"
    return (
        f"| {orientation} | {route} | {len(records)} | "
        f"{_eat_rate(records):.2%} | "
        f"{mean(item.walls_crossed == WALL_COUNT for item in records):.2%} | "
        f"{mean(item.stuck for item in records):.2%} | "
        f"{mean(item.collisions for item in records):.2f} |"
    )


def _eat_rate(records: list[MultiWallEvaluationRecord]) -> float:
    return mean(item.ate_food for item in records)


if __name__ == "__main__":
    main()
