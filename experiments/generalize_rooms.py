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
    RoomFoodWorld,
    RoomFoodWorldConfig,
    RoomScenarioSignature,
)


VALIDATION_STUCK_STEPS = 8


@dataclass(frozen=True)
class EvaluationRecord:
    split: str
    episode: int
    seed: int
    score: float
    ate_food: bool
    steps: int
    final_distance: float
    collisions: int


@dataclass(frozen=True)
class BalancedRoomScenario:
    index: int
    seed: int
    orientation: str
    start_side: str
    door_side: str

    @property
    def label(self) -> str:
        orientation = "V" if self.orientation == "vertical" else "H"
        start = "+" if self.start_side == "positive" else "-"
        door = "+" if self.door_side == "positive" else "-"
        return f"{orientation} | start {start} | door {door}"


@dataclass(frozen=True)
class ValidationGenerationRecord:
    generation: int
    average_score: float
    overall_eat_rate: float
    vertical_eat_rate: float
    horizontal_eat_rate: float
    wall_cross_rate: float
    stuck_rate: float
    average_collisions: float


@dataclass(frozen=True)
class ValidationEpisodeRecord:
    generation: int
    scenario: int
    seed: int
    orientation: str
    start_side: str
    door_side: str
    score: float
    ate_food: bool
    crossed_wall: bool
    stuck: bool
    collisions: int
    steps: int


@dataclass(frozen=True)
class GeneralizationTrainingResult:
    best_training_genome: BrainGenome
    best_validation_genome: BrainGenome
    history: list[GenerationStats]
    validation_history: list[ValidationGenerationRecord]
    validation_episode_records: list[ValidationEpisodeRecord]
    selected_validation: ValidationGenerationRecord
    selected_stats: GenerationStats
    selected_replays: list[RankedReplay]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train in two-room layouts and test on unseen door positions.",
    )
    add_evolution_arguments(parser, output=Path("results/generalize_rooms"))
    parser.set_defaults(population=80, generations=40, episodes=24, max_steps=100)
    parser.add_argument("--validation-episodes", type=int, default=24)
    parser.add_argument("--holdout-episodes", type=int, default=64)
    parser.add_argument("--door-width", type=float, default=1.8)
    parser.add_argument("--wall-sensor-range", type=float, default=2.8)
    parser.add_argument("--collision-penalty", type=float, default=0.25)
    args = parser.parse_args()
    validate_evolution_arguments(parser, args)
    if args.episodes < 8 or args.episodes % 8 != 0:
        parser.error("--episodes must be a positive multiple of 8")
    if args.validation_episodes < 8 or args.validation_episodes % 8 != 0:
        parser.error("--validation-episodes must be a positive multiple of 8")
    if args.holdout_episodes < 1:
        parser.error("--holdout-episodes must be at least 1")

    evolution_config = evolution_config_from_args(args)
    world_config = RoomFoodWorldConfig(
        max_steps=args.max_steps,
        door_width=args.door_width,
        wall_sensor_range=args.wall_sensor_range,
        collision_penalty=args.collision_penalty,
    )
    run_dir = _create_run_dir(args.output)
    training_scenarios = _balanced_room_scenarios(
        world_config,
        count=evolution_config.episodes_per_genome,
        first_seed=evolution_config.seed,
    )
    validation_scenarios = _balanced_room_scenarios(
        world_config,
        count=args.validation_episodes,
        first_seed=evolution_config.seed + 2_000_003,
    )
    holdout_scenarios = _balanced_room_scenarios(
        world_config,
        count=args.holdout_episodes,
        first_seed=evolution_config.seed + 4_000_003,
    )
    training_seeds = tuple(scenario.seed for scenario in training_scenarios)
    validation_seeds = tuple(scenario.seed for scenario in validation_scenarios)
    holdout_seeds = tuple(scenario.seed for scenario in holdout_scenarios)
    all_seeds = training_seeds + validation_seeds + holdout_seeds
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("Training, validation, and holdout seeds must be disjoint.")
    _write_json(
        run_dir / "config.json",
        {
            "experiment": "generalize_rooms",
            "evolution": asdict(evolution_config),
            "world": asdict(world_config),
            "selection": {
                "primary": "validation_eat_rate",
                "tie_breaker": "validation_average_score",
            },
            "training_scenarios": [asdict(item) for item in training_scenarios],
            "validation_scenarios": [
                asdict(item) for item in validation_scenarios
            ],
            "holdout_scenarios": [asdict(item) for item in holdout_scenarios],
            "training_seeds": list(training_seeds),
            "validation_seeds": list(validation_seeds),
            "holdout_seeds": list(holdout_seeds),
        },
    )

    print(f"Run: {run_dir}")
    live_viewer = _start_live_viewer(args, evolution_config, world_config)
    print("generation,best_score,avg_score,best_eat_rate,avg_hidden", flush=True)

    def on_generation(stats: GenerationStats) -> None:
        print(_format_stats(stats), flush=True)

    trace_episode_indices = _live_trace_episode_indices(
        training_seeds,
        args.live_samples,
    )

    def on_validation(
        stats: GenerationStats,
        validation: ValidationGenerationRecord,
        replays: list[RankedReplay],
        selected: bool,
        selected_generation: int,
    ) -> None:
        print(
            "  validation: "
            f"eat={validation.overall_eat_rate:.0%}, "
            f"vertical={validation.vertical_eat_rate:.0%}, "
            f"horizontal={validation.horizontal_eat_rate:.0%}, "
            f"crossed={validation.wall_cross_rate:.0%}, "
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
        trace_episode_indices=trace_episode_indices if live_viewer else None,
    )
    best_training_genome = training_result.best_training_genome
    best_genome = training_result.best_validation_genome
    history = training_result.history
    validation_history = training_result.validation_history
    validation_episode_records = training_result.validation_episode_records
    best_validation_record = training_result.selected_validation
    best_validation_stats = training_result.selected_stats
    best_validation_replays = training_result.selected_replays

    training_records, _ = _evaluate_scenarios(
        best_genome,
        world_config,
        training_seeds,
        split="training",
        record_replays=False,
    )
    holdout_records, holdout_replays = _evaluate_scenarios(
        best_genome,
        world_config,
        holdout_seeds,
        split="holdout",
        record_replays=True,
    )

    _write_metrics(run_dir / "metrics.csv", history)
    _write_evaluations(run_dir / "evaluation.csv", training_records + holdout_records)
    _write_validation_history(run_dir / "validation.csv", validation_history)
    _write_validation_episodes(
        run_dir / "validation_episodes.csv",
        validation_episode_records,
    )
    best_genome.save_json(run_dir / "best_genome.json")
    best_genome.save_json(run_dir / "best_validation_genome.json")
    best_training_genome.save_json(run_dir / "best_training_genome.json")
    _write_json(
        run_dir / "selection.json",
        {
            "selected_generation": best_validation_stats.generation,
            "training_metrics_at_selection": asdict(best_validation_stats),
            "validation_metrics_at_selection": asdict(best_validation_record),
            "rule": ["maximum validation eat rate", "maximum validation score"],
            "holdout_used_for_selection": False,
        },
    )

    viewer_path = None
    validation_viewer_path = None
    validation_payload = [asdict(record) for record in validation_history]
    if not args.no_visualization:
        viewer_path = write_visualization(
            run_dir=run_dir,
            best_genome=best_genome,
            history=history,
            evolution_config=evolution_config,
            world_config=world_config,
            seed=holdout_seeds[0],
            world_class=RoomFoodWorld,
            experiment="generalize_rooms",
            replay_episodes=holdout_replays,
            diagnostics=validation_payload,
            selected_generation=best_validation_stats.generation,
        )
        validation_viewer_path = write_ranked_visualization(
            run_dir=run_dir,
            ranked_replays=best_validation_replays,
            history=history,
            evolution_config=evolution_config,
            world_config=world_config,
            experiment="generalize_rooms",
            label=(
                "validation checkpoint | generation "
                f"{best_validation_stats.generation + 1}/{evolution_config.generations}"
            ),
            diagnostics=validation_payload,
            viewer_filename="validation-viewer.html",
            data_filename="validation-replay.json",
            selected_stats=best_validation_stats,
        )
    _write_report(
        run_dir / "report.md",
        evolution_config,
        training_records,
        holdout_records,
        best_validation_record,
        best_validation_stats,
        validation_history[-1],
        len(validation_scenarios),
        viewer_path,
        validation_viewer_path,
    )

    if live_viewer is not None:
        best_episode = best_validation_replays[0].episode
        live_viewer.finish(
            best_validation_stats,
            best_episode,
            validation_viewer_path or viewer_path,
            ranked_replays=best_validation_replays,
            message=(
                "Training complete: validation checkpoint from generation "
                f"{best_validation_stats.generation + 1}/"
                f"{evolution_config.generations}"
            ),
        )
        live_viewer.wait_until_final_served()

    training_eat_rate = _eat_rate(training_records)
    holdout_eat_rate = _eat_rate(holdout_records)
    print()
    print(f"Training eat rate: {training_eat_rate:.2%}")
    print(
        "Selected validation generation: "
        f"{best_validation_stats.generation + 1}/{evolution_config.generations} "
        f"(index {best_validation_stats.generation})"
    )
    print(f"Validation eat rate: {best_validation_record.overall_eat_rate:.2%}")
    print(f"Holdout eat rate: {holdout_eat_rate:.2%}")
    print(f"Generalization gap: {training_eat_rate - holdout_eat_rate:+.2%}")
    print(f"Saved results to: {run_dir}")
    if viewer_path is not None:
        print(f"Open visualization: {viewer_path}")
        print(f"Open validation: {validation_viewer_path}")
        if args.open_viewer and live_viewer is None:
            _open_in_browser(viewer_path.resolve().as_uri())


def _run_training_with_validation(
    *,
    evolution_config: EvolutionConfig,
    world_config: RoomFoodWorldConfig,
    training_scenarios: tuple[BalancedRoomScenario, ...],
    validation_scenarios: tuple[BalancedRoomScenario, ...],
    on_generation: Callable[[GenerationStats], None] | None = None,
    on_validation: (
        Callable[
            [
                GenerationStats,
                ValidationGenerationRecord,
                list[RankedReplay],
                bool,
                int,
            ],
            None,
        ]
        | None
    ) = None,
    on_evaluation: Callable[[EvaluationProgress], None] | None = None,
    trace_sample_size: int = 0,
    trace_episode_indices: tuple[int, ...] | None = None,
) -> GeneralizationTrainingResult:
    """Train on one scenario set and select a checkpoint on another."""
    training_seeds = tuple(scenario.seed for scenario in training_scenarios)
    validation_history: list[ValidationGenerationRecord] = []
    validation_episode_records: list[ValidationEpisodeRecord] = []
    selected_genome: BrainGenome | None = None
    selected_validation: ValidationGenerationRecord | None = None
    selected_stats: GenerationStats | None = None
    selected_replays: list[RankedReplay] = []
    selected_key: tuple[float, float] | None = None

    def on_generation_ranked(
        stats: GenerationStats,
        candidates: list[RankedCandidate],
    ) -> None:
        nonlocal selected_genome
        nonlocal selected_validation
        nonlocal selected_stats
        nonlocal selected_key

        validation, replays, episode_records = _evaluate_validation(
            candidates[0].genome,
            world_config,
            validation_scenarios,
            generation=stats.generation,
        )
        validation_history.append(validation)
        validation_episode_records.extend(episode_records)
        validation_key = _validation_selection_key(validation)
        selected = selected_key is None or validation_key > selected_key
        if selected:
            selected_key = validation_key
            selected_genome = candidates[0].genome.clone()
            selected_validation = validation
            selected_stats = stats
            selected_replays.clear()
            selected_replays.extend(
                _representative_validation_replays(replays, validation_scenarios)
            )
        if on_validation is not None:
            if selected_stats is None:
                raise RuntimeError("Validation callback ran before checkpoint selection.")
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
        world_class=RoomFoodWorld,
        trace_episode_indices=trace_episode_indices,
        episode_seeds=training_seeds,
    )
    if (
        selected_genome is None
        or selected_validation is None
        or selected_stats is None
    ):
        raise RuntimeError("Evolution completed without a validation checkpoint.")
    return GeneralizationTrainingResult(
        best_training_genome=best_training_genome,
        best_validation_genome=selected_genome,
        history=history,
        validation_history=validation_history,
        validation_episode_records=validation_episode_records,
        selected_validation=selected_validation,
        selected_stats=selected_stats,
        selected_replays=selected_replays,
    )


def _start_live_viewer(
    args: argparse.Namespace,
    evolution_config: EvolutionConfig,
    world_config: RoomFoodWorldConfig,
) -> LiveRunViewer | None:
    if not args.open_viewer:
        return None
    try:
        viewer = LiveRunViewer(
            evolution_config=evolution_config,
            world_config=world_config,
            sample_limit=max(args.live_samples, 8),
            experiment="generalize_rooms",
        )
        live_url = viewer.start()
        print(f"Live visualization: {live_url}", flush=True)
        _open_in_browser(live_url)
        return viewer
    except OSError as error:
        print(f"Could not start live visualization: {error}", flush=True)
        return None


def _balanced_room_scenarios(
    world_config: RoomFoodWorldConfig,
    *,
    count: int,
    first_seed: int,
) -> tuple[BalancedRoomScenario, ...]:
    """Find deterministic seeds with balanced categorical room factors."""
    if count < 1:
        raise ValueError("A balanced room set requires at least one scenario.")
    wanted = tuple(
        RoomScenarioSignature(orientation, start_side, door_side)
        for orientation, start_side, door_side in (
            ("vertical", "negative", "negative"),
            ("horizontal", "positive", "positive"),
            ("vertical", "negative", "positive"),
            ("horizontal", "positive", "negative"),
            ("vertical", "positive", "negative"),
            ("horizontal", "negative", "positive"),
            ("vertical", "positive", "positive"),
            ("horizontal", "negative", "negative"),
        )
    )
    base_count, remainder = divmod(count, len(wanted))
    quotas = {
        signature: base_count + (index < remainder)
        for index, signature in enumerate(wanted)
    }
    world = RoomFoodWorld(world_config)
    found: dict[RoomScenarioSignature, list[int]] = {
        signature: [] for signature in wanted
    }
    seed = first_seed
    while sum(len(seeds) for seeds in found.values()) < count:
        signature = world.scenario_signature_for_seed(seed)
        if signature in found and len(found[signature]) < quotas[signature]:
            found[signature].append(seed)
        seed += 1
        if seed - first_seed > max(100_000, count * 10_000):
            raise RuntimeError("Could not build a balanced room scenario set.")

    scenarios = []
    for round_index in range(max(quotas.values())):
        for signature in wanted:
            if round_index >= quotas[signature]:
                continue
            scenarios.append(
                BalancedRoomScenario(
                    index=len(scenarios),
                    seed=found[signature][round_index],
                    orientation=signature.orientation,
                    start_side=signature.start_side,
                    door_side=signature.door_side,
                )
            )
    return tuple(scenarios)


def _representative_validation_replays(
    replays: list[RankedReplay],
    scenarios: tuple[BalancedRoomScenario, ...],
) -> list[RankedReplay]:
    """Return one replay for each categorical combination shown in the viewer."""
    if len(replays) != len(scenarios):
        raise ValueError("Validation replays and scenarios must have equal lengths.")
    selected = []
    seen: set[tuple[str, str, str]] = set()
    for replay, scenario in zip(replays, scenarios):
        signature = (
            scenario.orientation,
            scenario.start_side,
            scenario.door_side,
        )
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(replay)
    return selected


def _evaluate_validation(
    genome: BrainGenome,
    world_config: RoomFoodWorldConfig,
    scenarios: tuple[BalancedRoomScenario, ...],
    *,
    generation: int,
) -> tuple[
    ValidationGenerationRecord,
    list[RankedReplay],
    list[ValidationEpisodeRecord],
]:
    world = RoomFoodWorld(world_config)
    outcomes: list[tuple[BalancedRoomScenario, EpisodeResult, bool, bool]] = []
    replays = []
    episode_records = []
    for scenario in scenarios:
        episode = world.evaluate(
            genome,
            random.Random(scenario.seed),
            record_trace=True,
        )
        crossed_wall = _crossed_wall(episode)
        stuck = _has_collision_streak(episode, VALIDATION_STUCK_STEPS)
        outcomes.append((scenario, episode, crossed_wall, stuck))
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
            ValidationEpisodeRecord(
                generation=generation,
                scenario=scenario.index,
                seed=scenario.seed,
                orientation=scenario.orientation,
                start_side=scenario.start_side,
                door_side=scenario.door_side,
                score=episode.score,
                ate_food=episode.ate_food,
                crossed_wall=crossed_wall,
                stuck=stuck,
                collisions=episode.collisions,
                steps=episode.steps,
            )
        )

    vertical = [item for item in outcomes if item[0].orientation == "vertical"]
    horizontal = [item for item in outcomes if item[0].orientation == "horizontal"]
    validation = ValidationGenerationRecord(
        generation=generation,
        average_score=mean(item[1].score for item in outcomes),
        overall_eat_rate=mean(item[1].ate_food for item in outcomes),
        vertical_eat_rate=mean(item[1].ate_food for item in vertical),
        horizontal_eat_rate=mean(item[1].ate_food for item in horizontal),
        wall_cross_rate=mean(item[2] for item in outcomes),
        stuck_rate=mean(item[3] for item in outcomes),
        average_collisions=mean(item[1].collisions for item in outcomes),
    )
    return validation, replays, episode_records


def _validation_selection_key(
    record: ValidationGenerationRecord,
) -> tuple[float, float]:
    """Rank checkpoints without looking at holdout results."""
    return record.overall_eat_rate, record.average_score


def _crossed_wall(episode: EpisodeResult) -> bool:
    if not episode.trace or not episode.obstacles:
        return False
    obstacle = episode.obstacles[0]
    vertical = (
        obstacle.y_max - obstacle.y_min > obstacle.x_max - obstacle.x_min
    )
    wall_position = (
        (obstacle.x_min + obstacle.x_max) / 2
        if vertical
        else (obstacle.y_min + obstacle.y_max) / 2
    )
    start = episode.trace[0]
    start_axis = (
        start.agent_x - wall_position
        if vertical
        else start.agent_y - wall_position
    )
    return any(
        (
            step.agent_x - wall_position
            if vertical
            else step.agent_y - wall_position
        )
        * start_axis
        < 0
        for step in episode.trace
    )


def _has_collision_streak(episode: EpisodeResult, threshold: int) -> bool:
    streak = 0
    for step in episode.trace or []:
        streak = streak + 1 if step.collided else 0
        if streak >= threshold:
            return True
    return False


def _live_trace_episode_indices(
    seeds: tuple[int, ...],
    sample_count: int,
) -> tuple[int, ...]:
    """Choose representative training rooms for the live candidate tiles."""
    pools = {"vertical": [], "horizontal": []}
    for index, seed in enumerate(seeds):
        orientation = RoomFoodWorld.layout_orientation_for_seed(seed)
        pools[orientation].append(index)

    selected: list[int] = []
    for sample_index in range(sample_count):
        preferred = "vertical" if sample_index % 2 == 0 else "horizontal"
        fallback = "horizontal" if preferred == "vertical" else "vertical"
        if pools[preferred]:
            selected.append(pools[preferred].pop(0))
        elif pools[fallback]:
            selected.append(pools[fallback].pop(0))
        else:
            selected.append(selected[sample_index % len(selected)] if selected else 0)
    return tuple(selected)


def _evaluate_scenarios(
    genome: BrainGenome,
    world_config: RoomFoodWorldConfig,
    seeds: tuple[int, ...],
    *,
    split: str,
    record_replays: bool,
) -> tuple[list[EvaluationRecord], list[ReplayEpisode]]:
    world = RoomFoodWorld(world_config)
    records = []
    replays = []
    for index, seed in enumerate(seeds):
        result = world.evaluate(
            genome,
            random.Random(seed),
            record_trace=record_replays,
        )
        records.append(_evaluation_record(split, index, seed, result))
        if record_replays:
            replays.append(ReplayEpisode(index=index, seed=seed, episode=result))
    return records, replays


def _evaluation_record(
    split: str,
    index: int,
    seed: int,
    result: EpisodeResult,
) -> EvaluationRecord:
    return EvaluationRecord(
        split=split,
        episode=index,
        seed=seed,
        score=result.score,
        ate_food=result.ate_food,
        steps=result.steps,
        final_distance=result.final_distance,
        collisions=result.collisions,
    )


def _write_evaluations(path: Path, records: list[EvaluationRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(EvaluationRecord.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_validation_history(
    path: Path,
    records: list[ValidationGenerationRecord],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(ValidationGenerationRecord.__annotations__),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_validation_episodes(
    path: Path,
    records: list[ValidationEpisodeRecord],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(ValidationEpisodeRecord.__annotations__),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_report(
    path: Path,
    evolution_config: EvolutionConfig,
    training_records: list[EvaluationRecord],
    holdout_records: list[EvaluationRecord],
    selected_validation: ValidationGenerationRecord,
    selected_stats: GenerationStats,
    final_validation: ValidationGenerationRecord,
    validation_scenario_count: int,
    viewer_path: Path | None,
    validation_viewer_path: Path | None,
) -> None:
    training_score = mean(record.score for record in training_records)
    holdout_score = mean(record.score for record in holdout_records)
    training_eat_rate = _eat_rate(training_records)
    holdout_eat_rate = _eat_rate(holdout_records)
    lines = [
        "# Обобщение между комнатами",
        "",
        "Training определял родителей и мутации. После каждого поколения его",
        "победитель проверялся на отдельном validation-наборе. Holdout был открыт",
        "только после выбора validation-checkpoint.",
        "",
        "| Split | Episodes | Average score | Eat rate | Avg collisions |",
        "| --- | ---: | ---: | ---: | ---: |",
        _summary_row("training", training_records),
        _validation_summary_row(
            "validation",
            validation_scenario_count,
            selected_validation,
        ),
        _summary_row("holdout", holdout_records),
        "",
        (
            f"- Selected generation: {selected_stats.generation + 1}/"
            f"{evolution_config.generations} (index {selected_stats.generation})"
        ),
        f"- Final generation: {evolution_config.generations}/{evolution_config.generations}",
        f"- Training score: {training_score:.3f}",
        f"- Validation score: {selected_validation.average_score:.3f}",
        f"- Holdout score: {holdout_score:.3f}",
        f"- Eat-rate gap: {training_eat_rate - holdout_eat_rate:+.2%}",
        f"- Training scenarios: {evolution_config.episodes_per_genome}",
        f"- Validation scenarios: {validation_scenario_count}",
        "",
        "`best_genome.json` и `best_validation_genome.json` содержат выбранный",
        "checkpoint. `best_training_genome.json` сохранен отдельно и не использовался",
        "для выбора по holdout.",
        "",
        "## Balanced validation",
        "",
        "Checkpoint выбран сначала по максимальному validation eat rate, затем по",
        "максимальному validation score. Validation не менял родителей, но выбирал",
        "поколение для финального holdout.",
        "",
        "| Generation | Score | Overall | Vertical | Horizontal | Crossed wall | Stuck | Avg collisions |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _validation_detail_row(selected_validation, "selected"),
        _validation_detail_row(final_validation, "final"),
        "",
        "Каждый полный блок из восьми комнат покрывает все сочетания ориентации",
        "стены, стороны старта и стороны двери. `Stuck` означает не менее восьми",
        "последовательных шагов со столкновением.",
        "",
        "Маленький разрыв между training и holdout при высоком eat rate означает,",
        "что поведение переносится на новые двери. Высокий training при низком",
        "holdout указывает на запоминание ограниченного набора сцен.",
        "",
    ]
    if viewer_path is not None:
        relative_path = viewer_path.relative_to(path.parent).as_posix()
        lines.extend([f"[Открыть holdout viewer]({relative_path})", ""])
    if validation_viewer_path is not None:
        relative_path = validation_viewer_path.relative_to(path.parent).as_posix()
        lines.extend([f"[Открыть validation viewer]({relative_path})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _validation_summary_row(
    split: str,
    episodes: int,
    record: ValidationGenerationRecord,
) -> str:
    return (
        f"| {split} | {episodes} | {record.average_score:.3f} | "
        f"{record.overall_eat_rate:.2%} | {record.average_collisions:.2f} |"
    )


def _validation_detail_row(
    record: ValidationGenerationRecord,
    label: str,
) -> str:
    return (
        f"| {record.generation + 1} ({label}) | {record.average_score:.3f} | "
        f"{record.overall_eat_rate:.2%} | "
        f"{record.vertical_eat_rate:.2%} | "
        f"{record.horizontal_eat_rate:.2%} | "
        f"{record.wall_cross_rate:.2%} | "
        f"{record.stuck_rate:.2%} | "
        f"{record.average_collisions:.2f} |"
    )


def _summary_row(split: str, records: list[EvaluationRecord]) -> str:
    return (
        f"| {split} | {len(records)} | "
        f"{mean(record.score for record in records):.3f} | "
        f"{_eat_rate(records):.2%} | "
        f"{mean(record.collisions for record in records):.2f} |"
    )


def _eat_rate(records: list[EvaluationRecord]) -> float:
    return sum(record.ate_food for record in records) / len(records)


if __name__ == "__main__":
    main()
