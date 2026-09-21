from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random
from statistics import mean, pstdev
from typing import Callable

from evolution import EvaluationProgress, EvolutionConfig, GenerationStats
from experiments.generalize_rooms import (
    BalancedRoomScenario,
    EvaluationRecord,
    GeneralizationTrainingResult,
    ValidationGenerationRecord,
    VALIDATION_STUCK_STEPS,
    _balanced_room_scenarios,
    _crossed_wall,
    _evaluate_scenarios,
    _has_collision_streak,
    _live_trace_episode_indices,
    _run_training_with_validation,
    _write_evaluations,
    _write_validation_episodes,
    _write_validation_history,
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
from world import RoomFoodWorld, RoomFoodWorldConfig


TARGET_MEAN_HOLDOUT = 0.80
TARGET_RUN_HOLDOUT = 0.70
TARGET_PASS_FRACTION = 0.80
TARGET_ORIENTATION_GAP = 0.15
TARGET_STUCK_RATE = 0.10
SHOWCASE_SEED = 9_000_007
SCENARIO_NAMESPACE_SIZE = 10_000_000
VALIDATION_SEED_OFFSET = 2_000_003
HOLDOUT_SEED_OFFSET = 4_000_003


@dataclass(frozen=True)
class GeneralizationComparisonRecord:
    replicate: int
    seed: int
    selected_generation: int
    training_average_score: float
    training_eat_rate: float
    validation_average_score: float
    validation_eat_rate: float
    validation_vertical_eat_rate: float
    validation_horizontal_eat_rate: float
    holdout_average_score: float
    holdout_eat_rate: float
    holdout_vertical_eat_rate: float
    holdout_horizontal_eat_rate: float
    holdout_orientation_gap: float
    holdout_wall_cross_rate: float
    holdout_stuck_rate: float
    holdout_average_collisions: float
    generalization_gap: float
    worst_category: str
    worst_category_eat_rate: float


@dataclass(frozen=True)
class CategoryRecord:
    replicate: int
    seed: int
    orientation: str
    start_side: str
    door_side: str
    successes: int
    episodes: int
    eat_rate: float
    average_collisions: float
    average_final_distance: float


@dataclass(frozen=True)
class ReplicateResult:
    record: GeneralizationComparisonRecord
    evolution_config: EvolutionConfig
    training_scenarios: tuple[BalancedRoomScenario, ...]
    validation_scenarios: tuple[BalancedRoomScenario, ...]
    holdout_scenarios: tuple[BalancedRoomScenario, ...]
    training: GeneralizationTrainingResult
    training_records: list[EvaluationRecord]
    holdout_records: list[EvaluationRecord]
    holdout_replays: list[ReplayEpisode]
    category_records: list[CategoryRecord]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat the balanced room experiment across independent seeds.",
    )
    add_evolution_arguments(parser, output=Path("results/compare_generalization"))
    parser.set_defaults(population=80, generations=40, episodes=24, max_steps=100)
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=24,
        help="Balanced unseen rooms used to select one generation checkpoint.",
    )
    parser.add_argument(
        "--holdout-episodes",
        type=int,
        default=64,
        help="Balanced unseen rooms opened only after checkpoint selection.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=5,
        help="Number of independent evolutionary runs.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=10,
        help="Increment between generated seeds; ignored when --seeds is used.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit independent seeds, overriding --replicates and --seed-step.",
    )
    parser.add_argument("--door-width", type=float, default=1.8)
    parser.add_argument("--wall-sensor-range", type=float, default=2.8)
    parser.add_argument("--collision-penalty", type=float, default=0.25)
    args = parser.parse_args()
    validate_evolution_arguments(parser, args)
    if args.episodes < 8 or args.episodes % 8 != 0:
        parser.error("--episodes must be a positive multiple of 8")
    if args.validation_episodes < 8 or args.validation_episodes % 8 != 0:
        parser.error("--validation-episodes must be a positive multiple of 8")
    if args.holdout_episodes < 8 or args.holdout_episodes % 8 != 0:
        parser.error("--holdout-episodes must be a positive multiple of 8")
    if args.replicates < 1:
        parser.error("--replicates must be at least 1")
    if args.seed_step < 1:
        parser.error("--seed-step must be at least 1")

    seeds = _resolve_seeds(args.seed, args.replicates, args.seed_step, args.seeds)
    if len(set(seeds)) != len(seeds):
        parser.error("comparison seeds must be distinct")

    base_config = evolution_config_from_args(args)
    world_config = RoomFoodWorldConfig(
        max_steps=args.max_steps,
        door_width=args.door_width,
        wall_sensor_range=args.wall_sensor_range,
        collision_penalty=args.collision_penalty,
    )
    run_dir = _create_run_dir(args.output)
    _write_json(
        run_dir / "config.json",
        {
            "experiment": "compare_generalization",
            "evolution": asdict(base_config),
            "world": asdict(world_config),
            "seeds": list(seeds),
            "scenario_namespace_size": SCENARIO_NAMESPACE_SIZE,
            "validation_episodes": args.validation_episodes,
            "holdout_episodes": args.holdout_episodes,
            "targets": {
                "mean_holdout_eat_rate": TARGET_MEAN_HOLDOUT,
                "run_holdout_eat_rate": TARGET_RUN_HOLDOUT,
                "passing_run_fraction": TARGET_PASS_FRACTION,
                "mean_orientation_gap": TARGET_ORIENTATION_GAP,
                "mean_stuck_rate": TARGET_STUCK_RATE,
            },
        },
    )

    print(f"Run: {run_dir}")
    print(
        "replicate,seed,generation,best_score,avg_score,best_eat_rate,avg_hidden",
        flush=True,
    )
    live_viewer = _start_live_viewer(args, base_config, world_config, seeds)
    results: list[ReplicateResult] = []
    viewer_paths: dict[int, Path] = {}
    used_scenario_seeds: set[int] = set()

    for replicate, seed in enumerate(seeds):
        evolution_config = replace(base_config, seed=seed)
        label = f"seed {seed} | repeat {replicate + 1}/{len(seeds)}"
        if live_viewer is not None and replicate > 0:
            live_viewer.begin_run(
                evolution_config,
                world_config,
                experiment="generalize_rooms",
                label=label,
            )

        def on_generation(stats: GenerationStats) -> None:
            print(
                f"{replicate},{seed},{_format_stats(stats)}",
                flush=True,
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
                f"horizontal={validation.horizontal_eat_rate:.0%}"
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

        result = _run_replicate(
            replicate=replicate,
            seed=seed,
            evolution_config=evolution_config,
            world_config=world_config,
            validation_episodes=args.validation_episodes,
            holdout_episodes=args.holdout_episodes,
            on_generation=on_generation,
            on_validation=on_validation,
            on_evaluation=(live_viewer.record_evaluation if live_viewer else None),
            trace_sample_size=args.live_samples if live_viewer else 0,
        )
        scenario_seeds = {
            scenario.seed
            for scenarios in (
                result.training_scenarios,
                result.validation_scenarios,
                result.holdout_scenarios,
            )
            for scenario in scenarios
        }
        overlap = used_scenario_seeds & scenario_seeds
        if overlap:
            raise RuntimeError(
                "Scenario seeds overlap between replicates: "
                f"{sorted(overlap)[:5]}"
            )
        used_scenario_seeds.update(scenario_seeds)
        results.append(result)
        replicate_dir = run_dir / "runs" / f"seed-{seed}"
        _write_replicate_artifacts(replicate_dir, result, world_config)

        if not args.no_visualization:
            viewer_path = write_visualization(
                run_dir=replicate_dir,
                best_genome=result.training.best_validation_genome,
                history=result.training.history,
                evolution_config=evolution_config,
                world_config=world_config,
                seed=result.holdout_replays[0].seed,
                world_class=RoomFoodWorld,
                experiment="generalize_rooms",
                replay_episodes=result.holdout_replays,
                diagnostics=[
                    asdict(item) for item in result.training.validation_history
                ],
                selected_generation=result.training.selected_stats.generation,
            )
            write_ranked_visualization(
                run_dir=replicate_dir,
                ranked_replays=result.training.selected_replays,
                history=result.training.history,
                evolution_config=evolution_config,
                world_config=world_config,
                experiment="generalize_rooms",
                label=(
                    f"seed {seed} validation checkpoint | generation "
                    f"{result.training.selected_stats.generation + 1}/"
                    f"{evolution_config.generations}"
                ),
                diagnostics=[
                    asdict(item) for item in result.training.validation_history
                ],
                viewer_filename="validation-viewer.html",
                data_filename="validation-replay.json",
                selected_stats=result.training.selected_stats,
            )
            viewer_paths[seed] = viewer_path

        print(
            f"result,{replicate},{seed},"
            f"validation={result.record.validation_eat_rate:.2%},"
            f"holdout={result.record.holdout_eat_rate:.2%},"
            f"orientation_gap={result.record.holdout_orientation_gap:.2%}",
            flush=True,
        )

    ranked_results, ranked_replays = _rank_results(
        results,
        args.live_samples,
        world_config,
    )
    winner = ranked_results[0]
    comparison_viewer_path = None
    if not args.no_visualization:
        comparison_viewer_path = write_ranked_visualization(
            run_dir=run_dir,
            ranked_replays=ranked_replays,
            history=winner.training.history,
            evolution_config=winner.evolution_config,
            world_config=world_config,
            experiment="generalize_rooms",
            label="generalization across seeds",
            diagnostics=[asdict(item) for item in winner.training.validation_history],
            selected_stats=winner.training.selected_stats,
            status_message=(
                f"Top {len(ranked_replays)} seeds ranked by holdout eat rate"
            ),
        )

    comparison_records = [result.record for result in results]
    category_records = [
        record for result in results for record in result.category_records
    ]
    _write_csv(run_dir / "comparison.csv", comparison_records)
    _write_csv(run_dir / "categories.csv", category_records)
    _write_report(
        run_dir / "report.md",
        comparison_records,
        category_records,
        viewer_paths,
        comparison_viewer_path,
    )

    if live_viewer is not None:
        live_viewer.finish(
            winner.training.selected_stats,
            ranked_replays[0].episode,
            comparison_viewer_path,
            ranked_replays=ranked_replays,
            final_label="generalization across seeds",
            message=(
                f"Comparison complete: {len(results)} independent seeds"
            ),
        )
        live_viewer.wait_until_final_served()

    print(f"Saved comparison to: {run_dir}")
    if comparison_viewer_path is not None:
        print(f"Open comparison viewer: {comparison_viewer_path}")
        if args.open_viewer and live_viewer is None:
            _open_in_browser(comparison_viewer_path.resolve().as_uri())


def _resolve_seeds(
    base_seed: int,
    replicates: int,
    seed_step: int,
    explicit_seeds: list[int] | None,
) -> tuple[int, ...]:
    if explicit_seeds:
        return tuple(explicit_seeds)
    return tuple(base_seed + index * seed_step for index in range(replicates))


def _run_replicate(
    *,
    replicate: int,
    seed: int,
    evolution_config: EvolutionConfig,
    world_config: RoomFoodWorldConfig,
    validation_episodes: int,
    holdout_episodes: int,
    on_generation: Callable[[GenerationStats], None] | None = None,
    on_validation: Callable[
        [
            GenerationStats,
            ValidationGenerationRecord,
            list[RankedReplay],
            bool,
            int,
        ],
        None,
    ]
    | None = None,
    on_evaluation: Callable[[EvaluationProgress], None] | None = None,
    trace_sample_size: int = 0,
) -> ReplicateResult:
    scenario_namespace = _scenario_namespace(seed)
    training_scenarios = _balanced_room_scenarios(
        world_config,
        count=evolution_config.episodes_per_genome,
        first_seed=scenario_namespace,
    )
    validation_scenarios = _balanced_room_scenarios(
        world_config,
        count=validation_episodes,
        first_seed=scenario_namespace + VALIDATION_SEED_OFFSET,
    )
    holdout_scenarios = _balanced_room_scenarios(
        world_config,
        count=holdout_episodes,
        first_seed=scenario_namespace + HOLDOUT_SEED_OFFSET,
    )
    all_seeds = tuple(
        scenario.seed
        for scenarios in (
            training_scenarios,
            validation_scenarios,
            holdout_scenarios,
        )
        for scenario in scenarios
    )
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("Training, validation, and holdout seeds must be disjoint.")

    training = _run_training_with_validation(
        evolution_config=evolution_config,
        world_config=world_config,
        training_scenarios=training_scenarios,
        validation_scenarios=validation_scenarios,
        on_generation=on_generation,
        on_validation=on_validation,
        on_evaluation=on_evaluation,
        trace_sample_size=trace_sample_size,
        trace_episode_indices=(
            _live_trace_episode_indices(
                tuple(item.seed for item in training_scenarios),
                trace_sample_size,
            )
            if trace_sample_size
            else None
        ),
    )
    training_records, _ = _evaluate_scenarios(
        training.best_validation_genome,
        world_config,
        tuple(item.seed for item in training_scenarios),
        split="training",
        record_replays=False,
    )
    holdout_records, holdout_replays = _evaluate_scenarios(
        training.best_validation_genome,
        world_config,
        tuple(item.seed for item in holdout_scenarios),
        split="holdout",
        record_replays=True,
    )
    category_records = _category_records(
        replicate,
        seed,
        holdout_scenarios,
        holdout_records,
    )
    record = _comparison_record(
        replicate,
        seed,
        training,
        training_records,
        holdout_scenarios,
        holdout_records,
        holdout_replays,
        category_records,
    )
    return ReplicateResult(
        record=record,
        evolution_config=evolution_config,
        training_scenarios=training_scenarios,
        validation_scenarios=validation_scenarios,
        holdout_scenarios=holdout_scenarios,
        training=training,
        training_records=training_records,
        holdout_records=holdout_records,
        holdout_replays=holdout_replays,
        category_records=category_records,
    )


def _comparison_record(
    replicate: int,
    seed: int,
    training: GeneralizationTrainingResult,
    training_records: list[EvaluationRecord],
    holdout_scenarios: tuple[BalancedRoomScenario, ...],
    holdout_records: list[EvaluationRecord],
    holdout_replays: list[ReplayEpisode],
    category_records: list[CategoryRecord],
) -> GeneralizationComparisonRecord:
    paired = list(zip(holdout_scenarios, holdout_records))
    vertical = [record for scenario, record in paired if scenario.orientation == "vertical"]
    horizontal = [
        record for scenario, record in paired if scenario.orientation == "horizontal"
    ]
    vertical_eat_rate = _eat_rate(vertical)
    horizontal_eat_rate = _eat_rate(horizontal)
    holdout_eat_rate = _eat_rate(holdout_records)
    validation = training.selected_validation
    worst = min(
        category_records,
        key=lambda item: (
            item.eat_rate,
            item.orientation,
            item.start_side,
            item.door_side,
        ),
    )
    return GeneralizationComparisonRecord(
        replicate=replicate,
        seed=seed,
        selected_generation=training.selected_stats.generation,
        training_average_score=mean(item.score for item in training_records),
        training_eat_rate=_eat_rate(training_records),
        validation_average_score=validation.average_score,
        validation_eat_rate=validation.overall_eat_rate,
        validation_vertical_eat_rate=validation.vertical_eat_rate,
        validation_horizontal_eat_rate=validation.horizontal_eat_rate,
        holdout_average_score=mean(item.score for item in holdout_records),
        holdout_eat_rate=holdout_eat_rate,
        holdout_vertical_eat_rate=vertical_eat_rate,
        holdout_horizontal_eat_rate=horizontal_eat_rate,
        holdout_orientation_gap=abs(vertical_eat_rate - horizontal_eat_rate),
        holdout_wall_cross_rate=mean(
            _crossed_wall(replay.episode) for replay in holdout_replays
        ),
        holdout_stuck_rate=mean(
            _has_collision_streak(replay.episode, VALIDATION_STUCK_STEPS)
            for replay in holdout_replays
        ),
        holdout_average_collisions=mean(item.collisions for item in holdout_records),
        generalization_gap=_eat_rate(training_records) - holdout_eat_rate,
        worst_category=_category_label(worst),
        worst_category_eat_rate=worst.eat_rate,
    )


def _category_records(
    replicate: int,
    seed: int,
    scenarios: tuple[BalancedRoomScenario, ...],
    records: list[EvaluationRecord],
) -> list[CategoryRecord]:
    if len(scenarios) != len(records):
        raise ValueError("Holdout scenarios and records must have equal lengths.")
    grouped: dict[tuple[str, str, str], list[EvaluationRecord]] = {}
    for scenario, record in zip(scenarios, records):
        key = (scenario.orientation, scenario.start_side, scenario.door_side)
        grouped.setdefault(key, []).append(record)

    result = []
    for (orientation, start_side, door_side), items in sorted(grouped.items()):
        result.append(
            CategoryRecord(
                replicate=replicate,
                seed=seed,
                orientation=orientation,
                start_side=start_side,
                door_side=door_side,
                successes=sum(item.ate_food for item in items),
                episodes=len(items),
                eat_rate=_eat_rate(items),
                average_collisions=mean(item.collisions for item in items),
                average_final_distance=mean(item.final_distance for item in items),
            )
        )
    return result


def _rank_results(
    results: list[ReplicateResult],
    limit: int,
    world_config: RoomFoodWorldConfig,
) -> tuple[list[ReplicateResult], list[RankedReplay]]:
    ranked = sorted(
        results,
        key=lambda item: (
            item.record.holdout_eat_rate,
            item.record.holdout_average_score,
        ),
        reverse=True,
    )[:limit]
    replays = []
    for rank, result in enumerate(ranked, start=1):
        world = RoomFoodWorld(world_config)
        episode = world.evaluate(
            result.training.best_validation_genome,
            random.Random(SHOWCASE_SEED),
            record_trace=True,
        )
        replays.append(
            RankedReplay(
                rank=rank,
                score=result.record.holdout_average_score,
                eat_rate=result.record.holdout_eat_rate,
                episode=episode,
                label=f"seed {result.record.seed}",
                metric="holdout",
            )
        )
    return ranked, replays


def _write_replicate_artifacts(
    run_dir: Path,
    result: ReplicateResult,
    world_config: RoomFoodWorldConfig,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "config.json",
        {
            "experiment": "generalize_rooms",
            "replicate": result.record.replicate,
            "evolution": asdict(result.evolution_config),
            "world": asdict(world_config),
            "training_scenarios": [asdict(item) for item in result.training_scenarios],
            "validation_scenarios": [
                asdict(item) for item in result.validation_scenarios
            ],
            "holdout_scenarios": [asdict(item) for item in result.holdout_scenarios],
        },
    )
    _write_metrics(run_dir / "metrics.csv", result.training.history)
    _write_validation_history(
        run_dir / "validation.csv",
        result.training.validation_history,
    )
    _write_validation_episodes(
        run_dir / "validation_episodes.csv",
        result.training.validation_episode_records,
    )
    _write_evaluations(
        run_dir / "evaluation.csv",
        result.training_records + result.holdout_records,
    )
    _write_json(
        run_dir / "selection.json",
        {
            "selected_generation": result.training.selected_stats.generation,
            "training_metrics_at_selection": asdict(result.training.selected_stats),
            "validation_metrics_at_selection": asdict(
                result.training.selected_validation
            ),
            "holdout_used_for_selection": False,
        },
    )
    result.training.best_validation_genome.save_json(run_dir / "best_genome.json")
    result.training.best_validation_genome.save_json(
        run_dir / "best_validation_genome.json"
    )
    result.training.best_training_genome.save_json(
        run_dir / "best_training_genome.json"
    )


def _write_csv(path: Path, records: list[object]) -> None:
    if not records:
        raise ValueError("Cannot write an empty comparison CSV.")
    fieldnames = list(type(records[0]).__annotations__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_report(
    path: Path,
    records: list[GeneralizationComparisonRecord],
    categories: list[CategoryRecord],
    viewer_paths: dict[int, Path],
    comparison_viewer_path: Path | None,
) -> None:
    holdout_rates = [item.holdout_eat_rate for item in records]
    orientation_gaps = [item.holdout_orientation_gap for item in records]
    stuck_rates = [item.holdout_stuck_rate for item in records]
    required_passes = math.ceil(len(records) * TARGET_PASS_FRACTION)
    passing_runs = sum(rate >= TARGET_RUN_HOLDOUT for rate in holdout_rates)
    checks = {
        "mean_holdout": mean(holdout_rates) >= TARGET_MEAN_HOLDOUT,
        "passing_runs": passing_runs >= required_passes,
        "orientation_gap": mean(orientation_gaps) <= TARGET_ORIENTATION_GAP,
        "stuck_rate": mean(stuck_rates) <= TARGET_STUCK_RATE,
    }
    passed = all(checks.values())
    lines = [
        "# Обобщение на разных seed",
        "",
        f"**Вердикт: {'ЭТАП ПРОЙДЕН' if passed else 'НУЖНЫ ЕЩЕ УЛУЧШЕНИЯ'}**",
        "",
        "Каждая строка использует независимую начальную популяцию и три",
        "непересекающихся набора training, validation и holdout комнат.",
        "",
        "## Сводка",
        "",
        "| Метрика | Значение | Цель | Результат |",
        "| --- | ---: | ---: | --- |",
        _target_row(
            "Mean holdout eat rate",
            mean(holdout_rates),
            f">= {TARGET_MEAN_HOLDOUT:.0%}",
            checks["mean_holdout"],
        ),
        (
            f"| Runs above {TARGET_RUN_HOLDOUT:.0%} | {passing_runs}/{len(records)} | "
            f">= {required_passes}/{len(records)} | {_status(checks['passing_runs'])} |"
        ),
        _target_row(
            "Mean orientation gap",
            mean(orientation_gaps),
            f"<= {TARGET_ORIENTATION_GAP:.0%}",
            checks["orientation_gap"],
        ),
        _target_row(
            "Mean stuck rate",
            mean(stuck_rates),
            f"<= {TARGET_STUCK_RATE:.0%}",
            checks["stuck_rate"],
        ),
        "",
        (
            f"Holdout eat rate: mean {mean(holdout_rates):.2%}, "
            f"standard deviation {pstdev(holdout_rates):.2%}, "
            f"range {min(holdout_rates):.2%}-{max(holdout_rates):.2%}."
        ),
        "",
        "## Запуски",
        "",
        "| Seed | Checkpoint | Training | Validation | Holdout | Vertical | Horizontal | Gap | Stuck |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in sorted(records, key=lambda item: item.seed):
        lines.append(
            f"| {record.seed} | {record.selected_generation + 1} | "
            f"{record.training_eat_rate:.2%} | {record.validation_eat_rate:.2%} | "
            f"{record.holdout_eat_rate:.2%} | "
            f"{record.holdout_vertical_eat_rate:.2%} | "
            f"{record.holdout_horizontal_eat_rate:.2%} | "
            f"{record.holdout_orientation_gap:.2%} | "
            f"{record.holdout_stuck_rate:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Категории holdout",
            "",
            "| Orientation | Start | Door | Success | Eat rate | Avg collisions |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in _aggregate_categories(categories):
        lines.append(
            f"| {row['orientation']} | {row['start_side']} | {row['door_side']} | "
            f"{row['successes']}/{row['episodes']} | {row['eat_rate']:.2%} | "
            f"{row['average_collisions']:.2f} |"
        )

    if comparison_viewer_path is not None:
        relative = comparison_viewer_path.relative_to(path.parent).as_posix()
        lines.extend(["", "## Viewer", "", f"[Лучшие seed]({relative})", ""])
    if viewer_paths:
        lines.extend(["## Отдельные запуски", ""])
        for seed, viewer_path in sorted(viewer_paths.items()):
            relative = viewer_path.relative_to(path.parent).as_posix()
            lines.append(f"- Seed {seed}: [{relative}]({relative})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate_categories(categories: list[CategoryRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[CategoryRecord]] = {}
    for record in categories:
        key = (record.orientation, record.start_side, record.door_side)
        grouped.setdefault(key, []).append(record)

    rows = []
    for (orientation, start_side, door_side), items in sorted(grouped.items()):
        successes = sum(item.successes for item in items)
        episodes = sum(item.episodes for item in items)
        rows.append(
            {
                "orientation": orientation,
                "start_side": start_side,
                "door_side": door_side,
                "successes": successes,
                "episodes": episodes,
                "eat_rate": successes / episodes,
                "average_collisions": (
                    sum(item.average_collisions * item.episodes for item in items)
                    / episodes
                ),
            }
        )
    return rows


def _target_row(name: str, value: float, target: str, passed: bool) -> str:
    return f"| {name} | {value:.2%} | {target} | {_status(passed)} |"


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _category_label(record: CategoryRecord) -> str:
    return f"{record.orientation}|start={record.start_side}|door={record.door_side}"


def _scenario_namespace(seed: int) -> int:
    """Map every signed evolution seed to a disjoint non-negative seed block."""
    namespace_index = seed * 2 if seed >= 0 else -seed * 2 - 1
    return namespace_index * SCENARIO_NAMESPACE_SIZE


def _eat_rate(records: list[EvaluationRecord]) -> float:
    return sum(item.ate_food for item in records) / len(records)


def _start_live_viewer(
    args: argparse.Namespace,
    evolution_config: EvolutionConfig,
    world_config: RoomFoodWorldConfig,
    seeds: tuple[int, ...],
) -> LiveRunViewer | None:
    if not args.open_viewer:
        return None
    try:
        viewer = LiveRunViewer(
            evolution_config=evolution_config,
            world_config=world_config,
            sample_limit=max(args.live_samples, 8),
            experiment="generalize_rooms",
            label=f"seed {seeds[0]} | repeat 1/{len(seeds)}",
        )
        live_url = viewer.start()
        print(f"Live visualization: {live_url}", flush=True)
        _open_in_browser(live_url)
        return viewer
    except OSError as error:
        print(f"Could not start live visualization: {error}", flush=True)
        return None


if __name__ == "__main__":
    main()
