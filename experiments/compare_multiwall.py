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
from experiments.generalize_multiwall import (
    WALL_COUNT,
    BalancedMultiWallScenario,
    MultiWallEvaluationRecord,
    MultiWallTrainingResult,
    MultiWallValidationEpisodeRecord,
    MultiWallValidationRecord,
    _balanced_multiwall_scenarios,
    _evaluate_scenarios,
    _live_trace_episode_indices,
    _run_training_with_validation,
    _write_records,
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
from world import MultiWallFoodWorld, MultiWallFoodWorldConfig


TARGET_MEAN_HOLDOUT = 0.80
TARGET_RUN_HOLDOUT = 0.70
TARGET_PASS_FRACTION = 0.80
TARGET_ORIENTATION_GAP = 0.15
TARGET_ROUTE_GAP = 0.20
TARGET_WALL_CROSS_RATE = 0.85
TARGET_STUCK_RATE = 0.15
SHOWCASE_SEED = 9_000_007
SCENARIO_NAMESPACE_SIZE = 10_000_000
VALIDATION_SEED_OFFSET = 2_000_003
HOLDOUT_SEED_OFFSET = 4_000_003


@dataclass(frozen=True)
class MultiWallComparisonRecord:
    replicate: int
    seed: int
    selected_generation: int
    training_average_score: float
    training_eat_rate: float
    validation_average_score: float
    validation_eat_rate: float
    holdout_average_score: float
    holdout_eat_rate: float
    holdout_vertical_eat_rate: float
    holdout_horizontal_eat_rate: float
    holdout_orientation_gap: float
    holdout_aligned_eat_rate: float
    holdout_alternating_eat_rate: float
    holdout_route_gap: float
    holdout_all_walls_crossed_rate: float
    holdout_stuck_rate: float
    holdout_average_collisions: float
    generalization_gap: float
    worst_category: str
    worst_category_eat_rate: float


@dataclass(frozen=True)
class MultiWallCategoryRecord:
    replicate: int
    seed: int
    orientation: str
    start_side: str
    first_door_side: str
    door_pattern: str
    successes: int
    crossed_both: int
    episodes: int
    eat_rate: float
    cross_rate: float
    average_collisions: float
    average_final_distance: float


@dataclass(frozen=True)
class MultiWallReplicateResult:
    record: MultiWallComparisonRecord
    evolution_config: EvolutionConfig
    training_scenarios: tuple[BalancedMultiWallScenario, ...]
    validation_scenarios: tuple[BalancedMultiWallScenario, ...]
    holdout_scenarios: tuple[BalancedMultiWallScenario, ...]
    training: MultiWallTrainingResult
    training_records: list[MultiWallEvaluationRecord]
    holdout_records: list[MultiWallEvaluationRecord]
    holdout_replays: list[ReplayEpisode]
    category_records: list[MultiWallCategoryRecord]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat the two-wall experiment across independent seeds.",
    )
    add_evolution_arguments(parser, output=Path("results/compare_multiwall"))
    parser.set_defaults(population=80, generations=50, episodes=32, max_steps=160)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--holdout-episodes", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed-step", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+")
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
    if args.replicates < 1:
        parser.error("--replicates must be at least 1")
    if args.seed_step < 1:
        parser.error("--seed-step must be at least 1")

    seeds = _resolve_seeds(args.seed, args.replicates, args.seed_step, args.seeds)
    if len(set(seeds)) != len(seeds):
        parser.error("comparison seeds must be distinct")

    base_config = evolution_config_from_args(args)
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
    _write_json(
        run_dir / "config.json",
        {
            "experiment": "compare_multiwall",
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
                "mean_route_gap": TARGET_ROUTE_GAP,
                "mean_all_walls_crossed_rate": TARGET_WALL_CROSS_RATE,
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
    results: list[MultiWallReplicateResult] = []
    viewer_paths: dict[int, Path] = {}
    used_scenario_seeds: set[int] = set()

    for replicate, seed in enumerate(seeds):
        evolution_config = replace(base_config, seed=seed)
        label = f"two walls | seed {seed} | repeat {replicate + 1}/{len(seeds)}"
        if live_viewer is not None and replicate > 0:
            live_viewer.begin_run(
                evolution_config,
                world_config,
                experiment="generalize_multiwall",
                label=label,
            )

        def on_generation(stats: GenerationStats) -> None:
            print(f"{replicate},{seed},{_format_stats(stats)}", flush=True)

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

        result = _run_replicate(
            replicate=replicate,
            seed=seed,
            evolution_config=evolution_config,
            world_config=world_config,
            validation_episodes=args.validation_episodes,
            holdout_episodes=args.holdout_episodes,
            on_generation=on_generation,
            on_validation=on_validation,
            on_evaluation=live_viewer.record_evaluation if live_viewer else None,
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
                f"Scenario seeds overlap between replicates: {sorted(overlap)[:5]}"
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
                world_class=MultiWallFoodWorld,
                experiment="generalize_multiwall",
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
                experiment="generalize_multiwall",
                label=(
                    f"seed {seed} | two-wall validation checkpoint | generation "
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
            f"route_gap={result.record.holdout_route_gap:.2%},"
            f"crossed2={result.record.holdout_all_walls_crossed_rate:.2%}",
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
            experiment="generalize_multiwall",
            label="two-wall generalization across seeds",
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
            final_label="two-wall generalization across seeds",
            message=f"Comparison complete: {len(results)} independent seeds",
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
    world_config: MultiWallFoodWorldConfig,
    validation_episodes: int,
    holdout_episodes: int,
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
) -> MultiWallReplicateResult:
    scenario_namespace = _scenario_namespace(seed)
    training_scenarios = _balanced_multiwall_scenarios(
        world_config,
        count=evolution_config.episodes_per_genome,
        first_seed=scenario_namespace,
    )
    validation_scenarios = _balanced_multiwall_scenarios(
        world_config,
        count=validation_episodes,
        first_seed=scenario_namespace + VALIDATION_SEED_OFFSET,
    )
    holdout_scenarios = _balanced_multiwall_scenarios(
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
            _live_trace_episode_indices(training_scenarios, trace_sample_size)
            if trace_sample_size
            else None
        ),
    )
    training_records, _ = _evaluate_scenarios(
        training.best_validation_genome,
        world_config,
        training_scenarios,
        split="training",
        record_replays=False,
    )
    holdout_records, holdout_replays = _evaluate_scenarios(
        training.best_validation_genome,
        world_config,
        holdout_scenarios,
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
        holdout_records,
        category_records,
    )
    return MultiWallReplicateResult(
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
    training: MultiWallTrainingResult,
    training_records: list[MultiWallEvaluationRecord],
    holdout_records: list[MultiWallEvaluationRecord],
    category_records: list[MultiWallCategoryRecord],
) -> MultiWallComparisonRecord:
    vertical = [item for item in holdout_records if item.orientation == "vertical"]
    horizontal = [
        item for item in holdout_records if item.orientation == "horizontal"
    ]
    aligned = [item for item in holdout_records if item.door_pattern == "aligned"]
    alternating = [
        item for item in holdout_records if item.door_pattern == "alternating"
    ]
    vertical_rate = _eat_rate(vertical)
    horizontal_rate = _eat_rate(horizontal)
    aligned_rate = _eat_rate(aligned)
    alternating_rate = _eat_rate(alternating)
    holdout_rate = _eat_rate(holdout_records)
    worst = min(
        category_records,
        key=lambda item: (
            item.eat_rate,
            item.orientation,
            item.start_side,
            item.first_door_side,
            item.door_pattern,
        ),
    )
    validation = training.selected_validation
    return MultiWallComparisonRecord(
        replicate=replicate,
        seed=seed,
        selected_generation=training.selected_stats.generation,
        training_average_score=mean(item.score for item in training_records),
        training_eat_rate=_eat_rate(training_records),
        validation_average_score=validation.average_score,
        validation_eat_rate=validation.overall_eat_rate,
        holdout_average_score=mean(item.score for item in holdout_records),
        holdout_eat_rate=holdout_rate,
        holdout_vertical_eat_rate=vertical_rate,
        holdout_horizontal_eat_rate=horizontal_rate,
        holdout_orientation_gap=abs(vertical_rate - horizontal_rate),
        holdout_aligned_eat_rate=aligned_rate,
        holdout_alternating_eat_rate=alternating_rate,
        holdout_route_gap=abs(aligned_rate - alternating_rate),
        holdout_all_walls_crossed_rate=mean(
            item.walls_crossed == WALL_COUNT for item in holdout_records
        ),
        holdout_stuck_rate=mean(item.stuck for item in holdout_records),
        holdout_average_collisions=mean(
            item.collisions for item in holdout_records
        ),
        generalization_gap=_eat_rate(training_records) - holdout_rate,
        worst_category=_category_label(worst),
        worst_category_eat_rate=worst.eat_rate,
    )


def _category_records(
    replicate: int,
    seed: int,
    scenarios: tuple[BalancedMultiWallScenario, ...],
    records: list[MultiWallEvaluationRecord],
) -> list[MultiWallCategoryRecord]:
    if len(scenarios) != len(records):
        raise ValueError("Holdout scenarios and records must have equal lengths.")
    grouped: dict[tuple[str, str, str, str], list[MultiWallEvaluationRecord]] = {}
    for scenario, record in zip(scenarios, records):
        key = (
            scenario.orientation,
            scenario.start_side,
            scenario.first_door_side,
            scenario.door_pattern,
        )
        grouped.setdefault(key, []).append(record)

    result = []
    for (orientation, start, first_door, pattern), items in sorted(grouped.items()):
        successes = sum(item.ate_food for item in items)
        crossed_both = sum(item.walls_crossed == WALL_COUNT for item in items)
        result.append(
            MultiWallCategoryRecord(
                replicate=replicate,
                seed=seed,
                orientation=orientation,
                start_side=start,
                first_door_side=first_door,
                door_pattern=pattern,
                successes=successes,
                crossed_both=crossed_both,
                episodes=len(items),
                eat_rate=successes / len(items),
                cross_rate=crossed_both / len(items),
                average_collisions=mean(item.collisions for item in items),
                average_final_distance=mean(item.final_distance for item in items),
            )
        )
    return result


def _rank_results(
    results: list[MultiWallReplicateResult],
    limit: int,
    world_config: MultiWallFoodWorldConfig,
) -> tuple[list[MultiWallReplicateResult], list[RankedReplay]]:
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
        episode = MultiWallFoodWorld(world_config).evaluate(
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
    result: MultiWallReplicateResult,
    world_config: MultiWallFoodWorldConfig,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "config.json",
        {
            "experiment": "generalize_multiwall",
            "replicate": result.record.replicate,
            "evolution": asdict(result.evolution_config),
            "world": asdict(world_config),
            "training_scenarios": [asdict(item) for item in result.training_scenarios],
            "validation_scenarios": [
                asdict(item) for item in result.validation_scenarios
            ],
            "holdout_scenarios": [
                asdict(item) for item in result.holdout_scenarios
            ],
        },
    )
    _write_metrics(run_dir / "metrics.csv", result.training.history)
    _write_records(
        run_dir / "validation.csv",
        MultiWallValidationRecord,
        result.training.validation_history,
    )
    _write_records(
        run_dir / "validation_episodes.csv",
        MultiWallValidationEpisodeRecord,
        result.training.validation_episode_records,
    )
    _write_records(
        run_dir / "evaluation.csv",
        MultiWallEvaluationRecord,
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
    records: list[MultiWallComparisonRecord],
    categories: list[MultiWallCategoryRecord],
    viewer_paths: dict[int, Path],
    comparison_viewer_path: Path | None,
) -> None:
    holdout_rates = [item.holdout_eat_rate for item in records]
    orientation_gaps = [item.holdout_orientation_gap for item in records]
    route_gaps = [item.holdout_route_gap for item in records]
    cross_rates = [item.holdout_all_walls_crossed_rate for item in records]
    stuck_rates = [item.holdout_stuck_rate for item in records]
    required_passes = math.ceil(len(records) * TARGET_PASS_FRACTION)
    passing_runs = sum(rate >= TARGET_RUN_HOLDOUT for rate in holdout_rates)
    checks = {
        "mean_holdout": mean(holdout_rates) >= TARGET_MEAN_HOLDOUT,
        "passing_runs": passing_runs >= required_passes,
        "orientation_gap": mean(orientation_gaps) <= TARGET_ORIENTATION_GAP,
        "route_gap": mean(route_gaps) <= TARGET_ROUTE_GAP,
        "cross_rate": mean(cross_rates) >= TARGET_WALL_CROSS_RATE,
        "stuck_rate": mean(stuck_rates) <= TARGET_STUCK_RATE,
    }
    passed = all(checks.values())
    lines = [
        "# Две стены на разных seed",
        "",
        f"**Вердикт: {'ЭТАП ПРОЙДЕН' if passed else 'НУЖНЫ ЕЩЕ УЛУЧШЕНИЯ'}**",
        "",
        "Каждая строка использует независимую начальную популяцию и свои",
        "непересекающиеся training, validation и holdout-сцены. Абсолютные",
        "координаты во всех запусках выключены.",
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
            "Mean same/zigzag gap",
            mean(route_gaps),
            f"<= {TARGET_ROUTE_GAP:.0%}",
            checks["route_gap"],
        ),
        _target_row(
            "Mean crossed both walls",
            mean(cross_rates),
            f">= {TARGET_WALL_CROSS_RATE:.0%}",
            checks["cross_rate"],
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
        "| Seed | Checkpoint | Train | Validation | Holdout | V | H | Orientation gap | Same | Zigzag | Route gap | Crossed both | Stuck |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in sorted(records, key=lambda item: item.seed):
        lines.append(
            f"| {record.seed} | {record.selected_generation + 1} | "
            f"{record.training_eat_rate:.2%} | {record.validation_eat_rate:.2%} | "
            f"{record.holdout_eat_rate:.2%} | "
            f"{record.holdout_vertical_eat_rate:.2%} | "
            f"{record.holdout_horizontal_eat_rate:.2%} | "
            f"{record.holdout_orientation_gap:.2%} | "
            f"{record.holdout_aligned_eat_rate:.2%} | "
            f"{record.holdout_alternating_eat_rate:.2%} | "
            f"{record.holdout_route_gap:.2%} | "
            f"{record.holdout_all_walls_crossed_rate:.2%} | "
            f"{record.holdout_stuck_rate:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Категории holdout",
            "",
            "| Orientation | Start | First door | Route | Success | Crossed both | Eat rate | Avg collisions |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _aggregate_categories(categories):
        lines.append(
            f"| {row['orientation']} | {row['start_side']} | "
            f"{row['first_door_side']} | {row['door_pattern']} | "
            f"{row['successes']}/{row['episodes']} | "
            f"{row['crossed_both']}/{row['episodes']} | "
            f"{row['eat_rate']:.2%} | {row['average_collisions']:.2f} |"
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


def _aggregate_categories(
    categories: list[MultiWallCategoryRecord],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[MultiWallCategoryRecord]] = {}
    for record in categories:
        key = (
            record.orientation,
            record.start_side,
            record.first_door_side,
            record.door_pattern,
        )
        grouped.setdefault(key, []).append(record)

    rows = []
    for (orientation, start, first_door, pattern), items in sorted(grouped.items()):
        successes = sum(item.successes for item in items)
        crossed_both = sum(item.crossed_both for item in items)
        episodes = sum(item.episodes for item in items)
        rows.append(
            {
                "orientation": orientation,
                "start_side": start,
                "first_door_side": first_door,
                "door_pattern": pattern,
                "successes": successes,
                "crossed_both": crossed_both,
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


def _category_label(record: MultiWallCategoryRecord) -> str:
    return (
        f"{record.orientation}|start={record.start_side}|"
        f"first={record.first_door_side}|route={record.door_pattern}"
    )


def _scenario_namespace(seed: int) -> int:
    namespace_index = seed * 2 if seed >= 0 else -seed * 2 - 1
    return namespace_index * SCENARIO_NAMESPACE_SIZE


def _eat_rate(records: list[MultiWallEvaluationRecord]) -> float:
    return sum(item.ate_food for item in records) / len(records)


def _start_live_viewer(
    args: argparse.Namespace,
    evolution_config: EvolutionConfig,
    world_config: MultiWallFoodWorldConfig,
    seeds: tuple[int, ...],
) -> LiveRunViewer | None:
    if not args.open_viewer:
        return None
    try:
        viewer = LiveRunViewer(
            evolution_config=evolution_config,
            world_config=world_config,
            sample_limit=max(args.live_samples, 8),
            experiment="generalize_multiwall",
            label=f"two walls | seed {seeds[0]} | repeat 1/{len(seeds)}",
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
