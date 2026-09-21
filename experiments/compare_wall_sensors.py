from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random
from statistics import mean, pstdev

from evolution import EvolutionConfig, GenerationStats
from experiments.compare_generalization import (
    ReplicateResult,
    TARGET_MEAN_HOLDOUT,
    TARGET_ORIENTATION_GAP,
    TARGET_PASS_FRACTION,
    TARGET_RUN_HOLDOUT,
    TARGET_STUCK_RATE,
    _resolve_seeds,
    _run_replicate,
    _write_csv,
    _write_replicate_artifacts,
)
from experiments.generalize_rooms import (
    BalancedRoomScenario,
    EvaluationRecord,
    ValidationGenerationRecord,
    VALIDATION_STUCK_STEPS,
    _crossed_wall,
    _evaluate_scenarios,
    _has_collision_streak,
    _write_evaluations,
)
from experiments.runner import (
    _create_run_dir,
    _format_stats,
    _open_in_browser,
    _write_json,
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


FULL = "full"
NO_WALL = "no-wall-sensors"
KNOCKOUT = "full-knockout"
NO_POSITION = "no-position-sensors"
POSITION_KNOCKOUT = "full-position-knockout"
DEFAULT_EFFECT_THRESHOLD = 0.10


@dataclass(frozen=True)
class AblationSpec:
    experiment: str
    output: Path
    description: str
    report_title: str
    config_field: str
    signal_name: str
    zeroed_inputs: str
    ablated_condition: str
    knockout_condition: str
    retrained_summary_label: str
    retrained_description: str
    knockout_description: str
    retrained_showcase_label: str
    knockout_showcase_label: str
    final_label: str


WALL_SENSOR_SPEC = AblationSpec(
    experiment="compare_wall_sensors",
    output=Path("results/compare_wall_sensors"),
    description=(
        "Compare normal wall sensing, retraining with zero wall sensors, "
        "and test-time sensor knockout."
    ),
    report_title="Ablation датчиков стен",
    config_field="use_wall_sensors",
    signal_name="wall-сенсоры",
    zeroed_inputs="последние три входа всегда равны нулю",
    ablated_condition=NO_WALL,
    knockout_condition=KNOCKOUT,
    retrained_summary_label="Retrained without wall sensors",
    retrained_description="Новая стратегия без wall-сигнала",
    knockout_description="Те же full-мозги, но три wall-входа обнулены",
    retrained_showcase_label="retrained without wall sensors",
    knockout_showcase_label="full brain, wall sensors off",
    final_label="wall sensor ablation",
)


POSITION_SENSOR_SPEC = AblationSpec(
    experiment="compare_position_sensors",
    output=Path("results/compare_position_sensors"),
    description=(
        "Compare normal position sensing, retraining with zero agent x/y, "
        "and test-time position knockout."
    ),
    report_title="Ablation абсолютных координат",
    config_field="use_position_sensors",
    signal_name="координаты agent x/y",
    zeroed_inputs="входы agent x и agent y всегда равны нулю",
    ablated_condition=NO_POSITION,
    knockout_condition=POSITION_KNOCKOUT,
    retrained_summary_label="Retrained without position",
    retrained_description="Новая стратегия без абсолютной позиции",
    knockout_description="Те же full-мозги, но входы agent x/y обнулены",
    retrained_showcase_label="retrained without position",
    knockout_showcase_label="full brain, position off",
    final_label="position sensor ablation",
)


@dataclass(frozen=True)
class EvaluationSummary:
    average_score: float
    eat_rate: float
    vertical_eat_rate: float
    horizontal_eat_rate: float
    orientation_gap: float
    wall_cross_rate: float
    stuck_rate: float
    average_collisions: float


@dataclass(frozen=True)
class InputAblationRecord:
    replicate: int
    seed: int
    full_selected_generation: int
    ablated_selected_generation: int
    full_training_eat_rate: float
    ablated_training_eat_rate: float
    full_validation_eat_rate: float
    ablated_validation_eat_rate: float
    full_holdout_eat_rate: float
    ablated_holdout_eat_rate: float
    trained_ablation_delta: float
    knockout_holdout_eat_rate: float
    knockout_delta: float
    full_vertical_eat_rate: float
    ablated_vertical_eat_rate: float
    knockout_vertical_eat_rate: float
    full_horizontal_eat_rate: float
    ablated_horizontal_eat_rate: float
    knockout_horizontal_eat_rate: float
    full_orientation_gap: float
    ablated_orientation_gap: float
    knockout_orientation_gap: float
    full_stuck_rate: float
    ablated_stuck_rate: float
    knockout_stuck_rate: float
    full_average_collisions: float
    ablated_average_collisions: float
    knockout_average_collisions: float


@dataclass(frozen=True)
class AblationCategoryRecord:
    replicate: int
    seed: int
    condition: str
    orientation: str
    start_side: str
    door_side: str
    successes: int
    episodes: int
    eat_rate: float
    average_collisions: float
    average_final_distance: float


@dataclass(frozen=True)
class PairedAblationResult:
    record: InputAblationRecord
    full: ReplicateResult
    ablated: ReplicateResult
    knockout_records: list[EvaluationRecord]
    knockout_replays: list[ReplayEpisode]
    category_records: list[AblationCategoryRecord]


@dataclass(frozen=True)
class PairViewerPaths:
    pair: Path
    full: Path
    ablated: Path
    knockout: Path


def main(spec: AblationSpec = WALL_SENSOR_SPEC) -> None:
    parser = argparse.ArgumentParser(
        description=spec.description,
    )
    add_evolution_arguments(parser, output=spec.output)
    parser.set_defaults(population=80, generations=40, episodes=24, max_steps=100)
    parser.add_argument("--validation-episodes", type=int, default=24)
    parser.add_argument("--holdout-episodes", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed-step", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--door-width", type=float, default=1.8)
    parser.add_argument("--wall-sensor-range", type=float, default=2.8)
    parser.add_argument("--collision-penalty", type=float, default=0.25)
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=DEFAULT_EFFECT_THRESHOLD,
        help="Mean paired eat-rate difference treated as a material effect.",
    )
    args = parser.parse_args()
    validate_evolution_arguments(parser, args)
    _validate_arguments(parser, args)

    seeds = _resolve_seeds(args.seed, args.replicates, args.seed_step, args.seeds)
    if len(set(seeds)) != len(seeds):
        parser.error("comparison seeds must be distinct")

    base_evolution_config = evolution_config_from_args(args)
    full_world_config = RoomFoodWorldConfig(
        max_steps=args.max_steps,
        door_width=args.door_width,
        wall_sensor_range=args.wall_sensor_range,
        collision_penalty=args.collision_penalty,
        use_wall_sensors=True,
        use_position_sensors=True,
    )
    if not hasattr(full_world_config, spec.config_field):
        raise RuntimeError(f"Unknown ablation input: {spec.config_field}")
    ablated_world_config = replace(
        full_world_config,
        **{spec.config_field: False},
    )
    run_dir = _create_run_dir(args.output)
    _write_json(
        run_dir / "config.json",
        {
            "experiment": spec.experiment,
            "ablated_input": spec.config_field,
            "evolution": asdict(base_evolution_config),
            "full_world": asdict(full_world_config),
            "ablated_world": asdict(ablated_world_config),
            "seeds": list(seeds),
            "validation_episodes": args.validation_episodes,
            "holdout_episodes": args.holdout_episodes,
            "effect_threshold": args.effect_threshold,
            "baseline_targets": {
                "mean_holdout_eat_rate": TARGET_MEAN_HOLDOUT,
                "run_holdout_eat_rate": TARGET_RUN_HOLDOUT,
                "passing_run_fraction": TARGET_PASS_FRACTION,
                "mean_orientation_gap": TARGET_ORIENTATION_GAP,
                "mean_stuck_rate": TARGET_STUCK_RATE,
            },
            "design": {
                "paired_room_seeds": True,
                "same_input_size": True,
                "retrain_without_input": spec.config_field,
                "test_time_knockout": True,
                "holdout_used_for_selection": False,
            },
        },
    )

    print(f"Run: {run_dir}")
    print(
        "condition,replicate,seed,generation,best_score,avg_score,"
        "best_eat_rate,avg_hidden",
        flush=True,
    )
    live_viewer = _start_live_viewer(
        args,
        base_evolution_config,
        full_world_config,
        seeds,
    )
    results: list[PairedAblationResult] = []
    viewer_paths: dict[int, PairViewerPaths] = {}

    for replicate, seed in enumerate(seeds):
        evolution_config = replace(base_evolution_config, seed=seed)
        full = _run_condition(
            condition=FULL,
            replicate=replicate,
            seed=seed,
            repeat_count=len(seeds),
            evolution_config=evolution_config,
            world_config=full_world_config,
            validation_episodes=args.validation_episodes,
            holdout_episodes=args.holdout_episodes,
            live_viewer=live_viewer,
            live_samples=args.live_samples,
        )
        ablated = _run_condition(
            condition=spec.ablated_condition,
            replicate=replicate,
            seed=seed,
            repeat_count=len(seeds),
            evolution_config=evolution_config,
            world_config=ablated_world_config,
            validation_episodes=args.validation_episodes,
            holdout_episodes=args.holdout_episodes,
            live_viewer=live_viewer,
            live_samples=args.live_samples,
        )
        result = _build_ablation_result(
            replicate=replicate,
            seed=seed,
            full=full,
            ablated=ablated,
            ablated_world_config=ablated_world_config,
            spec=spec,
        )
        results.append(result)

        pair_dir = run_dir / "runs" / f"seed-{seed}"
        paths = _write_pair_artifacts(
            pair_dir=pair_dir,
            result=result,
            full_world_config=full_world_config,
            ablated_world_config=ablated_world_config,
            visual_seed=args.visual_seed,
            write_viewers=not args.no_visualization,
            spec=spec,
        )
        if paths is not None:
            viewer_paths[seed] = paths

        print(
            f"result,{replicate},{seed},"
            f"full={result.record.full_holdout_eat_rate:.2%},"
            f"ablated={result.record.ablated_holdout_eat_rate:.2%},"
            f"trained_delta={result.record.trained_ablation_delta:+.2%},"
            f"knockout={result.record.knockout_holdout_eat_rate:.2%},"
            f"knockout_delta={result.record.knockout_delta:+.2%}",
            flush=True,
        )

    records = [result.record for result in results]
    categories = [item for result in results for item in result.category_records]
    _write_csv(run_dir / "comparison.csv", records)
    _write_csv(run_dir / "categories.csv", categories)

    representative = _representative_result(results)
    representative_replays = _paired_showcase_replays(
        representative,
        full_world_config,
        ablated_world_config,
        args.visual_seed,
        spec,
    )
    comparison_viewer_path = None
    if not args.no_visualization:
        comparison_viewer_path = write_ranked_visualization(
            run_dir=run_dir,
            ranked_replays=representative_replays,
            history=representative.full.training.history,
            evolution_config=representative.full.evolution_config,
            world_config=full_world_config,
            experiment="generalize_rooms",
            label=(
                f"{spec.final_label} | representative seed "
                f"{representative.record.seed}"
            ),
            diagnostics=[
                asdict(item) for item in representative.full.training.validation_history
            ],
            selected_stats=representative.full.training.selected_stats,
            status_message=(
                "Paired sensor conditions in the same showcase room; "
                "summary metrics use all holdout rooms"
            ),
        )

    _write_report(
        run_dir / "report.md",
        records,
        categories,
        viewer_paths,
        comparison_viewer_path,
        args.effect_threshold,
        spec,
    )

    if live_viewer is not None:
        live_viewer.finish(
            representative.full.training.selected_stats,
            representative_replays[0].episode,
            comparison_viewer_path,
            ranked_replays=representative_replays,
            final_label=spec.final_label,
            message=f"Ablation complete: {len(results)} paired seeds",
        )
        live_viewer.wait_until_final_served()

    print(f"Saved comparison to: {run_dir}")
    if comparison_viewer_path is not None:
        print(f"Open comparison viewer: {comparison_viewer_path}")
        if args.open_viewer and live_viewer is None:
            _open_in_browser(comparison_viewer_path.resolve().as_uri())


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("episodes", "validation_episodes", "holdout_episodes"):
        value = getattr(args, name)
        if value < 8 or value % 8 != 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive multiple of 8")
    if args.replicates < 1:
        parser.error("--replicates must be at least 1")
    if args.seed_step < 1:
        parser.error("--seed-step must be at least 1")
    if not 0.0 <= args.effect_threshold <= 1.0:
        parser.error("--effect-threshold must be between 0 and 1")


def _run_condition(
    *,
    condition: str,
    replicate: int,
    seed: int,
    repeat_count: int,
    evolution_config: EvolutionConfig,
    world_config: RoomFoodWorldConfig,
    validation_episodes: int,
    holdout_episodes: int,
    live_viewer: LiveRunViewer | None,
    live_samples: int,
) -> ReplicateResult:
    label = (
        f"{condition} | seed {seed} | pair {replicate + 1}/{repeat_count}"
    )
    if live_viewer is not None:
        live_viewer.begin_run(
            evolution_config,
            world_config,
            experiment="generalize_rooms",
            label=label,
        )

    def on_generation(stats: GenerationStats) -> None:
        print(
            f"{condition},{replicate},{seed},{_format_stats(stats)}",
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
            f"  {condition} validation: "
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

    return _run_replicate(
        replicate=replicate,
        seed=seed,
        evolution_config=evolution_config,
        world_config=world_config,
        validation_episodes=validation_episodes,
        holdout_episodes=holdout_episodes,
        on_generation=on_generation,
        on_validation=on_validation,
        on_evaluation=(live_viewer.record_evaluation if live_viewer else None),
        trace_sample_size=live_samples if live_viewer else 0,
    )


def _run_ablation_pair(
    *,
    replicate: int,
    seed: int,
    evolution_config: EvolutionConfig,
    full_world_config: RoomFoodWorldConfig,
    ablated_world_config: RoomFoodWorldConfig,
    validation_episodes: int,
    holdout_episodes: int,
    spec: AblationSpec = WALL_SENSOR_SPEC,
) -> PairedAblationResult:
    """Run one small paired experiment; also used by the integration test."""
    full = _run_replicate(
        replicate=replicate,
        seed=seed,
        evolution_config=evolution_config,
        world_config=full_world_config,
        validation_episodes=validation_episodes,
        holdout_episodes=holdout_episodes,
    )
    ablated = _run_replicate(
        replicate=replicate,
        seed=seed,
        evolution_config=evolution_config,
        world_config=ablated_world_config,
        validation_episodes=validation_episodes,
        holdout_episodes=holdout_episodes,
    )
    return _build_ablation_result(
        replicate=replicate,
        seed=seed,
        full=full,
        ablated=ablated,
        ablated_world_config=ablated_world_config,
        spec=spec,
    )


def _build_ablation_result(
    *,
    replicate: int,
    seed: int,
    full: ReplicateResult,
    ablated: ReplicateResult,
    ablated_world_config: RoomFoodWorldConfig,
    spec: AblationSpec,
) -> PairedAblationResult:
    _assert_paired_scenarios(full, ablated)
    holdout_seeds = tuple(item.seed for item in full.holdout_scenarios)
    knockout_records, knockout_replays = _evaluate_scenarios(
        full.training.best_validation_genome,
        ablated_world_config,
        holdout_seeds,
        split=spec.knockout_condition,
        record_replays=True,
    )

    full_summary = _summarize_evaluation(
        full.holdout_scenarios,
        full.holdout_records,
        full.holdout_replays,
    )
    ablated_summary = _summarize_evaluation(
        ablated.holdout_scenarios,
        ablated.holdout_records,
        ablated.holdout_replays,
    )
    knockout_summary = _summarize_evaluation(
        full.holdout_scenarios,
        knockout_records,
        knockout_replays,
    )
    categories = []
    for condition, scenarios, records in (
        (FULL, full.holdout_scenarios, full.holdout_records),
        (
            spec.ablated_condition,
            ablated.holdout_scenarios,
            ablated.holdout_records,
        ),
        (spec.knockout_condition, full.holdout_scenarios, knockout_records),
    ):
        categories.extend(
            _category_records(replicate, seed, condition, scenarios, records)
        )

    record = InputAblationRecord(
        replicate=replicate,
        seed=seed,
        full_selected_generation=full.training.selected_stats.generation,
        ablated_selected_generation=ablated.training.selected_stats.generation,
        full_training_eat_rate=full.record.training_eat_rate,
        ablated_training_eat_rate=ablated.record.training_eat_rate,
        full_validation_eat_rate=full.record.validation_eat_rate,
        ablated_validation_eat_rate=ablated.record.validation_eat_rate,
        full_holdout_eat_rate=full_summary.eat_rate,
        ablated_holdout_eat_rate=ablated_summary.eat_rate,
        trained_ablation_delta=full_summary.eat_rate - ablated_summary.eat_rate,
        knockout_holdout_eat_rate=knockout_summary.eat_rate,
        knockout_delta=full_summary.eat_rate - knockout_summary.eat_rate,
        full_vertical_eat_rate=full_summary.vertical_eat_rate,
        ablated_vertical_eat_rate=ablated_summary.vertical_eat_rate,
        knockout_vertical_eat_rate=knockout_summary.vertical_eat_rate,
        full_horizontal_eat_rate=full_summary.horizontal_eat_rate,
        ablated_horizontal_eat_rate=ablated_summary.horizontal_eat_rate,
        knockout_horizontal_eat_rate=knockout_summary.horizontal_eat_rate,
        full_orientation_gap=full_summary.orientation_gap,
        ablated_orientation_gap=ablated_summary.orientation_gap,
        knockout_orientation_gap=knockout_summary.orientation_gap,
        full_stuck_rate=full_summary.stuck_rate,
        ablated_stuck_rate=ablated_summary.stuck_rate,
        knockout_stuck_rate=knockout_summary.stuck_rate,
        full_average_collisions=full_summary.average_collisions,
        ablated_average_collisions=ablated_summary.average_collisions,
        knockout_average_collisions=knockout_summary.average_collisions,
    )
    return PairedAblationResult(
        record=record,
        full=full,
        ablated=ablated,
        knockout_records=knockout_records,
        knockout_replays=knockout_replays,
        category_records=categories,
    )


def _assert_paired_scenarios(
    full: ReplicateResult,
    ablated: ReplicateResult,
) -> None:
    for split in ("training_scenarios", "validation_scenarios", "holdout_scenarios"):
        full_scenarios = getattr(full, split)
        ablated_scenarios = getattr(ablated, split)
        if full_scenarios != ablated_scenarios:
            raise RuntimeError(f"Paired conditions use different {split}.")


def _summarize_evaluation(
    scenarios: tuple[BalancedRoomScenario, ...],
    records: list[EvaluationRecord],
    replays: list[ReplayEpisode],
) -> EvaluationSummary:
    if len(scenarios) != len(records) or len(records) != len(replays):
        raise ValueError("Scenarios, records, and replays must have equal lengths.")
    paired = list(zip(scenarios, records))
    vertical = [record for scenario, record in paired if scenario.orientation == "vertical"]
    horizontal = [
        record for scenario, record in paired if scenario.orientation == "horizontal"
    ]
    vertical_rate = _eat_rate(vertical)
    horizontal_rate = _eat_rate(horizontal)
    return EvaluationSummary(
        average_score=mean(item.score for item in records),
        eat_rate=_eat_rate(records),
        vertical_eat_rate=vertical_rate,
        horizontal_eat_rate=horizontal_rate,
        orientation_gap=abs(vertical_rate - horizontal_rate),
        wall_cross_rate=mean(_crossed_wall(item.episode) for item in replays),
        stuck_rate=mean(
            _has_collision_streak(item.episode, VALIDATION_STUCK_STEPS)
            for item in replays
        ),
        average_collisions=mean(item.collisions for item in records),
    )


def _category_records(
    replicate: int,
    seed: int,
    condition: str,
    scenarios: tuple[BalancedRoomScenario, ...],
    records: list[EvaluationRecord],
) -> list[AblationCategoryRecord]:
    if len(scenarios) != len(records):
        raise ValueError("Scenarios and records must have equal lengths.")
    grouped: dict[tuple[str, str, str], list[EvaluationRecord]] = {}
    for scenario, record in zip(scenarios, records):
        key = (scenario.orientation, scenario.start_side, scenario.door_side)
        grouped.setdefault(key, []).append(record)

    result = []
    for (orientation, start_side, door_side), items in sorted(grouped.items()):
        result.append(
            AblationCategoryRecord(
                replicate=replicate,
                seed=seed,
                condition=condition,
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


def _write_pair_artifacts(
    *,
    pair_dir: Path,
    result: PairedAblationResult,
    full_world_config: RoomFoodWorldConfig,
    ablated_world_config: RoomFoodWorldConfig,
    visual_seed: int,
    write_viewers: bool,
    spec: AblationSpec,
) -> PairViewerPaths | None:
    full_dir = pair_dir / FULL
    ablated_dir = pair_dir / spec.ablated_condition
    knockout_dir = pair_dir / spec.knockout_condition
    _write_replicate_artifacts(full_dir, result.full, full_world_config)
    _write_replicate_artifacts(
        ablated_dir,
        result.ablated,
        ablated_world_config,
    )
    knockout_dir.mkdir(parents=True, exist_ok=False)
    _write_evaluations(knockout_dir / "evaluation.csv", result.knockout_records)
    _write_json(
        knockout_dir / "config.json",
        {
            "experiment": spec.knockout_condition,
            "ablated_input": spec.config_field,
            "source_genome": "../full/best_validation_genome.json",
            "world": asdict(ablated_world_config),
            "holdout_scenarios": [
                asdict(item) for item in result.full.holdout_scenarios
            ],
            "holdout_used_for_selection": False,
        },
    )
    _write_json(pair_dir / "comparison.json", asdict(result.record))

    if not write_viewers:
        return None

    full_viewer = _write_condition_viewer(
        full_dir,
        result.full,
        full_world_config,
    )
    ablated_viewer = _write_condition_viewer(
        ablated_dir,
        result.ablated,
        ablated_world_config,
    )
    knockout_viewer = write_visualization(
        run_dir=knockout_dir,
        best_genome=result.full.training.best_validation_genome,
        history=result.full.training.history,
        evolution_config=result.full.evolution_config,
        world_config=ablated_world_config,
        seed=result.knockout_replays[0].seed,
        world_class=RoomFoodWorld,
        experiment="generalize_rooms",
        replay_episodes=result.knockout_replays,
        diagnostics=[asdict(item) for item in result.full.training.validation_history],
        selected_generation=result.full.training.selected_stats.generation,
    )
    pair_viewer = write_ranked_visualization(
        run_dir=pair_dir,
        ranked_replays=_paired_showcase_replays(
            result,
            full_world_config,
            ablated_world_config,
            visual_seed,
            spec,
        ),
        history=result.full.training.history,
        evolution_config=result.full.evolution_config,
        world_config=full_world_config,
        experiment="generalize_rooms",
        label=f"{spec.final_label} | seed {result.record.seed}",
        diagnostics=[asdict(item) for item in result.full.training.validation_history],
        viewer_filename="pair-viewer.html",
        data_filename="pair-replay.json",
        selected_stats=result.full.training.selected_stats,
        status_message="Three paired input conditions in the same room",
    )
    return PairViewerPaths(
        pair=pair_viewer,
        full=full_viewer,
        ablated=ablated_viewer,
        knockout=knockout_viewer,
    )


def _write_condition_viewer(
    run_dir: Path,
    result: ReplicateResult,
    world_config: RoomFoodWorldConfig,
) -> Path:
    viewer = write_visualization(
        run_dir=run_dir,
        best_genome=result.training.best_validation_genome,
        history=result.training.history,
        evolution_config=result.evolution_config,
        world_config=world_config,
        seed=result.holdout_replays[0].seed,
        world_class=RoomFoodWorld,
        experiment="generalize_rooms",
        replay_episodes=result.holdout_replays,
        diagnostics=[asdict(item) for item in result.training.validation_history],
        selected_generation=result.training.selected_stats.generation,
    )
    write_ranked_visualization(
        run_dir=run_dir,
        ranked_replays=result.training.selected_replays,
        history=result.training.history,
        evolution_config=result.evolution_config,
        world_config=world_config,
        experiment="generalize_rooms",
        label=(
            f"validation checkpoint | generation "
            f"{result.training.selected_stats.generation + 1}/"
            f"{result.evolution_config.generations}"
        ),
        diagnostics=[asdict(item) for item in result.training.validation_history],
        viewer_filename="validation-viewer.html",
        data_filename="validation-replay.json",
        selected_stats=result.training.selected_stats,
    )
    return viewer


def _paired_showcase_replays(
    result: PairedAblationResult,
    full_world_config: RoomFoodWorldConfig,
    ablated_world_config: RoomFoodWorldConfig,
    visual_seed: int,
    spec: AblationSpec = WALL_SENSOR_SPEC,
) -> list[RankedReplay]:
    full_world = RoomFoodWorld(full_world_config)
    ablated_world = RoomFoodWorld(ablated_world_config)
    full_genome = result.full.training.best_validation_genome
    ablated_genome = result.ablated.training.best_validation_genome
    return [
        RankedReplay(
            rank=1,
            score=result.full.record.holdout_average_score,
            eat_rate=result.record.full_holdout_eat_rate,
            episode=full_world.evaluate(
                full_genome,
                random.Random(visual_seed),
                record_trace=True,
            ),
            label=f"full | seed {result.record.seed}",
            metric="ablation",
        ),
        RankedReplay(
            rank=2,
            score=result.ablated.record.holdout_average_score,
            eat_rate=result.record.ablated_holdout_eat_rate,
            episode=ablated_world.evaluate(
                ablated_genome,
                random.Random(visual_seed),
                record_trace=True,
            ),
            label=f"{spec.retrained_showcase_label} | seed {result.record.seed}",
            metric="ablation",
        ),
        RankedReplay(
            rank=3,
            score=mean(item.score for item in result.knockout_records),
            eat_rate=result.record.knockout_holdout_eat_rate,
            episode=ablated_world.evaluate(
                full_genome,
                random.Random(visual_seed),
                record_trace=True,
            ),
            label=f"{spec.knockout_showcase_label} | seed {result.record.seed}",
            metric="ablation",
        ),
    ]


def _representative_result(
    results: list[PairedAblationResult],
) -> PairedAblationResult:
    target = mean(item.record.trained_ablation_delta for item in results)
    return min(
        results,
        key=lambda item: (
            abs(item.record.trained_ablation_delta - target),
            item.record.seed,
        ),
    )


def _write_report(
    path: Path,
    records: list[InputAblationRecord],
    categories: list[AblationCategoryRecord],
    viewer_paths: dict[int, PairViewerPaths],
    comparison_viewer_path: Path | None,
    effect_threshold: float,
    spec: AblationSpec,
) -> None:
    full_rates = [item.full_holdout_eat_rate for item in records]
    ablated_rates = [item.ablated_holdout_eat_rate for item in records]
    knockout_rates = [item.knockout_holdout_eat_rate for item in records]
    trained_deltas = [item.trained_ablation_delta for item in records]
    knockout_deltas = [item.knockout_delta for item in records]
    required_baseline_passes = math.ceil(len(records) * TARGET_PASS_FRACTION)
    baseline_passing_runs = sum(
        rate >= TARGET_RUN_HOLDOUT for rate in full_rates
    )
    mean_full_orientation_gap = mean(
        item.full_orientation_gap for item in records
    )
    mean_full_stuck_rate = mean(item.full_stuck_rate for item in records)
    baseline_ready = (
        mean(full_rates) >= TARGET_MEAN_HOLDOUT
        and baseline_passing_runs >= required_baseline_passes
        and mean_full_orientation_gap <= TARGET_ORIENTATION_GAP
        and mean_full_stuck_rate <= TARGET_STUCK_RATE
    )
    trained_material = mean(trained_deltas) >= effect_threshold
    knockout_material = mean(knockout_deltas) >= effect_threshold
    conclusion = _ablation_conclusion(
        trained_material,
        knockout_material,
        baseline_ready=baseline_ready,
        signal_name=spec.signal_name,
    )

    lines = [
        f"# {spec.report_title}",
        "",
        f"**Вывод: {conclusion}**",
        "",
        "Условия попарно используют одинаковые evolutionary seed и одинаковые",
        "training, validation и holdout-комнаты. Размер мозга в обоих условиях",
        f"одинаков: при ablation {spec.zeroed_inputs}.",
        "",
        "## Два разных вопроса",
        "",
        (
            f"- `{spec.ablated_condition}` обучается заново и проверяет, "
            f"может ли эволюция найти обходную стратегию без сигнала."
        ),
        (
            f"- `{spec.knockout_condition}` выключает сигнал только у готового "
            "full-мозга и проверяет его текущую зависимость."
        ),
        "",
        "## Сводка",
        "",
        "| Метрика | Значение | Интерпретация |",
        "| --- | ---: | --- |",
        (
            f"| Full holdout | {mean(full_rates):.2%} | "
            f"Baseline {'PASS' if baseline_ready else 'FAIL'} |"
        ),
        (
            f"| Full runs above {TARGET_RUN_HOLDOUT:.0%} | "
            f"{baseline_passing_runs}/{len(records)} | "
            f"Нужно >= {required_baseline_passes}/{len(records)} |"
        ),
        (
            f"| Full orientation gap | {mean_full_orientation_gap:.2%} | "
            f"Нужно <= {TARGET_ORIENTATION_GAP:.0%} |"
        ),
        (
            f"| Full stuck rate | {mean_full_stuck_rate:.2%} | "
            f"Нужно <= {TARGET_STUCK_RATE:.0%} |"
        ),
        (
            f"| {spec.retrained_summary_label} | {mean(ablated_rates):.2%} | "
            f"{spec.retrained_description} |"
        ),
        (
            f"| Paired retraining delta | {mean(trained_deltas):+.2%} | "
            f"{'Существенный эффект >=' if trained_material else 'Ниже'} {effect_threshold:.0%} |"
        ),
        (
            f"| Full brains after knockout | {mean(knockout_rates):.2%} | "
            f"{spec.knockout_description} |"
        ),
        (
            f"| Paired knockout delta | {mean(knockout_deltas):+.2%} | "
            f"{'Существенный эффект >=' if knockout_material else 'Ниже'} {effect_threshold:.0%} |"
        ),
        "",
        (
            f"Retraining delta: standard deviation {pstdev(trained_deltas):.2%}, "
            f"range {min(trained_deltas):+.2%} to {max(trained_deltas):+.2%}."
        ),
        (
            f"Knockout delta: standard deviation {pstdev(knockout_deltas):.2%}, "
            f"range {min(knockout_deltas):+.2%} to {max(knockout_deltas):+.2%}."
        ),
        "",
        f"Положительная delta означает преимущество сигнала: {spec.signal_name}.",
        "Порог служит практической границей эффекта, а не статистическим доказательством.",
        "Ablation интерпретируется только после прохождения full-baseline.",
        "",
        "## Запуски",
        "",
        "| Seed | Full checkpoint | Ablated checkpoint | Full | Ablated | Train delta | Knockout | Knockout delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(records, key=lambda row: row.seed):
        lines.append(
            f"| {item.seed} | {item.full_selected_generation + 1} | "
            f"{item.ablated_selected_generation + 1} | "
            f"{item.full_holdout_eat_rate:.2%} | "
            f"{item.ablated_holdout_eat_rate:.2%} | "
            f"{item.trained_ablation_delta:+.2%} | "
            f"{item.knockout_holdout_eat_rate:.2%} | "
            f"{item.knockout_delta:+.2%} |"
        )

    lines.extend(
        [
            "",
            "## Категории holdout",
            "",
            "| Condition | Orientation | Start | Door | Success | Eat rate | Avg collisions |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in _aggregate_categories(categories, spec):
        lines.append(
            f"| {row['condition']} | {row['orientation']} | "
            f"{row['start_side']} | {row['door_side']} | "
            f"{row['successes']}/{row['episodes']} | "
            f"{row['eat_rate']:.2%} | {row['average_collisions']:.2f} |"
        )

    if comparison_viewer_path is not None:
        relative = comparison_viewer_path.relative_to(path.parent).as_posix()
        lines.extend(
            [
                "",
                "## Viewer",
                "",
                f"[Репрезентативная парная тройка]({relative})",
                "",
            ]
        )
    if viewer_paths:
        lines.extend(["## Отдельные seed", ""])
        for seed, paths in sorted(viewer_paths.items()):
            pair = paths.pair.relative_to(path.parent).as_posix()
            full = paths.full.relative_to(path.parent).as_posix()
            ablated = paths.ablated.relative_to(path.parent).as_posix()
            knockout = paths.knockout.relative_to(path.parent).as_posix()
            lines.append(
                f"- Seed {seed}: [pair]({pair}) | [full]({full}) | "
                f"[ablated]({ablated}) | [knockout]({knockout})"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate_categories(
    categories: list[AblationCategoryRecord],
    spec: AblationSpec,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[AblationCategoryRecord]] = {}
    for item in categories:
        key = (item.condition, item.orientation, item.start_side, item.door_side)
        grouped.setdefault(key, []).append(item)

    rows = []
    condition_order = {
        FULL: 0,
        spec.ablated_condition: 1,
        spec.knockout_condition: 2,
    }
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            condition_order[item[0][0]],
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    )
    for (condition, orientation, start_side, door_side), items in ordered_groups:
        successes = sum(item.successes for item in items)
        episodes = sum(item.episodes for item in items)
        rows.append(
            {
                "condition": condition,
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


def _ablation_conclusion(
    trained_material: bool,
    knockout_material: bool,
    *,
    baseline_ready: bool = True,
    signal_name: str = "wall-сенсоры",
) -> str:
    if not baseline_ready:
        return "результат неинтерпретируем: full-baseline не освоил задачу"
    if trained_material and knockout_material:
        return f"{signal_name} помогают обучению, и full-мозги используют этот сигнал"
    if not trained_material and knockout_material:
        return f"full-мозги используют {signal_name}, но могут выучить альтернативу"
    if trained_material and not knockout_material:
        return (
            f"{signal_name} помогают эволюционному поиску, "
            "но готовое поведение устойчиво к knockout"
        )
    return f"существенная необходимость сигнала «{signal_name}» пока не обнаружена"


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
            label=f"full | seed {seeds[0]} | pair 1/{len(seeds)}",
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
