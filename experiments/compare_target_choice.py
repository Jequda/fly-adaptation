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
from experiments.choose_target import (
    BalancedTargetScenario,
    TargetEvaluationRecord,
    TargetTrainingResult,
    TargetValidationEpisodeRecord,
    TargetValidationRecord,
    TARGET_CATEGORY_COUNT,
    TARGET_TRAINING_BANKS,
    _balanced_target_scenario_sets,
    _balanced_target_scenarios,
    _evaluate_scenarios,
    _flatten_target_scenario_sets,
    _live_trace_episode_indices,
    _run_training_with_validation,
    _summarize_records,
    _write_records,
    _write_report as _write_condition_report,
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
    PAIR_ALIGNMENT_FITNESS_WEIGHT,
    PAIR_CONTRAST_FITNESS_WEIGHT,
    WORST_CATEGORY_FITNESS_WEIGHT,
    WORST_CATEGORY_SCORE_WEIGHT,
    TargetChoiceWorld,
    TargetChoiceWorldConfig,
)


TARGET_RECURRENT_MEAN = 0.80
TARGET_RECURRENT_RUN = 0.70
TARGET_PASS_FRACTION = 0.80
TARGET_STATELESS_MAX = 0.60
TARGET_RECURRENT_ADVANTAGE = 0.20
TARGET_CUE_GAP = 0.20
TARGET_KNOCKOUT_DROP = 0.20
TARGET_WORST_CATEGORY = 0.60
SHOWCASE_SEED = 9_500_007
SCENARIO_NAMESPACE_SIZE = 10_000_000
VALIDATION_SEED_OFFSET = 2_000_003
HOLDOUT_SEED_OFFSET = 4_000_003


@dataclass(frozen=True)
class TargetComparisonRecord:
    condition: str
    replicate: int
    seed: int
    selected_generation: int
    training_average_score: float
    training_success_rate: float
    validation_average_score: float
    validation_success_rate: float
    validation_cue_a_success_rate: float
    validation_cue_b_success_rate: float
    validation_pair_alignment: float
    validation_pair_contrast_rate: float
    holdout_average_score: float
    holdout_success_rate: float
    holdout_cue_a_success_rate: float
    holdout_cue_b_success_rate: float
    holdout_cue_gap: float
    holdout_pair_alignment: float
    holdout_pair_contrast_rate: float
    holdout_wrong_choice_rate: float
    holdout_no_choice_rate: float
    generalization_gap: float
    worst_category: str
    worst_category_success_rate: float


@dataclass(frozen=True)
class TargetPairRecord:
    replicate: int
    seed: int
    recurrent_holdout_success_rate: float
    stateless_holdout_success_rate: float
    recurrent_advantage: float
    recurrent_only_successes: int
    stateless_only_successes: int
    test_time_knockout_success_rate: float
    test_time_knockout_drop: float


@dataclass(frozen=True)
class TargetCategoryRecord:
    condition: str
    replicate: int
    seed: int
    correct_target: str
    target_a_side: str
    target_a_sector: str
    successes: int
    wrong_choices: int
    no_choices: int
    episodes: int
    success_rate: float


@dataclass(frozen=True)
class TargetConditionResult:
    record: TargetComparisonRecord
    evolution_config: EvolutionConfig
    world_config: TargetChoiceWorldConfig
    training: TargetTrainingResult
    training_records: list[TargetEvaluationRecord]
    holdout_records: list[TargetEvaluationRecord]
    holdout_replays: list[ReplayEpisode]
    category_records: list[TargetCategoryRecord]


@dataclass(frozen=True)
class TargetReplicateResult:
    replicate: int
    seed: int
    training_scenarios: tuple[BalancedTargetScenario, ...]
    training_scenario_sets: tuple[tuple[BalancedTargetScenario, ...], ...]
    validation_scenarios: tuple[BalancedTargetScenario, ...]
    holdout_scenarios: tuple[BalancedTargetScenario, ...]
    recurrent: TargetConditionResult
    stateless: TargetConditionResult
    knockout_records: list[TargetEvaluationRecord]
    pair_record: TargetPairRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare recurrent and stateless target choice across independent seeds."
        ),
    )
    add_evolution_arguments(parser, output=Path("results/compare_target_choice"))
    parser.set_defaults(population=80, generations=60, episodes=32, max_steps=80)
    parser.add_argument("--validation-episodes", type=int, default=64)
    parser.add_argument("--holdout-episodes", type=int, default=64)
    parser.add_argument("--cue-steps", type=int, default=6)
    parser.add_argument(
        "--wrong-target-penalty",
        type=float,
        default=TargetChoiceWorldConfig.wrong_target_penalty,
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed-step", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    validate_evolution_arguments(parser, args)
    for name in ("episodes", "validation_episodes", "holdout_episodes"):
        value = getattr(args, name)
        if value < TARGET_CATEGORY_COUNT or value % TARGET_CATEGORY_COUNT != 0:
            parser.error(
                f"--{name.replace('_', '-')} must be a positive multiple of "
                f"{TARGET_CATEGORY_COUNT}"
            )
    if args.cue_steps < 1 or args.cue_steps >= args.max_steps:
        parser.error("--cue-steps must be positive and smaller than --max-steps")
    if args.wrong_target_penalty < 0:
        parser.error("--wrong-target-penalty cannot be negative")
    if args.replicates < 1:
        parser.error("--replicates must be at least 1")
    if args.seed_step < 1:
        parser.error("--seed-step must be at least 1")

    seeds = _resolve_seeds(args.seed, args.replicates, args.seed_step, args.seeds)
    if len(set(seeds)) != len(seeds):
        parser.error("comparison seeds must be distinct")

    base_config = evolution_config_from_args(args)
    base_world_config = TargetChoiceWorldConfig(
        max_steps=args.max_steps,
        cue_steps=args.cue_steps,
        wrong_target_penalty=args.wrong_target_penalty,
        use_recurrence=True,
    )
    run_dir = _create_run_dir(args.output)
    _write_json(
        run_dir / "config.json",
        {
            "experiment": "compare_target_choice",
            "evolution": asdict(base_config),
            "world": asdict(base_world_config),
            "seeds": list(seeds),
            "scenario_namespace_size": SCENARIO_NAMESPACE_SIZE,
            "validation_episodes": args.validation_episodes,
            "holdout_episodes": args.holdout_episodes,
            "training_scenario_sets": TARGET_TRAINING_BANKS,
            "fitness": {
                "pair_alignment_weight": PAIR_ALIGNMENT_FITNESS_WEIGHT,
                "pair_contrast_weight": PAIR_CONTRAST_FITNESS_WEIGHT,
                "worst_category_success_weight": (
                    WORST_CATEGORY_FITNESS_WEIGHT
                ),
                "worst_category_score_weight": WORST_CATEGORY_SCORE_WEIGHT,
            },
            "pairing": {
                "initial_population": "shared within each seed",
                "training_scenarios": "shared within each seed",
                "validation_scenarios": "shared within each seed",
                "holdout_scenarios": "shared within each seed",
                "target_geometry": "identical A/B cue pairs",
            },
            "targets": {
                "recurrent_mean_holdout": TARGET_RECURRENT_MEAN,
                "recurrent_run_holdout": TARGET_RECURRENT_RUN,
                "passing_run_fraction": TARGET_PASS_FRACTION,
                "stateless_mean_max": TARGET_STATELESS_MAX,
                "mean_recurrent_advantage": TARGET_RECURRENT_ADVANTAGE,
                "mean_recurrent_cue_gap": TARGET_CUE_GAP,
                "mean_test_time_knockout_drop": TARGET_KNOCKOUT_DROP,
                "worst_recurrent_category": TARGET_WORST_CATEGORY,
            },
        },
    )

    print(f"Run: {run_dir}")
    print(
        "condition,replicate,seed,generation,best_fitness,avg_fitness,"
        "best_success_rate,avg_hidden",
        flush=True,
    )
    live_viewer = _start_live_viewer(
        args,
        base_config,
        base_world_config,
        seeds,
    )
    replicate_results: list[TargetReplicateResult] = []
    viewer_paths: dict[tuple[str, int], Path] = {}
    used_scenario_seeds: set[int] = set()

    for replicate, seed in enumerate(seeds):
        evolution_config = replace(base_config, seed=seed)

        def on_condition_start(
            condition: str,
            condition_evolution: EvolutionConfig,
            condition_world: TargetChoiceWorldConfig,
        ) -> None:
            if live_viewer is None:
                return
            live_viewer.begin_run(
                condition_evolution,
                condition_world,
                experiment="choose_target",
                label=(
                    f"target choice | {condition} | seed {seed} | "
                    f"pair {replicate + 1}/{len(seeds)}"
                ),
            )

        def on_generation(condition: str, stats: GenerationStats) -> None:
            print(
                f"{condition},{replicate},{seed},{_format_stats(stats)}",
                flush=True,
            )

        def on_validation(
            condition: str,
            stats: GenerationStats,
            validation: TargetValidationRecord,
            replays: list[RankedReplay],
            selected: bool,
            selected_generation: int,
        ) -> None:
            print(
                "  validation: "
                f"condition={condition}, "
                f"success={validation.overall_success_rate:.0%}, "
                f"cueA={validation.cue_a_success_rate:.0%}, "
                f"cueB={validation.cue_b_success_rate:.0%}, "
                f"alignment={validation.pair_alignment:+.0%}, "
                f"contrast={validation.pair_contrast_rate:.0%}, "
                f"worst={validation.worst_category_success_rate:.0%}, "
                f"wrong={validation.wrong_choice_rate:.0%}"
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
            base_world_config=base_world_config,
            validation_episodes=args.validation_episodes,
            holdout_episodes=args.holdout_episodes,
            on_condition_start=on_condition_start,
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
                f"Scenario seeds overlap between replicates: {sorted(overlap)[:5]}"
            )
        used_scenario_seeds.update(scenario_seeds)
        replicate_results.append(result)

        for condition_result in (result.recurrent, result.stateless):
            condition = condition_result.record.condition
            condition_dir = run_dir / "runs" / f"seed-{seed}" / condition
            knockout_records = (
                result.knockout_records if condition == "recurrent" else None
            )
            viewer_path = _write_condition_artifacts(
                condition_dir,
                condition_result,
                result,
                knockout_records,
                include_visualization=not args.no_visualization,
            )
            if viewer_path is not None:
                viewer_paths[(condition, seed)] = viewer_path

        print(
            f"result,{replicate},{seed},"
            f"recurrent={result.pair_record.recurrent_holdout_success_rate:.2%},"
            f"stateless={result.pair_record.stateless_holdout_success_rate:.2%},"
            f"advantage={result.pair_record.recurrent_advantage:+.2%},"
            f"knockout={result.pair_record.test_time_knockout_success_rate:.2%}",
            flush=True,
        )

    condition_results = [
        condition
        for result in replicate_results
        for condition in (result.recurrent, result.stateless)
    ]
    ranked_results, ranked_replays = _rank_results(
        condition_results,
        max(args.live_samples, 4),
    )
    best_recurrent = max(
        (result.recurrent for result in replicate_results),
        key=lambda item: (
            item.record.holdout_success_rate,
            item.record.holdout_average_score,
        ),
    )
    comparison_viewer_path = None
    if not args.no_visualization:
        comparison_viewer_path = write_ranked_visualization(
            run_dir=run_dir,
            ranked_replays=ranked_replays,
            history=best_recurrent.training.history,
            evolution_config=best_recurrent.evolution_config,
            world_config=best_recurrent.world_config,
            experiment="choose_target",
            label="target-rule memory across seeds",
            diagnostics=[
                asdict(item) for item in best_recurrent.training.validation_history
            ],
            selected_stats=best_recurrent.training.selected_stats,
            status_message=(
                "Best recurrent and stateless runs on one shared showcase scene"
            ),
        )

    comparison_records = [result.record for result in condition_results]
    pair_records = [result.pair_record for result in replicate_results]
    category_records = [
        category
        for result in condition_results
        for category in result.category_records
    ]
    _write_csv(run_dir / "comparison.csv", comparison_records)
    _write_csv(run_dir / "pairs.csv", pair_records)
    _write_csv(run_dir / "categories.csv", category_records)
    _write_report(
        run_dir / "report.md",
        comparison_records,
        pair_records,
        category_records,
        viewer_paths,
        comparison_viewer_path,
    )

    if live_viewer is not None:
        live_viewer.finish(
            best_recurrent.training.selected_stats,
            ranked_replays[0].episode,
            comparison_viewer_path,
            ranked_replays=ranked_replays,
            final_label="target-rule memory across seeds",
            message=f"Comparison complete: {len(seeds)} paired seeds",
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
    base_world_config: TargetChoiceWorldConfig,
    validation_episodes: int,
    holdout_episodes: int,
    on_condition_start: (
        Callable[[str, EvolutionConfig, TargetChoiceWorldConfig], None] | None
    ) = None,
    on_generation: Callable[[str, GenerationStats], None] | None = None,
    on_validation: (
        Callable[
            [
                str,
                GenerationStats,
                TargetValidationRecord,
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
) -> TargetReplicateResult:
    scenario_namespace = _scenario_namespace(seed)
    training_scenario_sets = _balanced_target_scenario_sets(
        base_world_config,
        count=evolution_config.episodes_per_genome,
        first_seed=scenario_namespace,
    )
    training_scenarios = _flatten_target_scenario_sets(training_scenario_sets)
    validation_scenarios = _balanced_target_scenarios(
        base_world_config,
        count=validation_episodes,
        first_seed=scenario_namespace + VALIDATION_SEED_OFFSET,
    )
    holdout_scenarios = _balanced_target_scenarios(
        base_world_config,
        count=holdout_episodes,
        first_seed=scenario_namespace + HOLDOUT_SEED_OFFSET,
    )
    split_seeds = tuple(
        scenario.seed
        for scenarios in (
            training_scenarios,
            validation_scenarios,
            holdout_scenarios,
        )
        for scenario in scenarios
    )
    if len(set(split_seeds)) != len(split_seeds):
        raise RuntimeError("Training, validation, and holdout seeds must be disjoint.")

    condition_results: dict[str, TargetConditionResult] = {}
    for condition in ("recurrent", "stateless"):
        world_config = replace(
            base_world_config,
            use_recurrence=condition == "recurrent",
        )
        if on_condition_start is not None:
            on_condition_start(condition, evolution_config, world_config)

        def generation_callback(stats: GenerationStats) -> None:
            if on_generation is not None:
                on_generation(condition, stats)

        def validation_callback(
            stats: GenerationStats,
            validation: TargetValidationRecord,
            replays: list[RankedReplay],
            selected: bool,
            selected_generation: int,
        ) -> None:
            if on_validation is not None:
                on_validation(
                    condition,
                    stats,
                    validation,
                    replays,
                    selected,
                    selected_generation,
                )

        condition_results[condition] = _run_condition(
            condition=condition,
            replicate=replicate,
            seed=seed,
            evolution_config=evolution_config,
            world_config=world_config,
            training_scenarios=training_scenarios,
            training_scenario_sets=training_scenario_sets,
            validation_scenarios=validation_scenarios,
            holdout_scenarios=holdout_scenarios,
            on_generation=generation_callback,
            on_validation=validation_callback,
            on_evaluation=on_evaluation,
            trace_sample_size=trace_sample_size,
        )

    recurrent = condition_results["recurrent"]
    stateless = condition_results["stateless"]
    knockout_config = replace(recurrent.world_config, use_recurrence=False)
    knockout_records, _ = _evaluate_scenarios(
        recurrent.training.best_validation_genome,
        knockout_config,
        holdout_scenarios,
        split="holdout-knockout",
        record_replays=False,
    )
    pair_record = _pair_record(
        replicate,
        seed,
        recurrent.holdout_records,
        stateless.holdout_records,
        knockout_records,
    )
    return TargetReplicateResult(
        replicate=replicate,
        seed=seed,
        training_scenarios=training_scenarios,
        training_scenario_sets=training_scenario_sets,
        validation_scenarios=validation_scenarios,
        holdout_scenarios=holdout_scenarios,
        recurrent=recurrent,
        stateless=stateless,
        knockout_records=knockout_records,
        pair_record=pair_record,
    )


def _run_condition(
    *,
    condition: str,
    replicate: int,
    seed: int,
    evolution_config: EvolutionConfig,
    world_config: TargetChoiceWorldConfig,
    training_scenarios: tuple[BalancedTargetScenario, ...],
    training_scenario_sets: tuple[tuple[BalancedTargetScenario, ...], ...],
    validation_scenarios: tuple[BalancedTargetScenario, ...],
    holdout_scenarios: tuple[BalancedTargetScenario, ...],
    on_generation: Callable[[GenerationStats], None] | None = None,
    on_validation: (
        Callable[
            [GenerationStats, TargetValidationRecord, list[RankedReplay], bool, int],
            None,
        ]
        | None
    ) = None,
    on_evaluation: Callable[[EvaluationProgress], None] | None = None,
    trace_sample_size: int = 0,
) -> TargetConditionResult:
    training = _run_training_with_validation(
        evolution_config=evolution_config,
        world_config=world_config,
        training_scenarios=training_scenario_sets[0],
        training_scenario_sets=training_scenario_sets,
        validation_scenarios=validation_scenarios,
        on_generation=on_generation,
        on_validation=on_validation,
        on_evaluation=on_evaluation,
        trace_sample_size=trace_sample_size,
        trace_episode_indices=(
            _live_trace_episode_indices(training_scenario_sets[0], trace_sample_size)
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
        condition,
        replicate,
        seed,
        holdout_records,
    )
    record = _comparison_record(
        condition,
        replicate,
        seed,
        training,
        training_records,
        holdout_records,
        category_records,
    )
    return TargetConditionResult(
        record=record,
        evolution_config=evolution_config,
        world_config=world_config,
        training=training,
        training_records=training_records,
        holdout_records=holdout_records,
        holdout_replays=holdout_replays,
        category_records=category_records,
    )


def _comparison_record(
    condition: str,
    replicate: int,
    seed: int,
    training: TargetTrainingResult,
    training_records: list[TargetEvaluationRecord],
    holdout_records: list[TargetEvaluationRecord],
    category_records: list[TargetCategoryRecord],
) -> TargetComparisonRecord:
    training_summary = _summarize_records(training_records)
    holdout_summary = _summarize_records(holdout_records)
    validation = training.selected_validation
    worst = min(
        category_records,
        key=lambda item: (
            item.success_rate,
            item.correct_target,
            item.target_a_sector,
        ),
    )
    return TargetComparisonRecord(
        condition=condition,
        replicate=replicate,
        seed=seed,
        selected_generation=training.selected_stats.generation,
        training_average_score=training_summary["average_score"],
        training_success_rate=training_summary["success_rate"],
        validation_average_score=validation.average_score,
        validation_success_rate=validation.overall_success_rate,
        validation_cue_a_success_rate=validation.cue_a_success_rate,
        validation_cue_b_success_rate=validation.cue_b_success_rate,
        validation_pair_alignment=validation.pair_alignment,
        validation_pair_contrast_rate=validation.pair_contrast_rate,
        holdout_average_score=holdout_summary["average_score"],
        holdout_success_rate=holdout_summary["success_rate"],
        holdout_cue_a_success_rate=holdout_summary["cue_a_success_rate"],
        holdout_cue_b_success_rate=holdout_summary["cue_b_success_rate"],
        holdout_cue_gap=holdout_summary["cue_gap"],
        holdout_pair_alignment=holdout_summary["pair_alignment"],
        holdout_pair_contrast_rate=holdout_summary["pair_contrast_rate"],
        holdout_wrong_choice_rate=holdout_summary["wrong_choice_rate"],
        holdout_no_choice_rate=holdout_summary["no_choice_rate"],
        generalization_gap=(
            training_summary["success_rate"] - holdout_summary["success_rate"]
        ),
        worst_category=_category_label(worst),
        worst_category_success_rate=worst.success_rate,
    )


def _pair_record(
    replicate: int,
    seed: int,
    recurrent_records: list[TargetEvaluationRecord],
    stateless_records: list[TargetEvaluationRecord],
    knockout_records: list[TargetEvaluationRecord],
) -> TargetPairRecord:
    recurrent_by_seed = {item.seed: item for item in recurrent_records}
    stateless_by_seed = {item.seed: item for item in stateless_records}
    knockout_by_seed = {item.seed: item for item in knockout_records}
    if not (
        recurrent_by_seed.keys()
        == stateless_by_seed.keys()
        == knockout_by_seed.keys()
    ):
        raise ValueError("Paired conditions must use identical holdout seeds.")

    recurrent_rate = mean(item.success for item in recurrent_records)
    stateless_rate = mean(item.success for item in stateless_records)
    knockout_rate = mean(item.success for item in knockout_records)
    recurrent_only = sum(
        recurrent_by_seed[item_seed].success
        and not stateless_by_seed[item_seed].success
        for item_seed in recurrent_by_seed
    )
    stateless_only = sum(
        stateless_by_seed[item_seed].success
        and not recurrent_by_seed[item_seed].success
        for item_seed in recurrent_by_seed
    )
    return TargetPairRecord(
        replicate=replicate,
        seed=seed,
        recurrent_holdout_success_rate=recurrent_rate,
        stateless_holdout_success_rate=stateless_rate,
        recurrent_advantage=recurrent_rate - stateless_rate,
        recurrent_only_successes=recurrent_only,
        stateless_only_successes=stateless_only,
        test_time_knockout_success_rate=knockout_rate,
        test_time_knockout_drop=recurrent_rate - knockout_rate,
    )


def _category_records(
    condition: str,
    replicate: int,
    seed: int,
    records: list[TargetEvaluationRecord],
) -> list[TargetCategoryRecord]:
    grouped: dict[tuple[str, str], list[TargetEvaluationRecord]] = {}
    for record in records:
        key = (record.correct_target, record.target_a_sector)
        grouped.setdefault(key, []).append(record)

    result = []
    for (correct_target, target_a_sector), items in sorted(grouped.items()):
        successes = sum(item.success for item in items)
        wrong_choices = sum(item.outcome == "wrong" for item in items)
        no_choices = sum(item.outcome == "no_choice" for item in items)
        result.append(
            TargetCategoryRecord(
                condition=condition,
                replicate=replicate,
                seed=seed,
                correct_target=correct_target,
                target_a_side=items[0].target_a_side,
                target_a_sector=target_a_sector,
                successes=successes,
                wrong_choices=wrong_choices,
                no_choices=no_choices,
                episodes=len(items),
                success_rate=successes / len(items),
            )
        )
    return result


def _rank_results(
    results: list[TargetConditionResult],
    limit: int,
) -> tuple[list[TargetConditionResult], list[RankedReplay]]:
    grouped = {
        condition: sorted(
            [item for item in results if item.record.condition == condition],
            key=lambda item: (
                item.record.holdout_success_rate,
                item.record.holdout_average_score,
            ),
            reverse=True,
        )
        for condition in ("recurrent", "stateless")
    }
    selected = []
    index = 0
    while len(selected) < limit:
        added = False
        for condition in ("recurrent", "stateless"):
            condition_results = grouped[condition]
            if index < len(condition_results) and len(selected) < limit:
                selected.append(condition_results[index])
                added = True
        if not added:
            break
        index += 1

    replays = []
    for rank, result in enumerate(selected, start=1):
        episode = TargetChoiceWorld(result.world_config).evaluate(
            result.training.best_validation_genome,
            random.Random(SHOWCASE_SEED),
            record_trace=True,
        )
        replays.append(
            RankedReplay(
                rank=rank,
                score=result.record.holdout_average_score,
                eat_rate=result.record.holdout_success_rate,
                episode=episode,
                label=f"{result.record.condition} | seed {result.record.seed}",
                metric="holdout",
            )
        )
    return selected, replays


def _write_condition_artifacts(
    run_dir: Path,
    result: TargetConditionResult,
    replicate_result: TargetReplicateResult,
    knockout_records: list[TargetEvaluationRecord] | None,
    *,
    include_visualization: bool,
) -> Path | None:
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "config.json",
        {
            "experiment": "choose_target",
            "condition": result.record.condition,
            "replicate": result.record.replicate,
            "evolution": asdict(result.evolution_config),
            "world": asdict(result.world_config),
            "fitness": {
                "pair_alignment_weight": PAIR_ALIGNMENT_FITNESS_WEIGHT,
                "pair_contrast_weight": PAIR_CONTRAST_FITNESS_WEIGHT,
                "worst_category_success_weight": (
                    WORST_CATEGORY_FITNESS_WEIGHT
                ),
                "worst_category_score_weight": WORST_CATEGORY_SCORE_WEIGHT,
                "training_scenario_sets": TARGET_TRAINING_BANKS,
            },
            "training_scenario_sets": [
                [asdict(item) for item in scenarios]
                for scenarios in replicate_result.training_scenario_sets
            ],
            "validation_scenarios": [
                asdict(item) for item in replicate_result.validation_scenarios
            ],
            "holdout_scenarios": [
                asdict(item) for item in replicate_result.holdout_scenarios
            ],
        },
    )
    _write_metrics(run_dir / "metrics.csv", result.training.history)
    _write_records(
        run_dir / "validation.csv",
        TargetValidationRecord,
        result.training.validation_history,
    )
    _write_records(
        run_dir / "validation_episodes.csv",
        TargetValidationEpisodeRecord,
        result.training.validation_episode_records,
    )
    _write_records(
        run_dir / "evaluation.csv",
        TargetEvaluationRecord,
        result.training_records + result.holdout_records,
    )
    if knockout_records is not None:
        _write_records(
            run_dir / "knockout_evaluation.csv",
            TargetEvaluationRecord,
            knockout_records,
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

    viewer_path = None
    validation_viewer_path = None
    if include_visualization:
        diagnostics = [asdict(item) for item in result.training.validation_history]
        viewer_path = write_visualization(
            run_dir=run_dir,
            best_genome=result.training.best_validation_genome,
            history=result.training.history,
            evolution_config=result.evolution_config,
            world_config=result.world_config,
            seed=result.holdout_replays[0].seed,
            world_class=TargetChoiceWorld,
            experiment="choose_target",
            replay_episodes=result.holdout_replays,
            diagnostics=diagnostics,
            selected_generation=result.training.selected_stats.generation,
        )
        validation_viewer_path = write_ranked_visualization(
            run_dir=run_dir,
            ranked_replays=result.training.selected_replays,
            history=result.training.history,
            evolution_config=result.evolution_config,
            world_config=result.world_config,
            experiment="choose_target",
            label=(
                f"{result.record.condition} | seed {result.record.seed} | "
                f"validation generation "
                f"{result.training.selected_stats.generation + 1}/"
                f"{result.evolution_config.generations}"
            ),
            diagnostics=diagnostics,
            viewer_filename="validation-viewer.html",
            data_filename="validation-replay.json",
            selected_stats=result.training.selected_stats,
        )

    _write_condition_report(
        run_dir / "report.md",
        result.record.condition,
        result.evolution_config,
        result.training_records,
        result.holdout_records,
        result.training.selected_validation,
        result.training.validation_history[-1],
        result.training.selected_stats,
        len(replicate_result.validation_scenarios),
        viewer_path,
        validation_viewer_path,
    )
    if knockout_records is not None:
        knockout_summary = _summarize_records(knockout_records)
        with (run_dir / "report.md").open("a", encoding="utf-8") as file:
            file.write(
                "\n## Test-time knockout\n\n"
                "Тот же recurrent-победитель повторно проверен без переноса "
                "hidden-state.\n\n"
                f"- Success: {knockout_summary['success_rate']:.2%}\n"
                f"- Cue A: {knockout_summary['cue_a_success_rate']:.2%}\n"
                f"- Cue B: {knockout_summary['cue_b_success_rate']:.2%}\n"
                f"- Pair alignment: {knockout_summary['pair_alignment']:+.2%}\n"
                f"- Pair contrast: {knockout_summary['pair_contrast_rate']:.2%}\n"
            )
    return viewer_path


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
    records: list[TargetComparisonRecord],
    pairs: list[TargetPairRecord],
    categories: list[TargetCategoryRecord],
    viewer_paths: dict[tuple[str, int], Path],
    comparison_viewer_path: Path | None,
) -> None:
    recurrent = [item for item in records if item.condition == "recurrent"]
    stateless = [item for item in records if item.condition == "stateless"]
    recurrent_rates = [item.holdout_success_rate for item in recurrent]
    stateless_rates = [item.holdout_success_rate for item in stateless]
    advantages = [item.recurrent_advantage for item in pairs]
    knockout_rates = [item.test_time_knockout_success_rate for item in pairs]
    knockout_drops = [item.test_time_knockout_drop for item in pairs]
    cue_gaps = [item.holdout_cue_gap for item in recurrent]
    recurrent_alignments = [item.holdout_pair_alignment for item in recurrent]
    stateless_alignments = [item.holdout_pair_alignment for item in stateless]
    recurrent_contrasts = [item.holdout_pair_contrast_rate for item in recurrent]
    stateless_contrasts = [item.holdout_pair_contrast_rate for item in stateless]
    recurrent_categories = _aggregate_categories(categories, "recurrent")
    worst_category = min(
        recurrent_categories,
        key=lambda item: (item["success_rate"], item["label"]),
    )
    required_passes = math.ceil(len(recurrent) * TARGET_PASS_FRACTION)
    passing_runs = sum(rate >= TARGET_RECURRENT_RUN for rate in recurrent_rates)
    recurrent_wins = sum(item.recurrent_advantage > 0 for item in pairs)
    checks = {
        "recurrent_mean": mean(recurrent_rates) >= TARGET_RECURRENT_MEAN,
        "passing_runs": passing_runs >= required_passes,
        "stateless_max": mean(stateless_rates) <= TARGET_STATELESS_MAX,
        "advantage": mean(advantages) >= TARGET_RECURRENT_ADVANTAGE,
        "paired_wins": recurrent_wins >= required_passes,
        "cue_gap": mean(cue_gaps) <= TARGET_CUE_GAP,
        "knockout_drop": mean(knockout_drops) >= TARGET_KNOCKOUT_DROP,
        "worst_category": (
            float(worst_category["success_rate"]) >= TARGET_WORST_CATEGORY
        ),
    }
    passed = all(checks.values())
    lines = [
        "# Выбор цели на разных seed",
        "",
        f"**Вердикт: {'ЭТАП ПРОЙДЕН' if passed else 'НУЖНЫ ЕЩЕ УЛУЧШЕНИЯ'}**",
        "",
        "Каждый seed создает независимую начальную популяцию и собственные",
        "непересекающиеся training, validation и holdout-сцены. Внутри пары",
        "recurrent и stateless получают одинаковые начальные веса и сцены.",
        "",
        "## Сводка",
        "",
        "| Метрика | Значение | Цель | Результат |",
        "| --- | ---: | ---: | --- |",
        _target_row(
            "Mean recurrent holdout",
            mean(recurrent_rates),
            f">= {TARGET_RECURRENT_MEAN:.0%}",
            checks["recurrent_mean"],
        ),
        (
            f"| Recurrent runs above {TARGET_RECURRENT_RUN:.0%} | "
            f"{passing_runs}/{len(recurrent)} | >= {required_passes}/{len(recurrent)} | "
            f"{_status(checks['passing_runs'])} |"
        ),
        _target_row(
            "Mean stateless holdout",
            mean(stateless_rates),
            f"<= {TARGET_STATELESS_MAX:.0%}",
            checks["stateless_max"],
        ),
        _target_row(
            "Mean recurrent advantage",
            mean(advantages),
            f">= {TARGET_RECURRENT_ADVANTAGE:.0%}",
            checks["advantage"],
        ),
        (
            f"| Paired recurrent wins | {recurrent_wins}/{len(pairs)} | "
            f">= {required_passes}/{len(pairs)} | "
            f"{_status(checks['paired_wins'])} |"
        ),
        _target_row(
            "Mean recurrent cue gap",
            mean(cue_gaps),
            f"<= {TARGET_CUE_GAP:.0%}",
            checks["cue_gap"],
        ),
        _target_row(
            "Mean test-time knockout drop",
            mean(knockout_drops),
            f">= {TARGET_KNOCKOUT_DROP:.0%}",
            checks["knockout_drop"],
        ),
        _target_row(
            f"Worst recurrent category ({worst_category['label']})",
            float(worst_category["success_rate"]),
            f">= {TARGET_WORST_CATEGORY:.0%}",
            checks["worst_category"],
        ),
        "",
        (
            f"Recurrent holdout: mean {mean(recurrent_rates):.2%}, "
            f"standard deviation {pstdev(recurrent_rates):.2%}, "
            f"range {min(recurrent_rates):.2%}-{max(recurrent_rates):.2%}."
        ),
        (
            f"Stateless holdout: mean {mean(stateless_rates):.2%}, "
            f"standard deviation {pstdev(stateless_rates):.2%}, "
            f"range {min(stateless_rates):.2%}-{max(stateless_rates):.2%}."
        ),
        (
            f"Test-time knockout: mean {mean(knockout_rates):.2%}; "
            f"drop from recurrent {mean(knockout_drops):+.2%}."
        ),
        (
            f"Pair alignment: recurrent {mean(recurrent_alignments):+.2%}, "
            f"stateless {mean(stateless_alignments):+.2%}."
        ),
        (
            f"Pair contrast: recurrent {mean(recurrent_contrasts):.2%}, "
            f"stateless {mean(stateless_contrasts):.2%}."
        ),
        "",
        "## Парные запуски",
        "",
        "| Seed | Recurrent | Stateless | Advantage | R-only | S-only | Knockout | Drop |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in sorted(pairs, key=lambda item: item.seed):
        lines.append(
            f"| {pair.seed} | {pair.recurrent_holdout_success_rate:.2%} | "
            f"{pair.stateless_holdout_success_rate:.2%} | "
            f"{pair.recurrent_advantage:+.2%} | "
            f"{pair.recurrent_only_successes} | "
            f"{pair.stateless_only_successes} | "
            f"{pair.test_time_knockout_success_rate:.2%} | "
            f"{pair.test_time_knockout_drop:+.2%} |"
        )

    lines.extend(
        [
            "",
            "## Holdout по категориям",
            "",
            "| Condition | Cue | A sector | Success | Wrong | No choice | Rate |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in ("recurrent", "stateless"):
        for row in _aggregate_categories(categories, condition):
            lines.append(
                f"| {condition} | {row['correct_target']} | "
                f"{row['target_a_sector']} | {row['successes']}/{row['episodes']} | "
                f"{row['wrong_choices']} | {row['no_choices']} | "
                f"{row['success_rate']:.2%} |"
            )

    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "Retraining stateless показывает, может ли эволюция решить задачу без",
            "памяти другой стратегией. Test-time knockout проверяет причинную",
            "зависимость уже найденных recurrent-мозгов от переноса hidden-state.",
            "Holdout не участвует ни в обучении, ни в выборе checkpoint.",
            "",
        ]
    )
    if comparison_viewer_path is not None:
        relative = comparison_viewer_path.relative_to(path.parent).as_posix()
        lines.extend(
            [
                "## Viewer",
                "",
                f"[Лучшие recurrent и stateless запуски]({relative})",
                "",
            ]
        )
    if viewer_paths:
        lines.extend(["## Отдельные запуски", ""])
        for (condition, seed), viewer_path in sorted(viewer_paths.items()):
            relative = viewer_path.relative_to(path.parent).as_posix()
            lines.append(f"- {condition}, seed {seed}: [{relative}]({relative})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _aggregate_categories(
    categories: list[TargetCategoryRecord],
    condition: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[TargetCategoryRecord]] = {}
    for record in categories:
        if record.condition != condition:
            continue
        key = (record.correct_target, record.target_a_sector)
        grouped.setdefault(key, []).append(record)

    rows = []
    for (correct_target, target_a_sector), items in sorted(grouped.items()):
        successes = sum(item.successes for item in items)
        wrong_choices = sum(item.wrong_choices for item in items)
        no_choices = sum(item.no_choices for item in items)
        episodes = sum(item.episodes for item in items)
        rows.append(
            {
                "label": f"Cue {correct_target}|A {target_a_sector}",
                "correct_target": correct_target,
                "target_a_sector": target_a_sector,
                "successes": successes,
                "wrong_choices": wrong_choices,
                "no_choices": no_choices,
                "episodes": episodes,
                "success_rate": successes / episodes,
            }
        )
    return rows


def _target_row(name: str, value: float, target: str, passed: bool) -> str:
    return f"| {name} | {value:.2%} | {target} | {_status(passed)} |"


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _category_label(record: TargetCategoryRecord) -> str:
    return f"Cue {record.correct_target}|A {record.target_a_sector}"


def _scenario_namespace(seed: int) -> int:
    namespace_index = seed * 2 if seed >= 0 else -seed * 2 - 1
    return namespace_index * SCENARIO_NAMESPACE_SIZE


def _start_live_viewer(
    args: argparse.Namespace,
    evolution_config: EvolutionConfig,
    world_config: TargetChoiceWorldConfig,
    seeds: tuple[int, ...],
) -> LiveRunViewer | None:
    if not args.open_viewer:
        return None
    try:
        viewer = LiveRunViewer(
            evolution_config=evolution_config,
            world_config=world_config,
            sample_limit=max(args.live_samples, 4),
            experiment="choose_target",
            label=f"target choice | recurrent | seed {seeds[0]}",
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
