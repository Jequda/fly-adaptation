from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
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
    PAIR_ALIGNMENT_FITNESS_WEIGHT,
    PAIR_CONTRAST_FITNESS_WEIGHT,
    TARGET_SECTORS,
    WORST_CATEGORY_FITNESS_WEIGHT,
    WORST_CATEGORY_SCORE_WEIGHT,
    TargetChoiceWorld,
    TargetChoiceWorldConfig,
)


TARGET_HOLDOUT_SUCCESS = 0.80
TARGET_CUE_GAP = 0.20
TARGET_WORST_CATEGORY = 0.60
TARGET_CATEGORY_COUNT = 2 * len(TARGET_SECTORS)
TARGET_TRAINING_BANKS = 16
TARGET_TRAINING_BANK_SEED_STEP = 100_003


@dataclass(frozen=True)
class BalancedTargetScenario:
    index: int
    seed: int
    geometry_key: int
    correct_target: str
    target_a_side: str
    target_a_sector: str

    @property
    def label(self) -> str:
        return f"cue {self.correct_target} | A {self.target_a_sector}"


@dataclass(frozen=True)
class TargetEvaluationRecord:
    split: str
    episode: int
    seed: int
    geometry_key: int
    correct_target: str
    target_a_side: str
    target_a_sector: str
    score: float
    success: bool
    choice: str | None
    outcome: str
    steps: int
    final_distance: float
    target_preference: float


@dataclass(frozen=True)
class TargetValidationRecord:
    generation: int
    average_score: float
    overall_success_rate: float
    cue_a_success_rate: float
    cue_b_success_rate: float
    cue_gap: float
    pair_alignment: float
    pair_contrast_rate: float
    target_a_negative_success_rate: float
    target_a_positive_success_rate: float
    worst_category: str
    worst_category_success_rate: float
    wrong_choice_rate: float
    no_choice_rate: float
    average_success_steps: float


@dataclass(frozen=True)
class TargetValidationEpisodeRecord:
    generation: int
    scenario: int
    seed: int
    geometry_key: int
    correct_target: str
    target_a_side: str
    target_a_sector: str
    score: float
    success: bool
    choice: str | None
    outcome: str
    steps: int
    final_distance: float
    target_preference: float


@dataclass(frozen=True)
class TargetTrainingResult:
    best_training_genome: BrainGenome
    best_validation_genome: BrainGenome
    history: list[GenerationStats]
    validation_history: list[TargetValidationRecord]
    validation_episode_records: list[TargetValidationEpisodeRecord]
    selected_validation: TargetValidationRecord
    selected_stats: GenerationStats
    selected_replays: list[RankedReplay]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remember a transient cue and choose between two targets.",
    )
    add_evolution_arguments(parser, output=Path("results/choose_target"))
    parser.set_defaults(population=80, generations=60, episodes=32, max_steps=80)
    parser.add_argument("--validation-episodes", type=int, default=64)
    parser.add_argument("--holdout-episodes", type=int, default=64)
    parser.add_argument("--cue-steps", type=int, default=6)
    parser.add_argument(
        "--wrong-target-penalty",
        type=float,
        default=TargetChoiceWorldConfig.wrong_target_penalty,
    )
    parser.add_argument(
        "--no-recurrence",
        action="store_true",
        help="Reset hidden state every step while keeping the same task.",
    )
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

    evolution_config = evolution_config_from_args(args)
    world_config = TargetChoiceWorldConfig(
        max_steps=args.max_steps,
        cue_steps=args.cue_steps,
        wrong_target_penalty=args.wrong_target_penalty,
        use_recurrence=not args.no_recurrence,
    )
    run_dir = _create_run_dir(args.output)
    training_scenario_sets = _balanced_target_scenario_sets(
        world_config,
        count=evolution_config.episodes_per_genome,
        first_seed=evolution_config.seed,
    )
    training_scenarios = _flatten_target_scenario_sets(training_scenario_sets)
    validation_scenarios = _balanced_target_scenarios(
        world_config,
        count=args.validation_episodes,
        first_seed=evolution_config.seed + 2_000_003,
    )
    holdout_scenarios = _balanced_target_scenarios(
        world_config,
        count=args.holdout_episodes,
        first_seed=evolution_config.seed + 4_000_003,
    )
    all_seeds = tuple(
        item.seed
        for scenarios in (training_scenarios, validation_scenarios, holdout_scenarios)
        for item in scenarios
    )
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("Training, validation, and holdout seeds must be disjoint.")

    _write_json(
        run_dir / "config.json",
        {
            "experiment": "choose_target",
            "condition": "recurrent" if world_config.use_recurrence else "stateless",
            "evolution": asdict(evolution_config),
            "world": asdict(world_config),
            "selection": {
                "primary": "minimum_validation_cue_sector_success_rate",
                "tie_breakers": [
                    "minimum_validation_cue_success_rate",
                    "validation_success_rate",
                    "validation_pair_alignment",
                    "validation_pair_contrast_rate",
                    "validation_average_score",
                ],
                "holdout_used": False,
            },
            "fitness": {
                "categories": "2 cues x 4 target sectors",
                "scenario_pairing": "identical geometry with opposite cues",
                "pair_alignment_weight": PAIR_ALIGNMENT_FITNESS_WEIGHT,
                "pair_contrast_weight": PAIR_CONTRAST_FITNESS_WEIGHT,
                "worst_category_success_weight": (
                    WORST_CATEGORY_FITNESS_WEIGHT
                ),
                "worst_category_score_weight": WORST_CATEGORY_SCORE_WEIGHT,
                "training_scenario_sets": TARGET_TRAINING_BANKS,
            },
            "targets": {
                "holdout_success_rate": TARGET_HOLDOUT_SUCCESS,
                "cue_gap": TARGET_CUE_GAP,
                "worst_category_success_rate": TARGET_WORST_CATEGORY,
            },
            "training_scenario_sets": [
                [asdict(item) for item in scenarios]
                for scenarios in training_scenario_sets
            ],
            "validation_scenarios": [asdict(item) for item in validation_scenarios],
            "holdout_scenarios": [asdict(item) for item in holdout_scenarios],
        },
    )

    condition = "recurrent" if world_config.use_recurrence else "stateless"
    print(f"Run: {run_dir}")
    live_viewer = _start_live_viewer(
        args,
        evolution_config,
        world_config,
        condition,
    )
    print(
        "generation,best_fitness,avg_fitness,best_success_rate,avg_hidden",
        flush=True,
    )

    def on_generation(stats: GenerationStats) -> None:
        print(_format_stats(stats), flush=True)

    def on_validation(
        stats: GenerationStats,
        validation: TargetValidationRecord,
        replays: list[RankedReplay],
        selected: bool,
        selected_generation: int,
    ) -> None:
        print(
            "  validation: "
            f"success={validation.overall_success_rate:.0%}, "
            f"cueA={validation.cue_a_success_rate:.0%}, "
            f"cueB={validation.cue_b_success_rate:.0%}, "
            f"alignment={validation.pair_alignment:+.0%}, "
            f"contrast={validation.pair_contrast_rate:.0%}, "
            f"worst={validation.worst_category_success_rate:.0%}, "
            f"wrong={validation.wrong_choice_rate:.0%}, "
            f"none={validation.no_choice_rate:.0%}"
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

    training = _run_training_with_validation(
        evolution_config=evolution_config,
        world_config=world_config,
        training_scenarios=training_scenario_sets[0],
        training_scenario_sets=training_scenario_sets,
        validation_scenarios=validation_scenarios,
        on_generation=on_generation,
        on_validation=on_validation,
        on_evaluation=live_viewer.record_evaluation if live_viewer else None,
        trace_sample_size=args.live_samples if live_viewer else 0,
        trace_episode_indices=(
            _live_trace_episode_indices(training_scenario_sets[0], args.live_samples)
            if live_viewer
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

    _write_metrics(run_dir / "metrics.csv", training.history)
    _write_records(run_dir / "evaluation.csv", TargetEvaluationRecord, training_records + holdout_records)
    _write_records(run_dir / "validation.csv", TargetValidationRecord, training.validation_history)
    _write_records(
        run_dir / "validation_episodes.csv",
        TargetValidationEpisodeRecord,
        training.validation_episode_records,
    )
    training.best_validation_genome.save_json(run_dir / "best_genome.json")
    training.best_validation_genome.save_json(run_dir / "best_validation_genome.json")
    training.best_training_genome.save_json(run_dir / "best_training_genome.json")
    _write_json(
        run_dir / "selection.json",
        {
            "selected_generation": training.selected_stats.generation,
            "training_metrics_at_selection": asdict(training.selected_stats),
            "validation_metrics_at_selection": asdict(training.selected_validation),
            "holdout_used_for_selection": False,
        },
    )

    diagnostics = [asdict(item) for item in training.validation_history]
    viewer_path = None
    validation_viewer_path = None
    if not args.no_visualization:
        viewer_path = write_visualization(
            run_dir=run_dir,
            best_genome=training.best_validation_genome,
            history=training.history,
            evolution_config=evolution_config,
            world_config=world_config,
            seed=holdout_scenarios[0].seed,
            world_class=TargetChoiceWorld,
            experiment="choose_target",
            replay_episodes=holdout_replays,
            diagnostics=diagnostics,
            selected_generation=training.selected_stats.generation,
        )
        validation_viewer_path = write_ranked_visualization(
            run_dir=run_dir,
            ranked_replays=training.selected_replays,
            history=training.history,
            evolution_config=evolution_config,
            world_config=world_config,
            experiment="choose_target",
            label=(
                f"{condition} target choice | checkpoint generation "
                f"{training.selected_stats.generation + 1}/"
                f"{evolution_config.generations}"
            ),
            diagnostics=diagnostics,
            viewer_filename="validation-viewer.html",
            data_filename="validation-replay.json",
            selected_stats=training.selected_stats,
        )

    _write_report(
        run_dir / "report.md",
        condition,
        evolution_config,
        training_records,
        holdout_records,
        training.selected_validation,
        training.validation_history[-1],
        training.selected_stats,
        len(validation_scenarios),
        viewer_path,
        validation_viewer_path,
    )

    if live_viewer is not None:
        live_viewer.finish(
            training.selected_stats,
            training.selected_replays[0].episode,
            validation_viewer_path or viewer_path,
            ranked_replays=training.selected_replays,
            message=(
                f"Training complete: {condition} checkpoint from generation "
                f"{training.selected_stats.generation + 1}/"
                f"{evolution_config.generations}"
            ),
        )
        live_viewer.wait_until_final_served()

    holdout_summary = _summarize_records(holdout_records)
    print()
    print(
        "Selected validation generation: "
        f"{training.selected_stats.generation + 1}/{evolution_config.generations}"
    )
    print(f"Validation success: {training.selected_validation.overall_success_rate:.2%}")
    print(f"Holdout success: {holdout_summary['success_rate']:.2%}")
    print(f"Holdout pair alignment: {holdout_summary['pair_alignment']:+.2%}")
    print(f"Holdout pair contrast: {holdout_summary['pair_contrast_rate']:.2%}")
    print(f"Holdout wrong choice: {holdout_summary['wrong_choice_rate']:.2%}")
    print(f"Holdout no choice: {holdout_summary['no_choice_rate']:.2%}")
    print(f"Saved results to: {run_dir}")
    if viewer_path is not None:
        print(f"Open visualization: {viewer_path}")
        print(f"Open validation: {validation_viewer_path}")
        if args.open_viewer and live_viewer is None:
            _open_in_browser(viewer_path.resolve().as_uri())


def _run_training_with_validation(
    *,
    evolution_config: EvolutionConfig,
    world_config: TargetChoiceWorldConfig,
    training_scenarios: tuple[BalancedTargetScenario, ...],
    training_scenario_sets: (
        tuple[tuple[BalancedTargetScenario, ...], ...] | None
    ) = None,
    validation_scenarios: tuple[BalancedTargetScenario, ...],
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
    trace_episode_indices: tuple[int, ...] | None = None,
) -> TargetTrainingResult:
    resolved_training_sets = training_scenario_sets or (training_scenarios,)
    if any(
        len(scenarios) != evolution_config.episodes_per_genome
        for scenarios in resolved_training_sets
    ):
        raise ValueError(
            "Every training scenario set must match episodes_per_genome."
        )
    validation_history: list[TargetValidationRecord] = []
    validation_episode_records: list[TargetValidationEpisodeRecord] = []
    selected_genome: BrainGenome | None = None
    selected_validation: TargetValidationRecord | None = None
    selected_stats: GenerationStats | None = None
    selected_replays: list[RankedReplay] = []
    selected_key: tuple[float, float, float, float, float, float] | None = None

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
        world_class=TargetChoiceWorld,
        trace_episode_indices=trace_episode_indices,
        episode_seed_sets=tuple(
            tuple(item.seed for item in scenarios)
            for scenarios in resolved_training_sets
        ),
    )
    if selected_genome is None or selected_validation is None or selected_stats is None:
        raise RuntimeError("Evolution completed without a validation checkpoint.")
    return TargetTrainingResult(
        best_training_genome=best_training_genome,
        best_validation_genome=selected_genome,
        history=history,
        validation_history=validation_history,
        validation_episode_records=validation_episode_records,
        selected_validation=selected_validation,
        selected_stats=selected_stats,
        selected_replays=selected_replays,
    )


def _balanced_target_scenarios(
    world_config: TargetChoiceWorldConfig,
    *,
    count: int,
    first_seed: int,
) -> tuple[BalancedTargetScenario, ...]:
    if count < TARGET_CATEGORY_COUNT or count % TARGET_CATEGORY_COUNT != 0:
        raise ValueError(
            "A balanced target set requires a positive multiple of "
            f"{TARGET_CATEGORY_COUNT} scenarios."
        )
    pairs_per_sector = count // TARGET_CATEGORY_COUNT
    found_pairs: dict[str, list[dict[str, int]]] = {
        sector: [] for sector in TARGET_SECTORS
    }
    pending: dict[tuple[str, int], dict[str, int]] = {}
    world = TargetChoiceWorld(world_config)
    seed = first_seed
    while sum(len(pairs) for pairs in found_pairs.values()) < count // 2:
        signature = world.scenario_signature_for_seed(seed)
        geometry_key = world.scenario_pair_key_for_seed(seed)
        sector = signature.target_a_sector
        if len(found_pairs[sector]) < pairs_per_sector:
            pair = pending.setdefault((sector, geometry_key), {})
            pair.setdefault(signature.correct_target, seed)
            if len(pair) == 2:
                found_pairs[sector].append(pair)
                del pending[(sector, geometry_key)]
        seed += 1
        if seed - first_seed > max(100_000, count * 20_000):
            raise RuntimeError("Could not build a balanced target scenario set.")

    scenarios = []
    for round_index in range(pairs_per_sector):
        for sector in TARGET_SECTORS:
            pair = found_pairs[sector][round_index]
            for correct_target in ("A", "B"):
                scenario_seed = pair[correct_target]
                scenarios.append(
                    BalancedTargetScenario(
                        index=len(scenarios),
                        seed=scenario_seed,
                        geometry_key=world.scenario_pair_key_for_seed(scenario_seed),
                        correct_target=correct_target,
                        target_a_side=(
                            "positive" if sector.endswith("upper") else "negative"
                        ),
                        target_a_sector=sector,
                    )
                )
    return tuple(scenarios)


def _balanced_target_scenario_sets(
    world_config: TargetChoiceWorldConfig,
    *,
    count: int,
    first_seed: int,
    set_count: int = TARGET_TRAINING_BANKS,
) -> tuple[tuple[BalancedTargetScenario, ...], ...]:
    if set_count < 1:
        raise ValueError("Training requires at least one scenario set.")
    scenario_sets = tuple(
        _balanced_target_scenarios(
            world_config,
            count=count,
            first_seed=first_seed + set_index * TARGET_TRAINING_BANK_SEED_STEP,
        )
        for set_index in range(set_count)
    )
    seeds = [item.seed for scenarios in scenario_sets for item in scenarios]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Target-choice training scenario sets must be disjoint.")
    return scenario_sets


def _flatten_target_scenario_sets(
    scenario_sets: tuple[tuple[BalancedTargetScenario, ...], ...],
) -> tuple[BalancedTargetScenario, ...]:
    return tuple(
        replace(scenario, index=index)
        for index, scenario in enumerate(
            scenario for scenarios in scenario_sets for scenario in scenarios
        )
    )


def _evaluate_validation(
    genome: BrainGenome,
    world_config: TargetChoiceWorldConfig,
    scenarios: tuple[BalancedTargetScenario, ...],
    *,
    generation: int,
) -> tuple[
    TargetValidationRecord,
    list[RankedReplay],
    list[TargetValidationEpisodeRecord],
]:
    world = TargetChoiceWorld(world_config)
    outcomes: list[tuple[BalancedTargetScenario, EpisodeResult]] = []
    replays = []
    episode_records = []
    for scenario in scenarios:
        episode = world.evaluate(genome, random.Random(scenario.seed), record_trace=True)
        outcomes.append((scenario, episode))
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
            TargetValidationEpisodeRecord(
                generation=generation,
                scenario=scenario.index,
                seed=scenario.seed,
                geometry_key=scenario.geometry_key,
                correct_target=scenario.correct_target,
                target_a_side=scenario.target_a_side,
                target_a_sector=scenario.target_a_sector,
                score=episode.score,
                success=episode.ate_food,
                choice=episode.choice,
                outcome=_outcome(episode),
                steps=episode.steps,
                final_distance=episode.final_distance,
                target_preference=_target_preference(episode),
            )
        )

    cue_a = [item for item in outcomes if item[0].correct_target == "A"]
    cue_b = [item for item in outcomes if item[0].correct_target == "B"]
    a_negative = [
        item for item in outcomes if item[0].target_a_side == "negative"
    ]
    a_positive = [
        item for item in outcomes if item[0].target_a_side == "positive"
    ]
    cue_a_rate = mean(item[1].ate_food for item in cue_a)
    cue_b_rate = mean(item[1].ate_food for item in cue_b)
    category_rates = {
        f"Cue {correct_target}|A {target_a_sector}": mean(
            episode.ate_food
            for scenario, episode in outcomes
            if scenario.correct_target == correct_target
            and scenario.target_a_sector == target_a_sector
        )
        for correct_target in ("A", "B")
        for target_a_sector in TARGET_SECTORS
    }
    worst_category, worst_category_rate = min(
        category_rates.items(),
        key=lambda item: (item[1], item[0]),
    )
    pair_alignment, pair_contrast_rate = _paired_metrics(
        [
            (
                scenario.geometry_key,
                scenario.correct_target,
                _target_preference(episode),
                episode.choice,
            )
            for scenario, episode in outcomes
        ]
    )
    successes = [item[1] for item in outcomes if item[1].ate_food]
    validation = TargetValidationRecord(
        generation=generation,
        average_score=mean(item[1].score for item in outcomes),
        overall_success_rate=mean(item[1].ate_food for item in outcomes),
        cue_a_success_rate=cue_a_rate,
        cue_b_success_rate=cue_b_rate,
        cue_gap=abs(cue_a_rate - cue_b_rate),
        pair_alignment=pair_alignment,
        pair_contrast_rate=pair_contrast_rate,
        target_a_negative_success_rate=mean(
            item[1].ate_food for item in a_negative
        ),
        target_a_positive_success_rate=mean(
            item[1].ate_food for item in a_positive
        ),
        worst_category=worst_category,
        worst_category_success_rate=worst_category_rate,
        wrong_choice_rate=mean(item[1].chose_correct is False for item in outcomes),
        no_choice_rate=mean(item[1].choice is None for item in outcomes),
        average_success_steps=(
            mean(item.steps for item in successes)
            if successes
            else float(world_config.max_steps)
        ),
    )
    return validation, replays, episode_records


def _evaluate_scenarios(
    genome: BrainGenome,
    world_config: TargetChoiceWorldConfig,
    scenarios: tuple[BalancedTargetScenario, ...],
    *,
    split: str,
    record_replays: bool,
) -> tuple[list[TargetEvaluationRecord], list[ReplayEpisode]]:
    world = TargetChoiceWorld(world_config)
    records = []
    replays = []
    for scenario in scenarios:
        episode = world.evaluate(
            genome,
            random.Random(scenario.seed),
            record_trace=record_replays,
        )
        records.append(
            TargetEvaluationRecord(
                split=split,
                episode=scenario.index,
                seed=scenario.seed,
                geometry_key=scenario.geometry_key,
                correct_target=scenario.correct_target,
                target_a_side=scenario.target_a_side,
                target_a_sector=scenario.target_a_sector,
                score=episode.score,
                success=episode.ate_food,
                choice=episode.choice,
                outcome=_outcome(episode),
                steps=episode.steps,
                final_distance=episode.final_distance,
                target_preference=_target_preference(episode),
            )
        )
        if record_replays:
            replays.append(
                ReplayEpisode(index=scenario.index, seed=scenario.seed, episode=episode)
            )
    return records, replays


def _representative_validation_replays(
    replays: list[RankedReplay],
    scenarios: tuple[BalancedTargetScenario, ...],
) -> list[RankedReplay]:
    if len(replays) != len(scenarios):
        raise ValueError("Validation replays and scenarios must have equal lengths.")
    selected = []
    seen: set[tuple[str, str]] = set()
    for replay, scenario in zip(replays, scenarios):
        signature = (scenario.correct_target, scenario.target_a_sector)
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(replay)
    return selected


def _live_trace_episode_indices(
    scenarios: tuple[BalancedTargetScenario, ...],
    sample_count: int,
) -> tuple[int, ...]:
    representatives = []
    seen: set[tuple[str, str]] = set()
    for index, scenario in enumerate(scenarios):
        signature = (scenario.correct_target, scenario.target_a_sector)
        if signature not in seen:
            seen.add(signature)
            representatives.append(index)
    return tuple(
        representatives[index % len(representatives)] for index in range(sample_count)
    )


def _validation_selection_key(
    record: TargetValidationRecord,
) -> tuple[float, float, float, float, float, float]:
    return (
        record.worst_category_success_rate,
        min(record.cue_a_success_rate, record.cue_b_success_rate),
        record.overall_success_rate,
        record.pair_alignment,
        record.pair_contrast_rate,
        record.average_score,
    )


def _target_preference(episode: EpisodeResult) -> float:
    if episode.target_preference is None:
        raise ValueError("Target-choice episode is missing target preference.")
    return episode.target_preference


def _paired_metrics(
    items: list[tuple[int, str, float, str | None]],
) -> tuple[float, float]:
    pairs: dict[int, dict[str, tuple[float, str | None]]] = {}
    for geometry_key, correct_target, preference, choice in items:
        pair = pairs.setdefault(geometry_key, {})
        if correct_target in pair:
            raise ValueError("A paired target set contains a duplicate cue.")
        pair[correct_target] = (preference, choice)
    if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Each target geometry must have one Cue A and one Cue B.")

    alignments = [
        (pair["A"][0] - pair["B"][0]) / 2.0
        for pair in pairs.values()
    ]
    contrasts = [
        {pair["A"][1], pair["B"][1]} == {"A", "B"}
        for pair in pairs.values()
    ]
    return mean(alignments), mean(contrasts)


def _outcome(episode: EpisodeResult) -> str:
    if episode.chose_correct is True:
        return "correct"
    if episode.chose_correct is False:
        return "wrong"
    return "no_choice"


def _summarize_records(
    records: list[TargetEvaluationRecord],
) -> dict[str, float]:
    cue_a = [item for item in records if item.correct_target == "A"]
    cue_b = [item for item in records if item.correct_target == "B"]
    cue_a_rate = mean(item.success for item in cue_a)
    cue_b_rate = mean(item.success for item in cue_b)
    category_rates = [
        mean(
            item.success
            for item in records
            if item.correct_target == correct_target
            and item.target_a_sector == target_a_sector
        )
        for correct_target in ("A", "B")
        for target_a_sector in TARGET_SECTORS
    ]
    pair_alignment, pair_contrast_rate = _paired_metrics(
        [
            (
                item.geometry_key,
                item.correct_target,
                item.target_preference,
                item.choice,
            )
            for item in records
        ]
    )
    successes = [item for item in records if item.success]
    return {
        "average_score": mean(item.score for item in records),
        "success_rate": mean(item.success for item in records),
        "cue_a_success_rate": cue_a_rate,
        "cue_b_success_rate": cue_b_rate,
        "cue_gap": abs(cue_a_rate - cue_b_rate),
        "pair_alignment": pair_alignment,
        "pair_contrast_rate": pair_contrast_rate,
        "worst_category_success_rate": min(category_rates),
        "wrong_choice_rate": mean(item.outcome == "wrong" for item in records),
        "no_choice_rate": mean(item.outcome == "no_choice" for item in records),
        "average_success_steps": (
            mean(item.steps for item in successes) if successes else 0.0
        ),
    }


def _write_records(path: Path, record_type: type, records: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(record_type.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_report(
    path: Path,
    condition: str,
    evolution_config: EvolutionConfig,
    training_records: list[TargetEvaluationRecord],
    holdout_records: list[TargetEvaluationRecord],
    selected_validation: TargetValidationRecord,
    final_validation: TargetValidationRecord,
    selected_stats: GenerationStats,
    validation_count: int,
    viewer_path: Path | None,
    validation_viewer_path: Path | None,
) -> None:
    training = _summarize_records(training_records)
    holdout = _summarize_records(holdout_records)
    passed = (
        holdout["success_rate"] >= TARGET_HOLDOUT_SUCCESS
        and holdout["cue_gap"] <= TARGET_CUE_GAP
        and holdout["worst_category_success_rate"] >= TARGET_WORST_CATEGORY
    )
    lines = [
        "# Выбор цели по исчезающей подсказке",
        "",
        f"**Условие: {condition}**",
        "",
        f"**Предварительный вердикт: {'PASS' if passed else 'FAIL'}**",
        "",
        "Две цели остаются видимыми весь эпизод. Cue A или Cue B присутствует",
        "только в начале, когда движение заблокировано. После скрытия cue мозг",
        "должен сохранить правило выбора во внутреннем состоянии.",
        "",
        "| Split | Episodes | Score | Success | Cue A | Cue B | Cue gap | Alignment | Contrast | Worst category | Wrong | No choice | Avg success steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _summary_row("training", len(training_records), training),
        (
            f"| validation | {validation_count} | "
            f"{selected_validation.average_score:.3f} | "
            f"{selected_validation.overall_success_rate:.2%} | "
            f"{selected_validation.cue_a_success_rate:.2%} | "
            f"{selected_validation.cue_b_success_rate:.2%} | "
            f"{selected_validation.cue_gap:.2%} | "
            f"{selected_validation.pair_alignment:+.2%} | "
            f"{selected_validation.pair_contrast_rate:.2%} | "
            f"{selected_validation.worst_category_success_rate:.2%} | "
            f"{selected_validation.wrong_choice_rate:.2%} | "
            f"{selected_validation.no_choice_rate:.2%} | "
            f"{selected_validation.average_success_steps:.2f} |"
        ),
        _summary_row("holdout", len(holdout_records), holdout),
        "",
        (
            f"- Selected generation: {selected_stats.generation + 1}/"
            f"{evolution_config.generations} (index {selected_stats.generation})"
        ),
        (
            f"- Generalization gap: "
            f"{training['success_rate'] - holdout['success_rate']:+.2%}"
        ),
        f"- Holdout success target: >= {TARGET_HOLDOUT_SUCCESS:.0%}",
        f"- Holdout cue-gap target: <= {TARGET_CUE_GAP:.0%}",
        (
            "- Holdout worst-category target: >= "
            f"{TARGET_WORST_CATEGORY:.0%}"
        ),
        "- Holdout did not participate in checkpoint selection.",
        "",
        "## Validation checkpoint",
        "",
        "| Generation | Overall | Cue A | Cue B | Gap | Alignment | Contrast | Worst | Wrong | No choice | Avg success steps |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _validation_row(selected_validation, "selected"),
        _validation_row(final_validation, "final"),
        "",
        "Высокий результат только для Cue A или только для Cue B означает",
        "фиксированную предпочтительную цель, а не использование правила.",
        "Stateless-контроль особенно важен: после скрытия cue его теоретический",
        "ориентир составляет около 50% на сбалансированном наборе.",
        "Alignment показывает направление незавершённых траекторий: 0% означает",
        "одинаковое поведение в паре, а положительное значение означает правильное",
        "расхождение под действием Cue A и Cue B. Contrast считает только пары,",
        "в которых агент уже дошёл до разных целей.",
        "",
    ]
    if viewer_path is not None:
        lines.extend([f"[Открыть holdout viewer]({viewer_path.name})", ""])
    if validation_viewer_path is not None:
        lines.extend(
            [f"[Открыть validation viewer]({validation_viewer_path.name})", ""]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_row(label: str, episodes: int, summary: dict[str, float]) -> str:
    return (
        f"| {label} | {episodes} | {summary['average_score']:.3f} | "
        f"{summary['success_rate']:.2%} | "
        f"{summary['cue_a_success_rate']:.2%} | "
        f"{summary['cue_b_success_rate']:.2%} | "
        f"{summary['cue_gap']:.2%} | "
        f"{summary['pair_alignment']:+.2%} | "
        f"{summary['pair_contrast_rate']:.2%} | "
        f"{summary['worst_category_success_rate']:.2%} | "
        f"{summary['wrong_choice_rate']:.2%} | "
        f"{summary['no_choice_rate']:.2%} | "
        f"{summary['average_success_steps']:.2f} |"
    )


def _validation_row(record: TargetValidationRecord, label: str) -> str:
    return (
        f"| {record.generation + 1} ({label}) | "
        f"{record.overall_success_rate:.2%} | "
        f"{record.cue_a_success_rate:.2%} | "
        f"{record.cue_b_success_rate:.2%} | "
        f"{record.cue_gap:.2%} | "
        f"{record.pair_alignment:+.2%} | "
        f"{record.pair_contrast_rate:.2%} | "
        f"{record.worst_category_success_rate:.2%} | "
        f"{record.wrong_choice_rate:.2%} | "
        f"{record.no_choice_rate:.2%} | "
        f"{record.average_success_steps:.2f} |"
    )


def _start_live_viewer(
    args: argparse.Namespace,
    evolution_config: EvolutionConfig,
    world_config: TargetChoiceWorldConfig,
    condition: str,
) -> LiveRunViewer | None:
    if not args.open_viewer:
        return None
    try:
        viewer = LiveRunViewer(
            evolution_config=evolution_config,
            world_config=world_config,
            sample_limit=max(args.live_samples, 4),
            experiment="choose_target",
            label=f"target rule | {condition}",
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
