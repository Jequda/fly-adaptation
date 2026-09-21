from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import random
from statistics import mean
from typing import Callable
import webbrowser

from brain import BrainGenome
from evolution import (
    EvaluationProgress,
    EvolutionConfig,
    GenerationStats,
    RankedCandidate,
    run_evolution,
)
from experiments.runner import (
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
from world import EpisodeResult, MemoryFoodWorld, MemoryFoodWorldConfig


@dataclass(frozen=True)
class ComparisonRecord:
    condition: str
    replicate: int
    seed: int
    training_best_score: float
    training_best_eat_rate: float
    holdout_average_score: float
    holdout_eat_rate: float


@dataclass(frozen=True)
class _ComparisonFinalist:
    record: ComparisonRecord
    stats: GenerationStats
    history: list[GenerationStats]
    episode: EpisodeResult
    viewer_path: Path | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare recurrent and stateless brains on the hidden-food task.",
    )
    add_evolution_arguments(
        parser,
        output=Path("results/compare_memory"),
    )
    parser.set_defaults(population=60, generations=30, episodes=10, max_steps=60)
    parser.add_argument("--visible-steps", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--holdout-episodes", type=int, default=30)
    args = parser.parse_args()
    validate_evolution_arguments(parser, args)
    if args.visible_steps < 1 or args.visible_steps >= args.max_steps:
        parser.error("--visible-steps must be between 1 and --max-steps - 1")
    if args.replicates < 1:
        parser.error("--replicates must be at least 1")
    if args.holdout_episodes < 1:
        parser.error("--holdout-episodes must be at least 1")

    run_dir = _create_run_dir(args.output)
    base_config = evolution_config_from_args(args)
    _write_json(
        run_dir / "config.json",
        {
            "experiment": "compare_memory",
            "evolution": asdict(base_config),
            "max_steps": args.max_steps,
            "visible_steps": args.visible_steps,
            "training_scenarios": "shared_fixed_for_run",
            "replicates": args.replicates,
            "holdout_episodes": args.holdout_episodes,
        },
    )

    print(f"Run: {run_dir}")
    live_viewer = _start_live_viewer(args, base_config)
    print(
        "condition,replicate,generation,best_score,average_score,best_eat_rate,average_hidden",
        flush=True,
    )
    print(
        "result,condition,replicate,training_best,training_eat_rate,holdout_score,holdout_eat_rate",
        flush=True,
    )
    records: list[ComparisonRecord] = []
    viewer_paths: dict[tuple[str, int], Path] = {}
    finalists: list[_ComparisonFinalist] = []
    for replicate in range(args.replicates):
        seed = args.seed + replicate * 100_000
        for condition in ("recurrent", "stateless"):
            world_config = MemoryFoodWorldConfig(
                max_steps=args.max_steps,
                visible_steps=args.visible_steps,
                use_recurrence=condition == "recurrent",
            )
            label = f"{condition}, repeat {replicate + 1}/{args.replicates}"
            if live_viewer is not None:
                live_viewer.begin_run(
                    replace(base_config, seed=seed),
                    world_config,
                    experiment="remember_food",
                    label=label,
                )

            def on_generation(stats: GenerationStats) -> None:
                print(_format_generation(condition, replicate, stats), flush=True)

            def on_generation_ranked(
                stats: GenerationStats,
                candidates: list[RankedCandidate],
            ) -> None:
                if live_viewer is None:
                    return
                world = MemoryFoodWorld(world_config)
                replays = [
                    RankedReplay(
                        rank=candidate.rank,
                        score=candidate.score,
                        eat_rate=candidate.eat_rate,
                        episode=world.evaluate(
                            candidate.genome,
                            random.Random(args.visual_seed + replicate),
                            record_trace=True,
                        ),
                    )
                    for candidate in candidates
                ]
                live_viewer.record_generation_leaders(stats, replays)

            record, best_genome, history, holdout_replays = _run_condition(
                condition=condition,
                replicate=replicate,
                seed=seed,
                evolution_config=replace(base_config, seed=seed),
                world_config=world_config,
                holdout_episodes=args.holdout_episodes,
                on_generation=on_generation,
                on_generation_ranked=(
                    on_generation_ranked if live_viewer else None
                ),
                on_evaluation=(live_viewer.record_evaluation if live_viewer else None),
                trace_sample_size=args.live_samples if live_viewer else 0,
                ranked_candidate_count=args.live_samples if live_viewer else 0,
            )
            records.append(record)
            best_genome.save_json(run_dir / f"{condition}-replicate-{replicate}-best_genome.json")
            viewer_path = None
            if not args.no_visualization:
                viewer_dir = run_dir / "visualizations" / f"{condition}-replicate-{replicate}"
                viewer_dir.mkdir(parents=True, exist_ok=True)
                viewer_path = write_visualization(
                    run_dir=viewer_dir,
                    best_genome=best_genome,
                    history=history,
                    evolution_config=replace(base_config, seed=seed),
                    world_config=world_config,
                    seed=args.visual_seed + replicate,
                    world_class=MemoryFoodWorld,
                    experiment="remember_food",
                    replay_episodes=holdout_replays,
                )
                viewer_paths[(condition, replicate)] = viewer_path

            best_stats = max(history, key=lambda stats: stats.best_score)
            finalists.append(
                _ComparisonFinalist(
                    record=record,
                    stats=best_stats,
                    history=history,
                    episode=holdout_replays[0].episode,
                    viewer_path=viewer_path,
                )
            )
            print(
                f"result,{record.condition},{record.replicate},"
                f"{record.training_best_score:.3f},{record.training_best_eat_rate:.2f},"
                f"{record.holdout_average_score:.3f},{record.holdout_eat_rate:.2f}",
                flush=True,
            )

    ranked_finalists, ranked_replays = _rank_finalists(
        finalists,
        args.live_samples,
    )
    winner = ranked_finalists[0]
    comparison_viewer_path = None
    if not args.no_visualization:
        summary_world_config = MemoryFoodWorldConfig(
            max_steps=args.max_steps,
            visible_steps=args.visible_steps,
            use_recurrence=winner.record.condition == "recurrent",
        )
        comparison_viewer_path = write_ranked_visualization(
            run_dir=run_dir,
            ranked_replays=ranked_replays,
            history=winner.history,
            evolution_config=replace(base_config, seed=winner.record.seed),
            world_config=summary_world_config,
        )

    _write_records(run_dir / "comparison.csv", records)
    _write_report(
        run_dir / "report.md",
        records,
        viewer_paths,
        comparison_viewer_path,
    )
    if live_viewer is not None:
        live_viewer.finish(
            winner.stats,
            winner.episode,
            winner.viewer_path,
            ranked_replays=ranked_replays,
            final_label="comparison summary",
            message=(
                f"Comparison complete: top {len(ranked_replays)} runs ranked by "
                "holdout score"
            ),
        )
        live_viewer.wait_until_final_served()
    print(f"Saved comparison to: {run_dir}")
    if comparison_viewer_path is not None:
        print(f"Open comparison viewer: {comparison_viewer_path}")


def _rank_finalists(
    finalists: list[_ComparisonFinalist],
    limit: int,
) -> tuple[list[_ComparisonFinalist], list[RankedReplay]]:
    ranked_finalists = sorted(
        finalists,
        key=lambda finalist: (
            finalist.record.holdout_average_score,
            finalist.record.holdout_eat_rate,
        ),
        reverse=True,
    )[:limit]
    ranked_replays = [
        RankedReplay(
            rank=rank,
            score=finalist.record.holdout_average_score,
            eat_rate=finalist.record.holdout_eat_rate,
            episode=finalist.episode,
            label=(
                f"{finalist.record.condition}, "
                f"repeat {finalist.record.replicate + 1}"
            ),
            metric="holdout",
        )
        for rank, finalist in enumerate(ranked_finalists, start=1)
    ]
    return ranked_finalists, ranked_replays


def _run_condition(
    condition: str,
    replicate: int,
    seed: int,
    evolution_config: EvolutionConfig,
    world_config: MemoryFoodWorldConfig,
    holdout_episodes: int,
    on_generation: Callable[[GenerationStats], None] | None = None,
    on_generation_best: Callable[[GenerationStats, BrainGenome], None] | None = None,
    on_generation_ranked: (
        Callable[[GenerationStats, list[RankedCandidate]], None] | None
    ) = None,
    on_evaluation: Callable[[EvaluationProgress], None] | None = None,
    trace_sample_size: int = 0,
    ranked_candidate_count: int = 0,
) -> tuple[
    ComparisonRecord,
    BrainGenome,
    list[GenerationStats],
    list[ReplayEpisode],
]:
    best_genome, history = run_evolution(
        evolution_config,
        world_config,
        on_generation=on_generation,
        on_generation_best=on_generation_best,
        on_generation_ranked=on_generation_ranked,
        on_evaluation=on_evaluation,
        trace_sample_size=trace_sample_size,
        ranked_candidate_count=ranked_candidate_count,
        world_class=MemoryFoodWorld,
    )
    best_stats = max(history, key=lambda stats: stats.best_score)
    world = MemoryFoodWorld(world_config)
    holdout_replays = []
    for episode_index in range(holdout_episodes):
        episode_seed = seed + 50_000 + episode_index
        episode = world.evaluate(
            best_genome,
            random.Random(episode_seed),
            record_trace=True,
        )
        holdout_replays.append(
            ReplayEpisode(
                index=episode_index,
                seed=episode_seed,
                episode=episode,
            )
        )
    holdout_results = [replay.episode for replay in holdout_replays]
    record = ComparisonRecord(
        condition=condition,
        replicate=replicate,
        seed=seed,
        training_best_score=best_stats.best_score,
        training_best_eat_rate=best_stats.best_eat_rate,
        holdout_average_score=mean(result.score for result in holdout_results),
        holdout_eat_rate=(
            sum(result.ate_food for result in holdout_results) / len(holdout_results)
        ),
    )
    return record, best_genome, history, holdout_replays


def _start_live_viewer(
    args: argparse.Namespace,
    evolution_config: EvolutionConfig,
) -> LiveRunViewer | None:
    if not args.open_viewer:
        return None

    initial_world_config = MemoryFoodWorldConfig(
        max_steps=args.max_steps,
        visible_steps=args.visible_steps,
        use_recurrence=True,
    )
    try:
        viewer = LiveRunViewer(
            evolution_config=evolution_config,
            world_config=initial_world_config,
            sample_limit=args.live_samples,
            experiment="remember_food",
            label=f"recurrent, repeat 1/{args.replicates}",
        )
        live_url = viewer.start()
        print(f"Live visualization: {live_url}", flush=True)
        webbrowser.open(live_url, new=2)
        return viewer
    except OSError as error:
        print(f"Could not start live visualization: {error}", flush=True)
        return None


def _format_generation(
    condition: str,
    replicate: int,
    stats: GenerationStats,
) -> str:
    return (
        f"{condition},{replicate},{stats.generation},"
        f"{stats.best_score:.3f},{stats.average_score:.3f},"
        f"{stats.best_eat_rate:.2f},{stats.average_hidden_size:.2f}"
    )


def _create_run_dir(root: Path) -> Path:
    run_dir = root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_records(path: Path, records: list[ComparisonRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(ComparisonRecord.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_report(
    path: Path,
    records: list[ComparisonRecord],
    viewer_paths: dict[tuple[str, int], Path],
    comparison_viewer_path: Path | None = None,
) -> None:
    grouped = {
        condition: [record for record in records if record.condition == condition]
        for condition in {record.condition for record in records}
    }
    lines = [
        "# Сравнение памяти",
        "",
        "Holdout-эпизоды используют новые размещения еды, которых не было при отборе.",
        "Все кандидаты и поколения проходят один фиксированный набор тренировочных сцен.",
        "",
        "| Условие | Holdout score | Holdout eat rate |",
        "| --- | ---: | ---: |",
    ]
    for condition in ("recurrent", "stateless"):
        condition_records = grouped[condition]
        score = mean(record.holdout_average_score for record in condition_records)
        eat_rate = mean(record.holdout_eat_rate for record in condition_records)
        lines.append(f"| {condition} | {score:.3f} | {eat_rate:.2%} |")
    lines.extend(
        [
            "",
            "Более высокий holdout-результат recurrent-варианта означает, что внутреннее состояние помогает после исчезновения подсказки.",
            "",
        ]
    )
    if comparison_viewer_path is not None:
        relative_path = comparison_viewer_path.relative_to(path.parent).as_posix()
        lines.extend(
            [
                "## Итоговый viewer",
                "",
                f"[Четыре лучших запуска]({relative_path})",
                "",
            ]
        )
    if viewer_paths:
        lines.extend(["## Viewers", ""])
        for (condition, replicate), viewer_path in sorted(viewer_paths.items()):
            relative_path = viewer_path.relative_to(path.parent).as_posix()
            lines.append(
                f"- {condition}, repeat {replicate + 1}: "
                f"[{relative_path}]({relative_path})"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
