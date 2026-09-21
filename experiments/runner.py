from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import random
import webbrowser

from evolution import EvolutionConfig, GenerationStats, RankedCandidate, run_evolution
from visualization import LiveRunViewer, RankedReplay, write_visualization
from world import FoodWorld, FoodWorldConfig


def add_evolution_arguments(
    parser: argparse.ArgumentParser,
    output: Path,
    include_visualization: bool = True,
) -> None:
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--min-hidden", type=int, default=8)
    parser.add_argument("--max-hidden", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--elite-fraction", type=float, default=0.15)
    parser.add_argument("--mutation-rate", type=float, default=0.08)
    parser.add_argument("--mutation-power", type=float, default=0.25)
    parser.add_argument("--structural-mutation-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=output)
    if include_visualization:
        parser.add_argument("--visual-seed", type=int, default=1007)
        parser.add_argument(
            "--live-samples",
            type=int,
            default=4,
            help=(
                "Number of candidate traces shown during evaluation and ranked "
                "leaders shown after each generation."
            ),
        )
        parser.add_argument("--no-visualization", action="store_true")
        parser.add_argument("--open-viewer", action="store_true")


def validate_evolution_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "live_samples", 1) < 1:
        parser.error("--live-samples must be at least 1")
    if args.population < 2:
        parser.error("--population must be at least 2")
    if args.generations < 1:
        parser.error("--generations must be at least 1")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")


def evolution_config_from_args(args: argparse.Namespace) -> EvolutionConfig:
    return EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        episodes_per_genome=args.episodes,
        hidden_size=args.hidden,
        min_hidden_size=args.min_hidden,
        max_hidden_size=args.max_hidden,
        elite_fraction=args.elite_fraction,
        mutation_rate=args.mutation_rate,
        mutation_power=args.mutation_power,
        structural_mutation_rate=args.structural_mutation_rate,
        seed=args.seed,
        workers=args.workers,
    )


def run_experiment(
    args: argparse.Namespace,
    *,
    experiment_name: str,
    world_class: type[FoodWorld],
    world_config: FoodWorldConfig,
    next_question: str,
) -> Path:
    evolution_config = evolution_config_from_args(args)
    run_dir = _create_run_dir(args.output)
    _write_json(
        run_dir / "config.json",
        {
            "experiment": experiment_name,
            "evolution": asdict(evolution_config),
            "world": asdict(world_config),
        },
    )

    print(f"Run: {run_dir}")
    live_viewer: LiveRunViewer | None = None
    if args.open_viewer:
        try:
            live_viewer = LiveRunViewer(
                evolution_config=evolution_config,
                world_config=world_config,
                sample_limit=args.live_samples,
                experiment=experiment_name,
            )
            live_url = live_viewer.start()
            print(f"Live visualization: {live_url}")
            _open_in_browser(live_url)
        except OSError as error:
            print(f"Could not start live visualization: {error}", flush=True)

    print("generation,best_score,avg_score,best_eat_rate,avg_hidden", flush=True)

    def on_generation(stats: GenerationStats) -> None:
        print(_format_stats(stats), flush=True)

    final_ranked_replays: list[RankedReplay] = []

    def on_generation_ranked(
        stats: GenerationStats,
        candidates: list[RankedCandidate],
    ) -> None:
        if live_viewer is None:
            return
        world = world_class(world_config)
        replays = [
            RankedReplay(
                rank=candidate.rank,
                score=candidate.score,
                eat_rate=candidate.eat_rate,
                episode=world.evaluate(
                    candidate.genome,
                    random.Random(args.visual_seed),
                    record_trace=True,
                ),
            )
            for candidate in candidates
        ]
        final_ranked_replays.clear()
        final_ranked_replays.extend(replays)
        live_viewer.record_generation_leaders(stats, replays)

    best_genome, history = run_evolution(
        evolution_config,
        world_config,
        on_generation=on_generation,
        on_generation_ranked=on_generation_ranked if live_viewer else None,
        on_evaluation=live_viewer.record_evaluation if live_viewer else None,
        trace_sample_size=args.live_samples if live_viewer else 0,
        ranked_candidate_count=args.live_samples if live_viewer else 0,
        world_class=world_class,
    )

    _write_metrics(run_dir / "metrics.csv", history)
    best_genome.save_json(run_dir / "best_genome.json")
    _write_notes(
        run_dir / "notes.md",
        experiment_name,
        evolution_config,
        world_config,
        history,
        next_question,
    )
    viewer_path = None
    if not args.no_visualization:
        viewer_path = write_visualization(
            run_dir=run_dir,
            best_genome=best_genome,
            history=history,
            evolution_config=evolution_config,
            world_config=world_config,
            seed=args.visual_seed,
            world_class=world_class,
            experiment=experiment_name,
        )

    final = history[-1]
    best_observed = max(history, key=lambda stats: stats.best_score)
    if live_viewer is not None:
        if final_ranked_replays:
            best_episode = final_ranked_replays[0].episode
        else:
            best_episode = world_class(world_config).evaluate(
                best_genome,
                random.Random(args.visual_seed),
                record_trace=True,
            )
        live_viewer.finish(
            final,
            best_episode,
            viewer_path,
            ranked_replays=final_ranked_replays or None,
        )
        live_viewer.wait_until_final_served()

    print()
    print(f"Final generation best score: {final.best_score:.3f}")
    print(f"Best observed score: {best_observed.best_score:.3f}")
    print(f"Best observed generation: {best_observed.generation}")
    print(f"Best observed eat rate: {best_observed.best_eat_rate:.2%}")
    print(f"Saved results to: {run_dir}")
    if viewer_path is not None:
        print(f"Open visualization: {viewer_path}")
        if args.open_viewer and live_viewer is None:
            _open_in_browser(viewer_path.resolve().as_uri())

    return run_dir


def _create_run_dir(root: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_metrics(path: Path, history: list[GenerationStats]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "generation",
                "best_score",
                "average_score",
                "best_eat_rate",
                "average_hidden_size",
            ],
        )
        writer.writeheader()
        for stats in history:
            writer.writerow(asdict(stats))


def _write_notes(
    path: Path,
    experiment_name: str,
    evolution_config: EvolutionConfig,
    world_config: FoodWorldConfig,
    history: list[GenerationStats],
    next_question: str,
) -> None:
    first = history[0]
    last = history[-1]
    best = max(history, key=lambda stats: stats.best_score)
    world_settings = asdict(world_config)
    lines = [
        f"# {experiment_name}",
        "",
        "## Setup",
        "",
        f"- Population: {evolution_config.population_size}",
        f"- Generations: {evolution_config.generations}",
        f"- Episodes per genome: {evolution_config.episodes_per_genome}",
        f"- Initial hidden neurons: {evolution_config.hidden_size}",
        f"- Workers: {evolution_config.workers}",
        "",
        "## World",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in world_settings.items())
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- First generation best score: {first.best_score:.3f}",
            f"- Final generation best score: {last.best_score:.3f}",
            f"- Best observed score: {best.best_score:.3f}",
            f"- Best observed generation: {best.generation}",
            f"- Best observed eat rate: {best.best_eat_rate:.2%}",
            f"- Final average hidden neurons: {last.average_hidden_size:.2f}",
            "",
            "## Next question",
            "",
            next_question,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_stats(stats: GenerationStats) -> str:
    return (
        f"{stats.generation},"
        f"{stats.best_score:.3f},"
        f"{stats.average_score:.3f},"
        f"{stats.best_eat_rate:.2f},"
        f"{stats.average_hidden_size:.2f}"
    )


def _open_in_browser(url: str) -> None:
    if webbrowser.open(url, new=2):
        return
    if hasattr(os, "startfile"):
        os.startfile(url)  # type: ignore[attr-defined]
