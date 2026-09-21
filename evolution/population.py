from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
import random
from statistics import mean
from typing import Callable

from brain import BrainGenome
from world import EpisodeResult, FoodWorld, FoodWorldConfig


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 80
    generations: int = 40
    episodes_per_genome: int = 5
    hidden_size: int = 24
    min_hidden_size: int = 8
    max_hidden_size: int = 80
    elite_fraction: float = 0.15
    mutation_rate: float = 0.08
    mutation_power: float = 0.25
    structural_mutation_rate: float = 0.08
    seed: int = 7
    workers: int = 1


@dataclass
class GenerationStats:
    generation: int
    best_score: float
    average_score: float
    best_eat_rate: float
    average_hidden_size: float


@dataclass(frozen=True)
class RankedCandidate:
    """One of the highest-scoring brains in a completed generation."""

    rank: int
    score: float
    eat_rate: float
    genome: BrainGenome


@dataclass
class EvaluationProgress:
    """A completed candidate evaluation, emitted in the parent process."""

    generation: int
    completed: int
    total: int
    candidate_index: int
    worker_pid: int
    score: float
    eat_rate: float
    genome: BrainGenome
    replay: EpisodeResult | None


@dataclass
class _EvaluationResult:
    index: int
    worker_pid: int
    score: float
    eat_rate: float
    genome: BrainGenome
    replay: EpisodeResult | None


_EvaluationJob = tuple[
    int,
    BrainGenome,
    FoodWorldConfig,
    tuple[int, ...],
    int | None,
    type[FoodWorld],
]


def run_evolution(
    config: EvolutionConfig,
    world_config: FoodWorldConfig | None = None,
    on_generation: Callable[[GenerationStats], None] | None = None,
    on_generation_best: Callable[[GenerationStats, BrainGenome], None] | None = None,
    on_generation_ranked: (
        Callable[[GenerationStats, list[RankedCandidate]], None] | None
    ) = None,
    on_evaluation: Callable[[EvaluationProgress], None] | None = None,
    trace_sample_size: int = 0,
    ranked_candidate_count: int = 0,
    world_class: type[FoodWorld] = FoodWorld,
    trace_episode_indices: tuple[int, ...] | None = None,
    episode_seeds: tuple[int, ...] | None = None,
    episode_seed_sets: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[BrainGenome, list[GenerationStats]]:
    if trace_sample_size < 0:
        raise ValueError("trace_sample_size must be zero or positive.")
    if ranked_candidate_count < 0:
        raise ValueError("ranked_candidate_count must be zero or positive.")
    if episode_seeds is not None and episode_seed_sets is not None:
        raise ValueError("Use either episode_seeds or episode_seed_sets, not both.")
    if episode_seed_sets is not None:
        if not episode_seed_sets:
            raise ValueError("episode_seed_sets cannot be empty.")
        resolved_episode_seed_sets = tuple(
            tuple(seed_set) for seed_set in episode_seed_sets
        )
    else:
        resolved_episode_seed_sets = (
            (
                tuple(episode_seeds)
                if episode_seeds is not None
                else training_episode_seeds(config)
            ),
        )
    for seed_set in resolved_episode_seed_sets:
        if len(seed_set) != config.episodes_per_genome:
            raise ValueError(
                "Each episode seed set must contain exactly "
                "episodes_per_genome seeds."
            )
        if len(set(seed_set)) != len(seed_set):
            raise ValueError("Episode seeds must be distinct within each set.")
    if trace_episode_indices is not None:
        if not trace_episode_indices:
            raise ValueError("trace_episode_indices cannot be empty.")
        if any(
            index < 0 or index >= config.episodes_per_genome
            for index in trace_episode_indices
        ):
            raise ValueError("trace episode index is outside the training episodes.")

    rng = random.Random(config.seed)
    world = world_class(world_config)
    population = [
        BrainGenome.random(
            rng,
            input_size=world.input_size,
            hidden_size=config.hidden_size,
            output_size=world.output_size,
        )
        for _ in range(config.population_size)
    ]

    history: list[GenerationStats] = []
    best_genome = population[0].clone()
    best_seen_score = float("-inf")

    for generation in range(config.generations):
        generation_episode_seeds = resolved_episode_seed_sets[
            generation % len(resolved_episode_seed_sets)
        ]
        scored = _evaluate_population(
            population,
            config,
            world.config,
            generation,
            on_evaluation=on_evaluation,
            trace_sample_size=trace_sample_size,
            trace_episode_indices=trace_episode_indices,
            world_class=world_class,
            episode_seeds=generation_episode_seeds,
        )
        scored.sort(key=lambda item: item[0], reverse=True)

        best_score, best_eat_rate, generation_best = scored[0]
        if best_score > best_seen_score:
            best_seen_score = best_score
            best_genome = generation_best.clone()

        stats = GenerationStats(
            generation=generation,
            best_score=best_score,
            average_score=mean(score for score, _, _ in scored),
            best_eat_rate=best_eat_rate,
            average_hidden_size=mean(genome.hidden_size for _, _, genome in scored),
        )
        history.append(stats)
        if on_generation is not None:
            on_generation(stats)
        if on_generation_best is not None:
            on_generation_best(stats, generation_best.clone())
        if on_generation_ranked is not None and ranked_candidate_count > 0:
            leaders = [
                RankedCandidate(
                    rank=rank,
                    score=score,
                    eat_rate=eat_rate,
                    genome=genome.clone(),
                )
                for rank, (score, eat_rate, genome) in enumerate(
                    scored[:ranked_candidate_count],
                    start=1,
                )
            ]
            on_generation_ranked(stats, leaders)

        if generation < config.generations - 1:
            population = _next_generation(scored, config, rng)

    return best_genome.clone(), history


def training_episode_seeds(config: EvolutionConfig) -> tuple[int, ...]:
    """Return the fixed scenarios shared by every candidate in a run."""
    return tuple(
        config.seed + episode * 10_007
        for episode in range(config.episodes_per_genome)
    )


def _evaluate_population(
    population: list[BrainGenome],
    config: EvolutionConfig,
    world_config: FoodWorldConfig,
    generation: int,
    on_evaluation: Callable[[EvaluationProgress], None] | None,
    trace_sample_size: int,
    trace_episode_indices: tuple[int, ...] | None,
    world_class: type[FoodWorld],
    episode_seeds: tuple[int, ...],
) -> list[tuple[float, float, BrainGenome]]:
    jobs = [
        (
            index,
            genome,
            world_config,
            episode_seeds,
            (
                trace_episode_indices[index % len(trace_episode_indices)]
                if index < trace_sample_size and trace_episode_indices is not None
                else 0 if index < trace_sample_size else None
            ),
            world_class,
        )
        for index, genome in enumerate(population)
    ]

    completed = 0
    results: list[_EvaluationResult | None] = [None for _ in jobs]

    if config.workers <= 1:
        for job in jobs:
            result = _evaluate_one(job)
            results[result.index] = result
            completed += 1
            _notify_evaluation(on_evaluation, generation, completed, len(jobs), result)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = [pool.submit(_evaluate_one, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results[result.index] = result
                completed += 1
                _notify_evaluation(on_evaluation, generation, completed, len(jobs), result)

    return [
        (result.score, result.eat_rate, result.genome)
        for result in results
        if result is not None
    ]


def _notify_evaluation(
    callback: Callable[[EvaluationProgress], None] | None,
    generation: int,
    completed: int,
    total: int,
    result: _EvaluationResult,
) -> None:
    if callback is None:
        return
    callback(
        EvaluationProgress(
            generation=generation,
            completed=completed,
            total=total,
            candidate_index=result.index,
            worker_pid=result.worker_pid,
            score=result.score,
            eat_rate=result.eat_rate,
            genome=result.genome,
            replay=result.replay,
        )
    )


def _evaluate_one(job: _EvaluationJob) -> _EvaluationResult:
    index, genome, world_config, episode_seeds, trace_episode_index, world_class = job
    world = world_class(world_config)
    results = [
        world.evaluate(
            genome,
            random.Random(episode_seed),
            record_trace=episode_index == trace_episode_index,
        )
        for episode_index, episode_seed in enumerate(episode_seeds)
    ]
    score = world.aggregate_fitness(results)
    eat_rate = sum(1 for result in results if result.ate_food) / len(episode_seeds)
    return _EvaluationResult(
        index=index,
        worker_pid=os.getpid(),
        score=score,
        eat_rate=eat_rate,
        genome=genome,
        replay=results[trace_episode_index] if trace_episode_index is not None else None,
    )


def _next_generation(
    scored: list[tuple[float, float, BrainGenome]],
    config: EvolutionConfig,
    rng: random.Random,
) -> list[BrainGenome]:
    elite_count = max(1, int(config.population_size * config.elite_fraction))
    elites = [genome for _, _, genome in scored[:elite_count]]

    next_population = [genome.clone() for genome in elites]
    while len(next_population) < config.population_size:
        parent = rng.choice(elites)
        child = parent.mutated(
            rng,
            weight_mutation_rate=config.mutation_rate,
            weight_mutation_power=config.mutation_power,
            structural_mutation_rate=config.structural_mutation_rate,
            min_hidden_size=config.min_hidden_size,
            max_hidden_size=config.max_hidden_size,
        )
        next_population.append(child)

    return next_population
