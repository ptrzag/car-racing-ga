# ga.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import os
import time
import json
import argparse
import numpy as np
import multiprocessing as mp
from typing import Optional

from agent import Car
from eval_utils import evaluate_car
import config


def save_generation(gen_idx: int, population: list[Car], folder: str):
    data = [c.to_json() for c in population]
    with open(os.path.join(folder, f"gen_{gen_idx}.json"), "w") as f:
        json.dump(data, f, indent=2)

def population_diversity(pop: list[Car]) -> float:
    # Average pairwise Euclidean distance between weights
    weights = np.array([c.weights for c in pop])
    n = len(weights)
    if n < 2:
        return 0.0
    dist_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            dist_sum += np.linalg.norm(weights[i] - weights[j])
            count += 1
    return dist_sum / count

def run_ga(
    pop_size: int = config.POP_SIZE,
    num_generations: int = config.GENERATIONS,
    p_crossover: float = config.P_CROSSOVER,
    p_mutation: float = config.P_MUTATION,
    sigma: float = config.SIGMA,
    max_steps: int = config.MAX_STEPS,
    seed: Optional[int] = None
) -> Car:
    if seed is None:
        seed = 71
    rng = np.random.default_rng(seed)

    feat = 17 if config.USE_EXTENDED_DISTANCES else 9
    weight_len = feat * 3
    population = [Car(rng.standard_normal(weight_len)) for _ in range(pop_size)]

    timestamp = time.strftime("%Y%m%d_%H%M")
    out_folder = os.path.join("data", "generations", timestamp)
    os.makedirs(out_folder, exist_ok=True)

    # Save global config metadata
    meta = {
        "timestamp": timestamp,
        "pop_size": pop_size,
        "generations": num_generations,
        "p_crossover": p_crossover,
        "p_mutation": p_mutation,
        "sigma": sigma,
        "max_steps": max_steps,
        "seed": seed
    }
    with open(os.path.join(out_folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    fitness_history: list[dict] = []

    # Set multiprocessing start method to spawn for safety (especially on Windows)
    mp.set_start_method('spawn', force=True)

    prev_elite_hash = None

    for gen in range(num_generations):
        print(f"Generation {gen+1}/{num_generations}")

        with mp.Pool() as pool:
            args = [(car, max_steps, seed, False) for car in population]
            fitnesses = pool.starmap(evaluate_car, args)
        for car, fit in zip(population, fitnesses):
            car.fitness = fit

        population.sort(key=lambda c: c.fitness, reverse=True)
        scores = [c.fitness for c in population]

        # filter out disqualified (-inf) values
        valid = [s for s in scores if np.isfinite(s)]
        if valid:
            max_f  = float(np.max(valid))
            mean_f = float(np.mean(valid))
            min_f  = float(np.min(valid))
        else:
            # everyone disqualified? fall back to -inf
            max_f = mean_f = min_f = float('-inf')

        print(f"  max={max_f:.2f}, mean={mean_f:.2f}, min={min_f:.2f}")

        diversity = population_diversity(population)
        print(f"  diversity={diversity:.4f}")
        unique_hashes = set(hash_agent(agent) for agent in population)
        print(f"Unique agents: {len(unique_hashes)} / {len(population)}")
        if diversity < 0.01:
            print("  Warning: population diversity is very low, may cause premature convergence")

        fitness_history.append({
            "generation": gen,
            "max": max_f,
            "mean": mean_f,
            "min": min_f
        })

        save_generation(gen, population, out_folder)
        new_pop = []
        
        elites = population[:config.ELITE_COUNT] 
        if elites:
            elites = population[:config.ELITE_COUNT]
            best_elite = population[0]  # assign here BEFORE the loop
            for elite in elites:
                new_pop.append(elite)
                mutated_weights = mutate(elite.weights.copy(), sigma * rng.uniform(1.0, 2.0), rng)
                new_pop.append(Car(mutated_weights))

            if best_elite:
                elite_hash = hash_agent(best_elite)
                if elite_hash != prev_elite_hash:
                    new_pop.append(best_elite)
                    prev_elite_hash = elite_hash
                else:
                    # mutate instead of copying same elite again
                    mutated_weights = mutate(best_elite.weights.copy(), sigma, rng)
                    new_pop.append(Car(mutated_weights))


        while len(new_pop) < pop_size:
            if rng.random() < 0.1:
                new_pop.append(Car(rng.standard_normal(weight_len)))
            else:
                p1 = tournament_selection(population, rng)
                p2 = tournament_selection(population, rng)
                if rng.random() < p_crossover:
                    c1_w, c2_w = crossover(p1.weights, p2.weights, rng)
                else:
                    c1_w, c2_w = p1.weights.copy(), p2.weights.copy()

                diversity_threshold = 1.0  # define a sensible threshold
                mutation_boost = 0.2 if diversity < diversity_threshold else 0.0

                if rng.random() < p_mutation + mutation_boost:
                    c1_w = mutate(c1_w, sigma, rng)
                if rng.random() < p_mutation + mutation_boost:
                    c2_w = mutate(c2_w, sigma, rng)

                new_pop.append(Car(c1_w))
                if len(new_pop) < pop_size:
                    new_pop.append(Car(c2_w))
        population = new_pop


    best = population[0]
    best_data = best.to_json()
    best_data["seed"] = seed  # Save the seed for reproducibility
    with open(os.path.join(out_folder, "best.json"), "w") as f:
        json.dump(best_data, f, indent=2)

    with open(os.path.join(out_folder, "fitness_history.json"), "w") as f:
        json.dump(fitness_history, f, indent=2)

    print(f"All outputs saved to: {out_folder}")
    return best

def crossover(parent1, parent2, rng):
    # e.g., single-point crossover
    point = rng.integers(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutate(w: np.ndarray, sigma: float, rng: np.random.Generator):
    noise = rng.uniform(-1, 1, size=w.shape)
    t_noise = np.arctan(np.pi * noise)  # range approx (-π/2, π/2)
    scaled = sigma * t_noise / (np.pi / 2)  # normalize to (-sigma, sigma)
    w_new = w + scaled
    return np.clip(w_new, -1, 1)

def tournament_selection(pop: list[Car], rng: np.random.Generator, k: int = 5) -> Car:
    candidates = rng.choice(pop, k, replace=False)
    return max(candidates, key=lambda c: c.fitness)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pop_size", type=int, default=config.POP_SIZE)
    p.add_argument("--generations", type=int, default=config.GENERATIONS)
    p.add_argument("--p_crossover", type=float, default=config.P_CROSSOVER)
    p.add_argument("--p_mutation", type=float, default=config.P_MUTATION)
    p.add_argument("--sigma", type=float, default=config.SIGMA)
    p.add_argument("--max_steps", type=int, default=config.MAX_STEPS)
    return p.parse_args()

def hash_agent(agent: Car):
    return tuple(np.round(w, 4).tobytes() for w in agent.weights)




if __name__ == "__main__":
    args = parse_args()
    run_ga(
        pop_size=args.pop_size,
        num_generations=args.generations,
        p_crossover=args.p_crossover,
        p_mutation=args.p_mutation,
        sigma=args.sigma,
        max_steps=args.max_steps
    )
