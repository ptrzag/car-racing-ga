# ga.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import os
import time
import json
import argparse
import numpy as np
import multiprocessing as mp

from agent import Car
from eval_utils import evaluate_car
import config

def save_generation(gen_idx: int, population: list[Car], folder: str):
    data = [c.to_json() for c in population]
    with open(os.path.join(folder, f"gen_{gen_idx}.json"), "w") as f:
        json.dump(data, f, indent=2)

def run_ga(
    pop_size: int = config.POP_SIZE,
    num_generations: int = config.GENERATIONS,
    p_crossover: float = config.P_CROSSOVER,
    p_mutation: float = config.P_MUTATION,
    sigma: float = config.SIGMA,
    max_steps: int = config.MAX_STEPS,
    seed: int | None = None
) -> Car:
    np.random.seed(seed)

    feat = 17 if config.USE_EXTENDED_DISTANCES else 9
    weight_len = feat * 3
    population = [Car(np.random.randn(weight_len)) for _ in range(pop_size)]

    timestamp = time.strftime("%Y%m%d_%H%M")
    out_folder = os.path.join("data", "generations", timestamp)
    os.makedirs(out_folder, exist_ok=True)

    fitness_history: list[dict] = []

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

        fitness_history.append({
            "generation": gen,
            "max": max_f,
            "mean": mean_f,
            "min": min_f
        })

        save_generation(gen, population, out_folder)

        new_pop = [population[0]]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            if np.random.rand() < p_crossover:
                c1_w, c2_w = crossover(p1.weights, p2.weights)
            else:
                c1_w, c2_w = p1.weights.copy(), p2.weights.copy()
            if np.random.rand() < p_mutation:
                c1_w = mutate(c1_w, sigma)
            if np.random.rand() < p_mutation:
                c2_w = mutate(c2_w, sigma)
            new_pop.append(Car(c1_w))
            if len(new_pop) < pop_size:
                new_pop.append(Car(c2_w))
        population = new_pop

    best = population[0]
    best_data = best.to_json()
    best_data["seed"] = seed  # Save the seed for reproducibility
    with open(os.path.join(out_folder, "best.json"), "w") as f:
        json.dump(best.to_json(), f, indent=2)

    with open(os.path.join(out_folder, "fitness_history.json"), "w") as f:
        json.dump(fitness_history, f, indent=2)

    print(f"All outputs saved to: {out_folder}")
    return best

def crossover(a: np.ndarray, b: np.ndarray):
    pt = np.random.randint(1, len(a)-1)
    return (np.concatenate([a[:pt], b[pt:]]),
            np.concatenate([b[:pt], a[pt:]]))

def mutate(w: np.ndarray, sigma: float):
    return w + np.random.normal(0, sigma, size=w.shape)

def tournament_selection(pop: list[Car], k: int = 3) -> Car:
    candidates = np.random.choice(pop, k)
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
