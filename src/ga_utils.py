import os
import json
import time
import threading
import numpy as np
import multiprocessing as mp
from agent import Car
from eval_utils import evaluate_population
import config


def save_generation_async(gen_idx: int, pop_weights: list[np.ndarray], folder: str):
    """Asynchronous save of generation weights only (no Car pickling)."""
    data = [{"weights": w.tolist()} for w in pop_weights]
    path = os.path.join(folder, f"gen_{gen_idx}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_ga(
    pop_size: int = config.POP_SIZE,
    num_generations: int = config.GENERATIONS,
    p_crossover: float = config.P_CROSSOVER,
    p_mutation: float = config.P_MUTATION,
    sigma: float = config.SIGMA,
    max_steps: int = config.MAX_STEPS,
    seed: int|None = None,
    procs: int|None = None
) -> Car:
    np.random.seed(seed)
    weight_len = (17 if config.USE_EXTENDED_DISTANCES else 9) * 3
    population = [Car(np.random.randn(weight_len)) for _ in range(pop_size)]

    ts = time.strftime("%Y%m%d_%H%M")
    out_folder = os.path.join("data", "generations", ts)
    os.makedirs(out_folder, exist_ok=True)
    # save config metadata
    with open(os.path.join(out_folder, "meta.json"), "w") as f:
        json.dump({"timestamp": ts, "pop_size": pop_size, "generations": num_generations,
                   "p_crossover": p_crossover, "p_mutation": p_mutation,
                   "sigma": sigma, "max_steps": max_steps, "seed": seed}, f, indent=2)

    cpu = mp.cpu_count()
    pool_size = max(cpu - 2, 1) if procs is None else procs

    fitness_history = []
    for gen in range(num_generations):
        print(f"Generation {gen+1}/{num_generations}")
        # vectorized evaluation
        fitnesses = evaluate_population(population, max_steps=max_steps, seed=seed)
        # sort by fitness
        for car, f in zip(population, fitnesses): car.fitness = f
        population.sort(key=lambda c: c.fitness or -np.inf, reverse=True)
        scores = [c.fitness for c in population if np.isfinite(c.fitness)]
        max_f = float(np.max(scores)) if scores else -np.inf
        mean_f = float(np.mean(scores)) if scores else -np.inf
        min_f = float(np.min(scores)) if scores else -np.inf
        print(f"  max={max_f:.2f}, mean={mean_f:.2f}, min={min_f:.2f}")
        fitness_history.append({"generation": gen, "max": max_f, "mean": mean_f, "min": min_f})
        # async save only weights
        threading.Thread(target=save_generation_async, args=(gen, [c.weights for c in population], out_folder), daemon=True).start()
        # next gen
        new_pop = [population[0]]
        while len(new_pop) < pop_size:
            p1, p2 = np.random.choice(population, 2, replace=False)
            if np.random.rand() < p_crossover:
                pt = np.random.randint(1, weight_len-1)
                w1 = np.concatenate((p1.weights[:pt], p2.weights[pt:]))
                w2 = np.concatenate((p2.weights[:pt], p1.weights[pt:]))
            else:
                w1, w2 = p1.weights.copy(), p2.weights.copy()
            if np.random.rand() < p_mutation:
                w1 += np.random.normal(0, sigma, size=weight_len)
            if np.random.rand() < p_mutation:
                w2 += np.random.normal(0, sigma, size=weight_len)
            new_pop.extend([Car(w1), Car(w2)])
        population = new_pop[:pop_size]

    # save best and history synchronously
    best = population[0]
    with open(os.path.join(out_folder, "best.json"), "w") as f:
        json.dump(best.to_json(), f, indent=2)
    with open(os.path.join(out_folder, "fitness_history.json"), "w") as f:
        json.dump(fitness_history, f, indent=2)
    print(f"All outputs saved to: {out_folder}")
    return best