import json
import os
import random
import numpy as np

from agent import Car
from eval_utils import evaluate_car
from env_utils import make_env


def tournament_selection(population: list[Car], k: int = 3) -> Car:
    participants = random.sample(population, k)
    return max(participants, key=lambda c: c.fitness)


def crossover(parent1: Car, parent2: Car, p_crossover: float) -> tuple[Car, Car]:
    w1 = parent1.weights.copy()
    w2 = parent2.weights.copy()
    num_features = len(w1)

    if random.random() < p_crossover and num_features > 1:
        cp = random.randint(1, num_features - 1)
        child1_w = np.concatenate([w1[:cp], w2[cp:]])
        child2_w = np.concatenate([w2[:cp], w1[cp:]])
    else:
        child1_w = w1.copy()
        child2_w = w2.copy()

    return Car(weights=child1_w), Car(weights=child2_w)


def mutate(car: Car, p_mutation: float, sigma: float):
    for i in range(len(car.weights)):
        if random.random() < p_mutation:
            car.weights[i] += np.random.normal(0, sigma)


def run_ga(
    pop_size: int = 50,
    num_generations: int = 300,
    num_features: int = 3,
    p_crossover: float = 0.8,
    p_mutation: float = 0.1,
    sigma: float = 0.2,
    max_steps: int = 1000,
    seed: int | None = None,
    log_dir: str = "data"
) -> tuple[Car, str]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(log_dir, exist_ok=True)
    generations_path = os.path.join(log_dir, "generations.json")
    stats_path = os.path.join(log_dir, "fitness_history.csv")

    # Prepare a CSV file for statistics
    with open(stats_path, "w") as sf:
        sf.write("gen,avg_fitness,max_fitness,median_fitness\n")

    # Population Initialization
    population: list[Car] = [
        Car(weights=np.random.uniform(-1, 1, size=num_features))
        for _ in range(pop_size)
    ]

    # Create a JSON file and save the opening "{"
    with open(generations_path, "w") as gf:
        gf.write("{\n")

    # Create environment once (no render)
    env = make_env(render=False)

    for gen in range(num_generations):
        print(f"--- Generation {gen} ---")

        # Evaluation of each unit
        fitnesses = []
        for car in population:
            fit = evaluate_car(car, env, max_steps=max_steps, seed=seed)
            fitnesses.append(fit)

        avg_fit = float(np.mean(fitnesses))
        max_fit = float(np.max(fitnesses))
        med_fit = float(np.median(fitnesses))

        # Saving statistics to CSV
        with open(stats_path, "a") as sf:
            sf.write(f"{gen},{avg_fit:.5f},{max_fit:.5f},{med_fit:.5f}\n")

        # Saving this generation's population to JSON
        gen_record = []
        for idx, car in enumerate(population):
            gen_record.append({
                "id": idx,
                "weights": car.weights.tolist(),
                "fitness": float(car.fitness),
                "disqualified": bool(car.disqualified)
            })

        # If not the last generation, we add a comma at the end
        sep = "," if gen < num_generations - 1 else ""
        with open(generations_path, "a") as gf:
            gf.write(f'  "{gen}": {json.dumps(gen_record, indent=2)}{sep}\n')

        # Selection
        parents: list[Car] = [
            tournament_selection(population, k=3)
            for _ in range(pop_size)
        ]

        # Crossbreeding and mutation → new population
        random.shuffle(parents)
        new_population: list[Car] = []
        for i in range(0, pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = crossover(parent1, parent2, p_crossover)
            mutate(child1, p_mutation, sigma)
            mutate(child2, p_mutation, sigma)
            new_population.extend([child1, child2])

        population = new_population

    # Close JSON (saving "}")
    with open(generations_path, "a") as gf:
        gf.write("}\n")

    # Last generation evaluation
    for car in population:
        _ = evaluate_car(car, env, max_steps=max_steps, seed=seed)

    # Save the best agent
    best_car = max(population, key=lambda c: c.fitness)
    best_path = os.path.join(log_dir, f"best_individual_gen{num_generations}.json")
    with open(best_path, "w") as bf:
        json.dump(best_car.to_json(), bf, indent=2)

    env.close()

    print(f"All generations saved: {generations_path}")
    print(f"Best agent from gen {num_generations} saved: {best_path}")
    return best_car, best_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch GA for CarRacing")
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument("--p_crossover", type=float, default=0.8)
    parser.add_argument("--p_mutation", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="data")
    args = parser.parse_args()

    run_ga(
        pop_size=args.pop_size,
        num_generations=args.generations,
        num_features=args.features,
        p_crossover=args.p_crossover,
        p_mutation=args.p_mutation,
        sigma=args.sigma,
        max_steps=args.max_steps,
        seed=args.seed,
        log_dir=args.log_dir
    )
