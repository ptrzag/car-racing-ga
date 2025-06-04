import json
import os
import numpy as np
import multiprocessing

from agent import Car
from eval_utils import evaluate_car
from env_utils import make_env


def evaluate_individual(args):
    print("evaaluate_individual")
    """
    Helper function for multiprocessing that evaluates one Car instance.
    Creates its own environment to avoid sharing issues.
    """
    car, max_steps, grass_penalty_per_frame, speed_penalty, min_speed, speed_penalty_scale, seed = args
    env = make_env(render=False)
    fitness = evaluate_car(
        car=car,
        env=env,
        max_steps=max_steps,
        seed=seed,
        grass_penalty_per_frame=grass_penalty_per_frame,
        render=False,
        speed_penalty=speed_penalty,
        min_speed=min_speed,
        speed_penalty_scale=speed_penalty_scale
    )
    print(f"{fitness=}")
    env.close()
    print(f"{fitness=}")
    return fitness


def tournament_selection(population: list[Car], rng: np.random.Generator, k: int = 3) -> Car:
    participants = rng.choice(population, size=k, replace=False)
    return max(participants, key=lambda c: c.fitness)


def crossover(parent1: Car, parent2: Car, p_crossover: float, rng: np.random.Generator) -> tuple[Car, Car]:
    w1 = parent1.weights.copy()
    w2 = parent2.weights.copy()
    num_features = len(w1)

    if rng.random() < p_crossover and num_features > 1:
        cp = rng.integers(1, num_features)
        child1_w = np.concatenate([w1[:cp], w2[cp:]])
        child2_w = np.concatenate([w2[:cp], w1[cp:]])
    else:
        child1_w = w1.copy()
        child2_w = w2.copy()

    return Car(weights=child1_w), Car(weights=child2_w)


def mutate(car: Car, p_mutation: float, sigma: float, rng: np.random.Generator):
    for i in range(len(car.weights)):
        if rng.random() < p_mutation:
            car.weights[i] += rng.normal(0, sigma)
            car.weights[i] = np.clip(car.weights[i], -1.0, 1.0)


def run_ga(
    pop_size: int = 50,
    num_generations: int = 300,
    p_crossover: float = 0.8,
    p_mutation: float = 0.1,
    sigma: float = 0.2,
    max_steps: int = 1000,
    seed: int | None = None,
    log_dir: str = "data",
    speed_penalty: bool = False,      
    min_speed: float = 0.3,           
    speed_penalty_scale: float = 50.0
) -> tuple[Car, str]:
    rng = np.random.default_rng(seed=seed)
    
    # Create main log directory
    os.makedirs(log_dir, exist_ok=True)
    # Create a subdirectory for per-generation JSON files
    gens_dir = os.path.join(log_dir, "generations")
    os.makedirs(gens_dir, exist_ok=True)

    stats_path = os.path.join(log_dir, "fitness_history.csv")

    # Prepare a CSV file for statistics
    with open(stats_path, "w") as sf:
        sf.write("gen,avg_fitness,max_fitness,median_fitness,avg_w0,std_w0\n")

    # Population Initialization
    population: list[Car] = [
        Car(weights=rng.uniform(-1, 1, size=27))
        for _ in range(pop_size)
    ]

    # Multiprocessing pool
    pool = multiprocessing.Pool()

    for gen in range(num_generations):
        print(f"--- Generation {gen} ---")

        # Evaluation of each unit (parallel)
        # Prepare arguments for each process
        eval_args = [
            (
                car,
                max_steps,
                -50.0,  # grass_penalty_per_frame
                True,  # speed_penalty (always True during GA)
                0.3,   # min_speed
                100.0, # speed_penalty_scale
                seed
            )
            for car in population
        ]
        print("cos")
        fitnesses = pool.map(evaluate_individual, eval_args)

        # Assign fitnesses back to cars
        for car, fit in zip(population, fitnesses):
            car.fitness = fit

        avg_fit = float(np.mean(fitnesses))
        max_fit = float(np.max(fitnesses))
        med_fit = float(np.median(fitnesses))

        weights_matrix = np.array([c.weights for c in population])
        avg_w0 = float(np.mean(weights_matrix[:, 0]))
        std_w0 = float(np.std(weights_matrix[:, 0]))

        # Saving statistics to CSV
        with open(stats_path, "a") as sf:
            sf.write(f"{gen},{avg_fit:.5f},{max_fit:.5f},{med_fit:.5f},{avg_w0:.5f},{std_w0:.5f}\n")

         # Prepare this generation's JSON record
        gen_record = []
        for idx, car in enumerate(population):
            gen_record.append({
                "id": idx,
                "weights": car.weights.tolist(),
                "fitness": float(car.fitness),
                "disqualified": bool(car.disqualified)
            })

        # Write to a separate file per generation
        gen_file = os.path.join(gens_dir, f"generation_{gen}.json")
        with open(gen_file, "w") as gf:
            json.dump(gen_record, gf, indent=2)

        # Selection
        parents: list[Car] = [
            tournament_selection(population, rng, k=3)
            for _ in range(pop_size)
        ]

        # Crossbreeding and mutation → new population
        rng.shuffle(parents)
        new_population: list[Car] = []
        for i in range(0, pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = crossover(parent1, parent2, p_crossover, rng)
            mutate(child1, p_mutation, sigma, rng)
            mutate(child2, p_mutation, sigma, rng)
            new_population.extend([child1, child2])

        population = new_population

    pool.close()
    pool.join()

    # Last generation evaluation (serial, or could also parallelize similarly)
    for car in population:
        _ = evaluate_car(
            car, env=make_env(render=False),
            max_steps=max_steps,
            seed=seed,
            speed_penalty=speed_penalty,
            min_speed=min_speed,
            speed_penalty_scale=speed_penalty_scale
        )

    # Save the best agent
    best_car = max(population, key=lambda c: c.fitness)
    best_path = os.path.join(log_dir, f"best_individual_gen{num_generations}.json")
    with open(best_path, "w") as bf:
        json.dump(best_car.to_json(), bf, indent=2)

    print(f"All per-generation files saved in: {gens_dir}")
    print(f"Best agent from gen {num_generations} saved: {best_path}")
    return best_car, best_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch GA for CarRacing")
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--p_crossover", type=float, default=0.8)
    parser.add_argument("--p_mutation", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="data")
    parser.add_argument(
        "--speed_penalty",
        action="store_true",
        help="Włącz karę za zbyt niską średnią prędkość"
    )
    parser.add_argument(
        "--min_speed",
        type=float,
        default=0.3,
        help="Minimalna średnia prędkość: poniżej tego = kara"
    )
    parser.add_argument(
        "--speed_penalty_scale",
        type=float,
        default=50.0,
        help="Skala kary za prędkość poniżej min_speed"
    )
    args = parser.parse_args()

    run_ga(
        pop_size=args.pop_size,
        num_generations=args.generations,
        p_crossover=args.p_crossover,
        p_mutation=args.p_mutation,
        sigma=args.sigma,
        max_steps=args.max_steps,
        seed=args.seed,
        log_dir=args.log_dir,
        speed_penalty=args.speed_penalty,
        min_speed=args.min_speed,
        speed_penalty_scale=args.speed_penalty_scale
    )
