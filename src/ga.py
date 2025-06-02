import json
import os
import random
import numpy as np
import gymnasium as gym

from agent import Car
from eval_utils import evaluate_car

# Tournament of three
def tournament_selection(population, k = 3):
    participants = random.sample(population, k)

    return max(participants, key = lambda c: c.fitness)
    
# Single point crossover
def crossover(parent1, parent2, p_crossover):
    w1, w2 = parent1.weights.copy(), parent2.weights.copy()
    num_features = len(w1)

    if random.random() < p_crossover:
        cp = random.randint(1, num_features - 1)
        child1_w = np.concatenate([w1[:cp], w2[cp:]])
        child2_w = np.concatenate([w2[:cp], w1[cp:]])
    else:
        child1_w, child2_w = w1.copy(), w2.copy()
    
    return Car(weights = child1_w), Car(weights = child2_w)

def mutate(car, p_mutation, sigma):
    for i in range(len(car.weights)):
        if random.random() < p_mutation:
            car.weights[i] += np.random.normal(0, sigma)


def run_ga(
    pop_size = 50,
    num_generations = 300,
    num_features = 3,
    p_crossover = 0.8,
    p_mutation = 0.1,
    sigma = 0.2,
    max_steps = 1000,
    log_dir = "data"
):
    # Creates a directory and points to the file with saved logs
    os.makedirs(log_dir, exist_ok = True)
    generations_path = os.path.join(log_dir, "generations.json")

    # Population Initialization
    population = [
        Car(weights = np.random.uniform(-1, 1, size = num_features))
        for _ in range(pop_size)
    ]

    # Structure for saving results: {gen_nr: [list of dictionaries with id, weights, fitness, disqualified]}
    all_gens = {}

    # Loop through the generations
    for gen in range(num_generations):
        print(f"--- Generation {gen} ---")

        # Fitness assessment of each individual
        env = gym.make("CarRacing-v3", render_mode = None)
        for idx, car in enumerate(population):
            fitness = evaluate_car(car, env, max_steps)
            car.fitness = fitness

        env.close()

        # Collect statistics for the current generation
        gen_record = []
        for idx, car in enumerate(population):
            gen_record.append({
                "id": idx,
                "weights": car.weights.tolist(),
                "fitness": float(car.fitness),
                "disqualified": bool(car.disqualified)
            })
        all_gens[str(gen)] = gen_record

        # Selection
        parents = []
        for _ in range(pop_size):
            parent = tournament_selection(population, k = 3)
            parents.append(parent)

        # Crossover and mutation
        new_population = []

        random.shuffle(parents)
        for i in range(0, pop_size, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2, p_crossover)
            mutate(child1, p_mutation, sigma)
            mutate(child2, p_mutation, sigma)
            new_population.extend([child1, child2])

        # New population
        population = new_population

    # Writing to file only after a loop
    with open(generations_path, "w") as fp:
        json.dump(all_gens, fp, indent = 2)

    # After all generations are finished, find the best one
    best_car = max(population, key = lambda c: c.fitness)
    best_path = os.path.join(log_dir, f"best_individual_gen{num_generations}.json")
    with open(best_path, "w") as fp:
        json.dump({
                "weights": best_car.weights.toList(),
                "fitness": float(best_car.fitness),
                "disqualified": bool(best_car.disqualified)
            }, fp, indent = 2)

    print(f"All generations have been saved to {generations_path}")
    print(f"Best after gen {num_generations}: saved to {best_path}")
