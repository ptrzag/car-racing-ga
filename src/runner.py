# runner.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import json
import argparse
import numpy as np
import os

from agent import Car
from eval_utils import evaluate_car

def load_car_from_best_json(path: str) -> tuple[Car, int | None]:
    with open(path, "r") as f:
        data = json.load(f)
    weights = np.array(data["weights"], dtype=np.float32)
    car = Car(weights)
    car.fitness = data.get("fitness")
    car.disqualified = data.get("disqualified", False)
    seed = data.get("seed", None)
    return car, seed

def load_best_from_generation(folder: str, gen_idx: int) -> tuple[Car, int | None]:
    gen_path = os.path.join(folder, f"gen_{gen_idx}.json")
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"Generation file not found: {gen_path}")
    
    with open(gen_path, "r") as f:
        cars_data = json.load(f)
    
    if not cars_data:
        raise ValueError("Generation file is empty.")

    # Find best car by fitness (skip disqualified)
    qualified_cars = [c for c in cars_data if not c.get("disqualified", False)]
    if not qualified_cars:
        raise ValueError("All cars in this generation are disqualified.")
    best_data = max(qualified_cars, key=lambda c: c.get("fitness", float("-inf")))
    weights = np.array(best_data["weights"], dtype=np.float32)
    car = Car(weights)
    car.fitness = best_data.get("fitness")
    car.disqualified = best_data.get("disqualified", False)
    seed = best_data.get("seed", None)
    return car, seed

def main(best_json, gen_folder, gen_idx, max_steps, render, seed):
    if best_json:
        car, saved_seed = load_car_from_best_json(best_json)
        print(f"Loaded car from: {best_json}")
        # Try to infer folder to check for meta.json
        meta_folder = os.path.dirname(best_json)
    elif gen_folder is not None and gen_idx is not None:
        car, saved_seed = load_best_from_generation(gen_folder, gen_idx)
        print(f"Loaded best car from generation {gen_idx} in {gen_folder}")
        meta_folder = gen_folder
    else:
        raise ValueError("You must specify either --best_json or (--gen_folder and --gen).")

    # Try meta.json if seed still missing
    if seed is None and saved_seed is None:
        meta_path = os.path.join(meta_folder, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            saved_seed = meta.get("seed", None)
            print(f"Loaded seed from meta.json: {saved_seed}")

    # Choose final seed
    final_seed = seed if seed is not None else saved_seed
    print(f"Using seed: {final_seed}")

    fit = evaluate_car(car, max_steps=max_steps, seed=final_seed, render=render)
    status = "DISQUALIFIED" if car.disqualified else "OK"
    print(f"Runner fitness: {fit:.2f} [{status}]")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--best_json", help="Path to best.json")
    p.add_argument("--gen_folder", help="Path to generation folder (e.g., data/generations/20250612_1530)")
    p.add_argument("--gen", type=int, help="Generation number to load (e.g., 5 for gen_5.json)")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--no_render", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    main(args.best_json, args.gen_folder, args.gen, args.max_steps, not args.no_render, args.seed)
