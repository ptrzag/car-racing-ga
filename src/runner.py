# runner.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import json
import argparse
import numpy as np

from agent import Car
from eval_utils import evaluate_car

def main(best_json, max_steps, render, seed):
    with open(best_json, "r") as f:
        data = json.load(f)
    weights = np.array(data["weights"], dtype=np.float32)
    car = Car(weights)
    car.fitness      = data.get("fitness")
    car.disqualified = data.get("disqualified", False)

    # Use seed from JSON if not provided
    seed = seed if seed is not None else data.get("seed", None)

    fit = evaluate_car(car, max_steps=max_steps, seed=seed, render=render)
    print(f"Runner fitness: {fit:.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("best_json", help="Path to best.json")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--no_render", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    main(args.best_json, args.max_steps, not args.no_render, args.seed)
