import json
import sys
import random
import numpy as np
import gymnasium as gym

from agent import Car
from eval_utils import evaluate_car
from env_utils import make_env


def load_car_from_json(path: str) -> Car:
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading file '{path}': {e}")
        sys.exit(1)

    return Car.from_json(data)


def run_single(
    car: Car,
    seed: int | None = None,
    max_steps: int = 1000,
    grass_penalty: float = 0.1
) -> float:
    if seed is None:
        seed = random.randint(0, 999_999)
        print(f"No seed given → drawn: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    env = make_env(render=True)
    fitness = evaluate_car(
        car=car,
        env=env,
        max_steps=max_steps,
        seed=seed,
        grass_penalty=grass_penalty,
        render=True
    )
    env.close()

    if car.disqualified:
        print(f"Agent disqualified. Fitness (after penalty): {fitness:.2f}")
    else:
        print(f"Episode Complete. Fitness: {fitness:.2f}")

    return fitness


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start agent from JSON")
    parser.add_argument("json_path", type=str, help="Path to the JSON file with weights")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps")
    parser.add_argument("--grass_penalty", type=float, default=0.1,
                        help="Grass Penalty Multiplier")
    args = parser.parse_args()

    car = load_car_from_json(args.json_path)
    print("Loaded agent with weights:", car.weights)

    run_single(
        car=car,
        seed=args.seed,
        max_steps=args.max_steps,
        grass_penalty=args.grass_penalty
    )
