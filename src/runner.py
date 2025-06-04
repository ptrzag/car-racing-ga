# runner.py

import json
import argparse
from agent import Car
from eval_utils import evaluate_car
from env_utils import make_env


def run_single_car(path: str, max_steps: int = 1000, render: bool = True):
    # Load saved car weights and metadata
    with open(path, "r") as f:
        data = json.load(f)

    # Recreate Car object
    car = Car.from_json(data)

    # Create environment with rendering
    env = make_env(render=render)

    # Evaluate the car (with rendering)
    fitness = evaluate_car(
        car=car,
        env=env,
        max_steps=max_steps,
        render=render,
        speed_penalty=True,         # Can be adjusted or made optional
        min_speed=0.3,
        speed_penalty_scale=50.0
    )

    print(f"Car fitness: {fitness:.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained car agent")
    parser.add_argument("path", type=str, help="Path to saved car JSON (e.g., data/best_individual_gen300.json)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max number of steps")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")

    args = parser.parse_args()
    run_single_car(path=args.path, max_steps=args.max_steps, render=not args.no_render)
