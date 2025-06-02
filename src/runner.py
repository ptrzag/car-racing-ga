import json
import gymnasium as gym
import random
import numpy as np
from agent import Car
from eval_utils import evaluate_car


def load_car_from_json(path: str) -> Car:
    with open(path, 'r') as f:
        data = json.load(f)

    weights = np.array(data["weights"], dtype = np.float32)
    car = Car(weights)

    car.fitness = data.get("fitness", None)
    car.disqualified = data.get("disqualified", False)

    return car

def run_single(
        car: Car, 
        seed: int = None, 
        max_steps: int = 1000,
        grass_penalty: float = 0.1
        ) -> float:
    
    if seed is None:
        seed = random.randint(0, 999_999)
        print(f"No seed given - drawn: {seed}")

    # Create an environment with human mode
    env = gym.make("CarRacing-v3", render_mode="human")

    final_fitness = evaluate_car(
        car = car,
        env = env,
        max_steps = max_steps,
        seed = seed,
        grass_penalty = grass_penalty,
        render = True
    )

    env.close()

    if car.disqualified:
        print(f"Agent disqualified. Fitness (with penalty): {final_fitness:.2f}")
    else:
        print(f"Episode ended normally. Fitness: {final_fitness:.2f}")

    return final_fitness

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = "Run a trained CarRacing agent with JSON"
    )

    parser.add_argument("json_path", type = str, help = "Path to JSON file with weights") 
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed (if not provided, random will be drawn)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of steps")
    parser.add_argument("--grass_penalty", type=float, default=0.1,
        help="Penalty multiplier for driving on grass")
    
    args = parser.parse_args()

    car = load_car_from_json(args.json_path)
    print("Loaded agent with weights:", car.weights)

    run_single(
        car=car,
        seed=args.seed,         
        max_steps=args.max_steps,
        grass_penalty=args.grass_penalty
    )