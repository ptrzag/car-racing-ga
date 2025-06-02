from agent import Car
from env_utils import extract_distances, is_on_grass


def evaluate_car(
    car: Car,
    env,
    max_steps: int = 1000,
    seed: int | None = None,
    grass_penalty: float = 0.1,
    render: bool = False
) -> float:
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    total_reward = 0.0
    done = False
    step_count = 0
    car.disqualified = False

    while not done and step_count < max_steps:
        if render:
            env.render()

        distances = extract_distances(obs)

        if step_count > 0 and is_on_grass(obs):
            car.disqualified = True
            break

        action = car.decide(distances)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        done = terminated or truncated
        step_count += 1

    fitness = total_reward * (grass_penalty if car.disqualified else 1.0)
    car.fitness = fitness
    return fitness
