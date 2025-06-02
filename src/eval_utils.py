from agent import Car
from env_utils import extract_distances, is_on_grass


def evaluate_car(
    car: Car,
    env,
    max_steps: int = 1000,
    seed = None,
    grass_penalty: float = 0.1,
    render: bool = False
) -> float:
    # Environment reset
    if seed is not None:
        obs, info = env.reset(seed = seed)
    else:
        obs, info = env.reset()

    total_reward = 0.0
    done = False
    step_count = 0
    car.disqualified = False
    first_step = True

    while not done and step_count < max_steps:
        if render:
            env.render()

        distances = extract_distances(obs)

        # Check if car is on grass
        if not first_step and is_on_grass(obs):
            car.disqualified = True
            break

        # Decision based on weights
        action = car.decide(distances)

        # Take a step into the environment
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Check if environment has ended the episode
        if terminated or truncated:
            done = True
            break

        first_step = False
        step_count += 1

    # Calculate fitness
    if car.disqualified:
        fitness = total_reward * grass_penalty
    else:
        fitness = total_reward

    # Save in car object
    car.fitness = fitness

    return fitness