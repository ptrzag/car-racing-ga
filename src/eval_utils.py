# eval_utils.py

import numpy as np
import gymnasium as gym

from config import (
    GRASS_PENALTY_PER_FRAME, MAX_STEPS,
    SPEED_PENALTY, MIN_SPEED, SPEED_PENALTY_SCALE,
    MAX_GRASS_FRAMES, SPEED_BONUS
)
from agent import Car
from env_utils import extract_distances, is_on_grass

def evaluate_car(
    car: Car, max_steps: int = MAX_STEPS,
    seed: int|None = None, render: bool = False
) -> float:
    env = gym.make("CarRacing-v3", render_mode="human" if render else None)
    base = env.unwrapped

    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    total_reward = 0.0
    speed_list   = []
    grass_count  = 0

    for _ in range(max_steps):
        if render:
            env.render()

        d = extract_distances(obs)
        action = car.decide(d)
        obs, rew, term, trunc, info = env.step(action)
        total_reward += rew

        if is_on_grass(obs):
            grass_count += 1
            total_reward += GRASS_PENALTY_PER_FRAME
            if grass_count >= MAX_GRASS_FRAMES:
                car.disqualified = True
                car.fitness = float('-inf')
                env.close()
                return car.fitness
        else:
            grass_count = 0

        sp = info.get("speed")
        if sp is None:
            vel = base.car.hull.linearVelocity
            sp = np.linalg.norm([vel[0], vel[1]])
        speed_list.append(sp)

        if term or trunc:
            break

    if SPEED_PENALTY and speed_list:
        avg = float(np.mean(speed_list))
        total_reward -= max(0.0, MIN_SPEED - avg) * SPEED_PENALTY_SCALE
        total_reward += SPEED_BONUS * avg

    car.fitness = total_reward
    env.close()
    return total_reward
