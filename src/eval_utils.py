# eval_utils.py

from typing import Optional
import numpy as np
import gymnasium as gym

from config import (
    MAX_STEPS
)
from agent import Car
from env_utils import extract_distances, is_on_grass

def evaluate_car(
    car: Car, max_steps: int = MAX_STEPS,
    seed: Optional[int] = None, render: bool = False
) -> float:
    env = gym.make("CarRacing-v3", render_mode="human" if render else None)
    base = env.unwrapped

    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        if render:
            env.render()

        d = extract_distances(obs)
        action = car.decide(d)
        obs, rew, term, trunc, info = env.step(action)
        total_reward += rew

        if term or trunc:
            break

    car.fitness = total_reward
    env.close()
    return total_reward

