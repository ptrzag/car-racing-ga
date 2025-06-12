import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from config import (GRASS_PENALTY_PER_FRAME, MAX_STEPS,
                    SPEED_PENALTY, MIN_SPEED, SPEED_PENALTY_SCALE,
                    MAX_GRASS_FRAMES, SPEED_BONUS)
from agent import Car
from env_utils import extract_distances, mask_grass, to_hsv


def make_env(seed=None):
    def _init():
        env = gym.make("CarRacing-v3")
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def evaluate_population(
    population: list[Car], max_steps: int = MAX_STEPS,
    seed: int|None = None
) -> list[float]:
    """
    Vectorized evaluation using AsyncVectorEnv over population.
    Returns list of fitnesses.
    """
    n = len(population)
    envs = AsyncVectorEnv([make_env(seed) for _ in range(n)])
    obs, _ = envs.reset()
    total_reward = np.zeros(n, dtype=np.float32)
    grass_count = np.zeros(n, dtype=int)
    speed_lists = [[] for _ in range(n)]

    for _ in range(max_steps):
        # extract distances for each env
        dists = np.stack([extract_distances(o) for o in obs])
        # compute actions
        acts = np.stack([car.decide(d) for car, d in zip(population, dists)])
        obs, rews, terms, truncs, infos = envs.step(acts)
        total_reward += rews
        # grass penalty & disqualification
        hsvs = [None]*n  # placeholder if needed
        grasses = [mask_grass(to_hsv(o))[90,48] for o in obs]
        for i, on_grass in enumerate(grasses):
            if on_grass:
                grass_count[i] += 1
                total_reward[i] += GRASS_PENALTY_PER_FRAME
                if grass_count[i] >= MAX_GRASS_FRAMES:
                    population[i].disqualified = True
                    total_reward[i] = -np.inf
            else:
                grass_count[i] = 0
        # speed tracking
        for i, info in enumerate(infos):
            sp = info.get("speed")
            if sp is None:
                vel = envs.envs[i].unwrapped.car.hull.linearVelocity
                sp = np.linalg.norm([vel[0], vel[1]])
            speed_lists[i].append(sp)
        if all(terms | truncs):
            break

    # post-process speed penalties
    for i, speeds in enumerate(speed_lists):
        if SPEED_PENALTY and speeds and total_reward[i] > -np.inf:
            avg = float(np.mean(speeds))
            total_reward[i] -= max(0.0, MIN_SPEED - avg) * SPEED_PENALTY_SCALE
            total_reward[i] += SPEED_BONUS * avg
        population[i].fitness = total_reward[i]
    envs.close()
    return total_reward.tolist()