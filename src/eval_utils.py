# eval_utils.py

from agent import Car
from env_utils import extract_distances, is_on_grass
import numpy as np

def evaluate_car(
    car: Car,
    env,
    max_steps: int = 1000,
    seed: int | None = None,
    grass_penalty_per_frame: float =  -50.0,
    render: bool = False,
    speed_penalty: bool = False,
    min_speed: float = 0.3,
    speed_penalty_scale: float = 50.0
) -> float:
    """
    Ewaluacja jednego samochodu (Car). Zwraca obliczony fitness.
    - Za każdą klatkę, w której auto jest na trawie, odejmujemy `grass_penalty_per_frame`.
      (Symulacja nie jest przerywana.)
    - Jeśli speed_penalty=True, na koniec obliczamy średnią prędkość i doliczamy karę
      za zbyt niską średnią prędkość.
    """
    # Reset środowiska (z podaniem ziarna, jeśli jest)
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    total_reward = 0.0
    done = False
    step_count = 0

    # Lista prędkości, by po epizodzie wyliczyć średnią
    speed_list = []

    while not done and step_count < max_steps:
        if render:
            env.render()

        # 1) Sprawdzenie odległości w 9 kierunkach
        distances = extract_distances(obs)
        
        # 2) Decyzja agenta i wykonanie akcji
        action = car.decide(distances)
        
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # 3) Jeśli auto jest na trawie, odejmujemy karę (ale nie przerywamy)
        if step_count > 0 and is_on_grass(obs):
            total_reward += grass_penalty_per_frame

        # 4) Zbieranie prędkości (jeśli dostępna)
        try:
            current_speed = info.get("speed", None)
            if current_speed is None:
                vel = env.car.hull.linearVelocity
                current_speed = np.linalg.norm([vel[0], vel[1]])
        except Exception:
            current_speed = 0.0

        speed_list.append(current_speed)

        done = terminated or truncated
        step_count += 1

    # 5) Kara za prędkość (jeśli włączona)
    if speed_penalty and len(speed_list) > 0:
        avg_speed = float(np.mean(speed_list))
        if avg_speed < min_speed:
            penalty = (min_speed - avg_speed) * speed_penalty_scale
            total_reward -= penalty

    # 6) Ustawiamy fitness i zwracamy
    car.fitness = total_reward
    return total_reward
