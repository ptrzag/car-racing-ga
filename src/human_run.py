import gymnasium as gym
import pygame
import numpy as np
import time

# PyGame key mappings to car actions
def get_action_from_keys(keys):
    action = np.array([0.0, 0.0, 0.0])  # [steer, gas, brake]

    if keys[pygame.K_LEFT]:
        action[0] = -1.0  # steer left
    elif keys[pygame.K_RIGHT]:
        action[0] = 1.0   # steer right

    if keys[pygame.K_UP]:
        action[1] = 1.0   # gas
    if keys[pygame.K_DOWN]:
        action[2] = 0.8   # brake

    return action

def main():
    pygame.init()
    screen = pygame.display.set_mode((200, 100))  # Small dummy window just to capture keyboard input
    pygame.display.set_caption("CarRacing Controller")

    env = gym.make("CarRacing-v3", render_mode="human", domain_randomize=False)
    obs, info = env.reset(seed=71)

    clock = pygame.time.Clock()
    running = True
    done = False

    while running and not done:
        # Check for quit or key press
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                done = True

        keys = pygame.key.get_pressed()
        action = get_action_from_keys(keys)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        clock.tick(60)  # Run at 60 FPS

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
