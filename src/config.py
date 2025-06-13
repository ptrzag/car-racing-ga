# config.py

# GA hyperparameters
POP_SIZE = 20
GENERATIONS = 15
P_CROSSOVER = 0.9
P_MUTATION = 0.2
SIGMA = 0.5

# Simulation settings
MAX_STEPS = 1000
GRASS_PENALTY_PER_FRAME = -1.0
MAX_GRASS_FRAMES = 50      # Disqualify after this many consecutive grass frames

# Speed penalties & rewards
SPEED_PENALTY = True
MIN_SPEED = 3.0
SPEED_PENALTY_SCALE = 2.0
SPEED_BONUS = 2.0         # Positive reward factor for avg speed

# Always use extended distances (17 inputs)
USE_EXTENDED_DISTANCES = True
ELITE_COUNT = 2  # or maybe 2
