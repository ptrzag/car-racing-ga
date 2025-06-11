# config.py

# GA hyperparameters
POP_SIZE = 60
GENERATIONS = 100
P_CROSSOVER = 0.8
P_MUTATION = 0.05
SIGMA = 0.5

# Simulation settings
MAX_STEPS = 1000
GRASS_PENALTY_PER_FRAME = -1.0
MAX_GRASS_FRAMES = 50      # Disqualify after this many consecutive grass frames

# Speed penalties & rewards
SPEED_PENALTY = True
MIN_SPEED = 2.0
SPEED_PENALTY_SCALE = 1.0
SPEED_BONUS = 0.5          # Positive reward factor for avg speed

# Always use extended distances (17 inputs)
USE_EXTENDED_DISTANCES = True
