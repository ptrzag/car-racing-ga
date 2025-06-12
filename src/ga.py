import argparse
from ga_utils import run_ga
import config

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pop_size", type=int, default=config.POP_SIZE)
    p.add_argument("--generations", type=int, default=config.GENERATIONS)
    p.add_argument("--p_crossover", type=float, default=config.P_CROSSOVER)
    p.add_argument("--p_mutation", type=float, default=config.P_MUTATION)
    p.add_argument("--sigma", type=float, default=config.SIGMA)
    p.add_argument("--max_steps", type=int, default=config.MAX_STEPS)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_ga(
        pop_size=args.pop_size,
        num_generations=args.generations,
        p_crossover=args.p_crossover,
        p_mutation=args.p_mutation,
        sigma=args.sigma,
        max_steps=args.max_steps
    )