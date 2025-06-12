# agent.py

import numpy as np
from config import USE_EXTENDED_DISTANCES

class Car:
    def __init__(self, weights: np.ndarray):
        if weights.ndim != 1:
            raise ValueError("weights must be 1D")
        self.weights      = weights.astype(np.float32)
        self.fitness      = None
        self.disqualified = False

    def decide(self, distances: np.ndarray) -> np.ndarray:
        expected = 17 if USE_EXTENDED_DISTANCES else 9
        if distances.shape != (expected,):
            raise ValueError(f"Expected distances shape {(expected,)}, got {distances.shape}")
        if self.weights.size != expected*3:
            raise ValueError(f"weights length must be {expected*3}")

        # split
        s_end = expected
        g_end = expected*2
        w = self.weights
        steer = np.tanh( np.dot(w[:s_end], distances) )
        gas   = 1/(1+np.exp(-np.dot(w[s_end:g_end], distances)))
        brake = 1/(1+np.exp(-np.dot(w[g_end:], distances)))
        return np.array([steer, gas, brake], dtype=np.float32)

    def to_json(self) -> dict:
        return {
            "weights":      self.weights.tolist(),
            "fitness":      None if self.fitness is None else float(self.fitness),
            "disqualified": bool(self.disqualified)
        }

    @classmethod
    def from_json(cls, data: dict) -> "Car":
        w = np.array(data["weights"], dtype=np.float32)
        car = cls(w)
        car.fitness      = data.get("fitness", None)
        car.disqualified = data.get("disqualified", False)
        return car
