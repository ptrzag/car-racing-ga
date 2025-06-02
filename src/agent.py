import numpy as np


class Car:
    def __init__(self, weights: np.ndarray):
        self.weights = weights # [w_left, w_right, w_acc, w_brake]
        self.fitness = None
        self.disqualified = False
    
    def decide(self, distances: np.ndarray) -> np.ndarray:
        # Get the distance
        d_left, d_right, d_front = distances

        # Calculate steering
        raw_steer = self.weights[0] * (d_right - d_left)
        steering = np.tanh(raw_steer) # steering = [-1; +1]

        # Calculate acceleration
        raw_gas = self.weights[1] * d_front
        gas = 1.0 / (1.0 + np.exp(-raw_gas)) # gas = [0; +1]

        # Calculate brake
        raw_brake = self.weights[2] * (1.0 - d_front)
        brake = 1.0 / (1.0 + np.exp(-raw_brake)) # brake = [0, +1]

        return np.array([steering, gas, brake], dtype=np.float32)
    
    def to_json(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "fitness": float(self.fitness) if self.fitness is not None else None,
            "disqualified": bool(self.disqualified)
        }
    
    @classmethod
    def from_json(cls, data: dict) -> "Car":
        # Loading a list of weights and converting to ndarray
        weights_list = data.get("weights", [])
        weights_array = np.array(weights_list, dtype = np.float32)

        # Initializing the Car instance
        car = cls(weights = weights_array)

        # Overwriting additional attributes
        if data.get("fitness", None) is not None:
            car.fitness = float(data["fitness"])
        else:
            car.fitness = None

        car.disqualified = bool(data.get("disqualified", False))

        return car