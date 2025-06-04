import numpy as np


class Car:
    def __init__(self, weights: np.ndarray):
        if not isinstance(weights, np.ndarray):
            raise TypeError("weights must be of type numpy.ndarray")
        if weights.ndim != 1:
            raise ValueError("weights must be a one-dimensional vector")
        self.weights: np.ndarray = weights
        self.fitness: float | None = None
        self.disqualified: bool = False

    def decide(self, distances: np.ndarray) -> np.ndarray:
        
        if not isinstance(distances, np.ndarray) or distances.shape != (9,):
            raise ValueError("distances must be numpy.ndarray of shape (9,)")

        # Ensure weights length = 9 * 3 = 27 (9 for steering, 9 for gas, 9 for brake)
        if self.weights.shape[0] != 27:
            raise ValueError("weights vector must have 27 values for 9 inputs")

        w = self.weights
        steer_raw = np.tanh(np.dot(w[0:9], distances))  # steering ∈ [-1, 1]
        # print(f"{w[0:9]=} {distances=} {steer_raw=}")
        gas_raw = 1.0 / (1.0 + np.exp(-np.dot(w[9:18], distances)))  # sigmoid
        brake_raw = 1.0 / (1.0 + np.exp(-np.dot(w[18:27], distances)))  # sigmoid

        return np.array([steer_raw, gas_raw, brake_raw], dtype=np.float32)

    def to_json(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "fitness": float(self.fitness) if self.fitness is not None else None,
            "disqualified": bool(self.disqualified)
        }

    @classmethod
    def from_json(cls, data: dict) -> "Car":
        if "weights" not in data:
            raise KeyError("Missing 'weights' key in JSON")
        weights_list = data["weights"]
        weights_array = np.array(weights_list, dtype=np.float32)
        car = cls(weights=weights_array)

        if data.get("fitness", None) is not None:
            car.fitness = float(data["fitness"])
        car.disqualified = bool(data.get("disqualified", False))
        return car
