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
        if not isinstance(distances, np.ndarray) or distances.shape != (3,):
            raise ValueError("distances must be numpy.ndarray of shape (3,)")

        d_left, d_right, d_front = distances

        # Steering: tanh(w0 * (d_right - d_left)) → w ∈ [-1, 1]
        raw_steer = self.weights[0] * (d_right - d_left)
        steering = np.tanh(raw_steer)

        # Gas: sigmoid(w1 * d_front) → w ∈ [0, 1]
        raw_gas = self.weights[1] * d_front
        gas = 1.0 / (1.0 + np.exp(-raw_gas))

        # Brake: sigmoid(w2 * (1 - d_front)) → w ∈ [0, 1]
        raw_brake = self.weights[2] * (1.0 - d_front)
        brake = 1.0 / (1.0 + np.exp(-raw_brake))

        return np.array([steering, gas, brake], dtype=np.float32)

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
