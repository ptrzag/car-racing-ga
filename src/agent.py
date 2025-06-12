import numpy as np

class Car:
    """
    Represents a car controlled by a set of weights.
    Decision logic now fully NumPy-vectorized.
    """
    def __init__(self, weights: np.ndarray):
        self.weights = weights.astype(np.float32)
        self.fitness: float | None = None
        self.disqualified: bool = False

    def decide(self, distances: np.ndarray) -> np.ndarray:
        """
        Vectorized steering, gas, brake actions from distance vector.
        distances shape: (n_features,)
        Returns actions shape: (3,)
        """
        w = self.weights.reshape(3, -1)
        steer = np.tanh(w[0] @ distances)
        gas   = 1.0 / (1.0 + np.exp(- (w[1] @ distances)))
        brake = 1.0 / (1.0 + np.exp(- (w[2] @ (1.0 - distances))))
        return np.array([steer, gas, brake], dtype=np.float32)

    def to_json(self) -> dict:
        return {"weights": self.weights.tolist(),
                "fitness": self.fitness,
                "disqualified": self.disqualified}

    @staticmethod
    def from_json(data: dict) -> "Car":
        w = np.array(data["weights"], dtype=np.float32)
        car = Car(w)
        car.fitness = data.get("fitness")
        car.disqualified = data.get("disqualified", False)
        return car