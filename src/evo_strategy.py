from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, List
from .strategies import Player
import pickle
import os


class EvoStrategy(Player):
    instance_counter = 0  # Class-level counter for naming instances
    lineage: List[Tuple[str, str, str]] = []  # Lineage tracking for evolutionary strategies

    def __init__(self, name: Optional[str] = None, action_history_size: int = 10) -> None:
        """
        Initializes an EvoStrategy instance with automatic naming and tracking.

        :param name: Optional name of the instance. Automatically generated if None.
        :param action_history_size: Number of actions in the history.
        """
        name = name or self._generate_name()
        super().__init__(name)
        self.parents: Tuple[str, str] = ('', '')  # Track parent names
        self.state_size = action_history_size + 6
        self.weights = self._rng.rand(self.state_size)
        self._state_scaler = MinMaxScaler()
        self._state = np.zeros(self.state_size, dtype=np.float64)
        self._actions_history = None
        if action_history_size > 0:
            num_true = action_history_size // 2
            self._actions_history = np.array([True] * num_true + [False] * (action_history_size - num_true))
            self._rng.shuffle(self._actions_history)
        self._full_history = []

    @classmethod
    def _generate_name(cls) -> str:
        """Generates a unique name for the strategy instance."""
        name = f"Evo_{cls.instance_counter}"
        cls.instance_counter += 1
        return name

    @classmethod
    def _record_lineage(cls, offspring_name: str, parent1_name: str, parent2_name: str) -> None:
        """Records the lineage of an offspring."""
        cls.lineage.append((offspring_name, parent1_name, parent2_name))

    def get_state(self, opponent: Player) -> np.ndarray:
        """Generates the current state vector for this strategy."""
        state_integers = np.array([
            self.cooperations,
            self.defections,
            self.score,
            opponent.cooperations,
            opponent.defections,
            opponent.score
        ])
        state_integers = self._state_scaler.fit_transform(state_integers.reshape(-1, 1)).flatten()
        if self._actions_history is not None:
            self._state = np.concatenate((self._actions_history, state_integers))
        else:
            self._state = state_integers
        return self._state

    def update(self, move: str, score: int) -> None:
        super().update(move, score)
        if self._actions_history is not None:
            self._actions_history = np.roll(self._actions_history, -1)
            self._actions_history[-1] = (False if move == 'D' else True)

    def strategy(self, opponent: Player) -> str:
        self.get_state(opponent)
        action = 'C' if np.dot(self.weights, self._state) > 0 else 'D'
        self._full_history.append(action)
        return action

    def reset_full_history(self) -> None:
        self._full_history = []

    def get_full_history(self) -> List[str]:
        return self._full_history

    def mutate(self, mutation_rate: float) -> None:
        """Applies mutation to the weights."""
        self.weights += self._rng.rand(self.state_size) * mutation_rate

    def crossover(self, other: 'EvoStrategy', strategy: str = "adaptive_weighted") -> 'EvoStrategy':
        """
        Produces a new EvoStrategy instance as the offspring of this instance and another.

        :param other: Another EvoStrategy instance.
        :param strategy: Crossover strategy ("adaptive_weighted", "BLX-α", "random_subset").
        :return: New EvoStrategy instance (offspring).
        """
        new_name = self._generate_name()
        best_parent = self if self.score > other.score else other
        action_history_size = best_parent.state_size - 6
        offspring = EvoStrategy(name=new_name, action_history_size=action_history_size)

        if strategy == "adaptive_weighted":
            alpha = self.score / (self.score + other.score) if self.score + other.score > 0 else 0.5
            offspring.weights = alpha * self.weights + (1 - alpha) * other.weights
        elif strategy == "BLX-α":
            α = 0.5  # Adjust this based on experimentation
            offspring.weights = self.weights + np.random.uniform(-α, α) * (other.weights - self.weights)
        elif strategy == "random_subset":
            mask = np.random.rand(self.state_size) > 0.5
            offspring.weights = np.where(mask, self.weights, other.weights)
        else:
            raise ValueError("Unsupported crossover strategy.")

        offspring.parents = (self.name, other.name)
        self._record_lineage(new_name, self.name, other.name)
        return offspring

    def save_strategy(self) -> None:
        """Saves an EvoStrategy instance using pickle."""
        if not os.path.exists('strategies'):
            os.makedirs('strategies')
        filepath = f'strategies/{self.name}.pkl'
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_strategy(cls, filename: str) -> 'EvoStrategy':
        """Loads an EvoStrategy instance from a pickle file."""
        filepath = f'strategies/{filename}.pkl'
        try:
            with open(filepath, 'rb') as file:
                strategy = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {filename} not found")
        print(f"Strategy loaded from {filepath}")
        return strategy
