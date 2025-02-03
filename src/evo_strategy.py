from __future__ import annotations
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, List
from .strategies import Player
import pickle
import os
from collections import deque


class EvoStrategy(Player):
    instance_counter = 0  # Class-level counter for naming instances
    lineage: List[Tuple[str, str, str]] = []  # Lineage tracking for evolutionary strategies

    def __init__(self, name: Optional[str] = None, action_history_size: int = 10, log_history: bool=False) -> None:
        """
        Initializes an EvoStrategy instance with automatic naming and tracking.

        :param name: Optional name of the instance. Automatically generated if None.
        :param action_history_size: Number of actions in the history. If zero, memory is not used.
        """
        name = name or self._generate_name()
        super().__init__(name)
        self.parents: Tuple[str, str] = ('', '')  # Track parent names
        self.stochastic = False # We use distributions only to initialize the weights and then for mutation and crossover

        # Determine state_size based on action_history_size
        if action_history_size > 0:
            self.state_size = 8  # 2 for memory counts ('C' and 'D') + 6 existing state variables
        else:
            self.state_size = 6  # Only the existing state variables

        self._weights = self._rng.normal(0, 1, self.state_size)
        self._state_scaler = MinMaxScaler()
        self._state = np.zeros(self.state_size, dtype=np.float64)

        # Initialize memory only if action_history_size is non-zero
        self.action_history_size = action_history_size
        if self.action_history_size > 0:
            self._memory = deque(maxlen=self.action_history_size)
            for _ in range(self.action_history_size):
                move = self._rng.choice(['C', 'D'])
                self._memory.append(move)

            # Initialize counts
            self._count_C = sum(1 for move in self._memory if move == 'C')
            self._count_D = self.action_history_size - self._count_C
        else:
            self._memory = None  # No memory used
            self._count_C = 0
            self._count_D = 0

        self.log_history = log_history
        self._full_history = []
        self._opponent_history = []
        # Number of matches won, lost, and tied
        self._log_match_results: List[int, int, int] = [0, 0, 0]

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
        
    
    @property
    def weights(self) -> np.ndarray:
        """Retrieves the weights of the EvoStrategy."""
        return self._weights
    
    
    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        """Sets the weights of the EvoStrategy."""
        self._weights = weights
        

    def get_state(self, opponent: Player) -> np.ndarray:
        """Generates the current state vector for this strategy."""
        # Gather existing state variables
        state_integers = np.array([
            self.cooperations,
            self.defections,
            self.score,
            opponent.cooperations,
            opponent.defections,
            opponent.score
        ])

        if self.action_history_size > 0:
            # Incorporate counts of 'C' and 'D' into the state
            memory_counts = np.array([self._count_C, self._count_D])
            # Combine all state components
            full_state = np.concatenate((memory_counts, state_integers))
        else:
            # Only existing state variables
            full_state = state_integers

        # Normalize the entire state once
        normalized_full_state = self._state_scaler.fit_transform(full_state.reshape(-1, 1)).flatten()
        self._state = normalized_full_state

        return self._state


    def update(self, move: str, score: int) -> None:
        """Updates the strategy's state based on the latest move and score."""
        super().update(move, score)
        if self.action_history_size > 0 and self._memory is not None:
            # Update counts based on the new move
            oldest_move = self._memory.popleft()
            if oldest_move == 'C':
                self._count_C -= 1
            else:
                self._count_D -= 1

            self._memory.append(move)
            if move == 'C':
                self._count_C += 1
            else:
                self._count_D += 1


    def strategy(self, opponent: Player) -> str:
        """Determines the next move ('C' or 'D') based on the current state."""
        self.get_state(opponent)
        action = 'C' if np.dot(self._weights, self._state) >= 0 else 'D'
        if self.log_history:
            # New opponent
            if not opponent.history:
                self._opponent_history.append('*')    
            else:
                self._opponent_history.append(opponent.history[-1])
            
            if not self.history:
                self._full_history.append('*')
            else:
                self._full_history.append(action)
            
        return action


    def reset_full_history(self) -> None:
        """Resets the full history of actions."""
        self._full_history = []
        self._opponent_history = []
        self._log_match_results = [0, 0, 0]
        
    
    def update_log_match_results(self, match_winner: str) -> None:
        """Updates the match results for the strategy."""
        if match_winner == self.name:
            self._log_match_results[0] += 1
        elif match_winner == 'Tie':
            self._log_match_results[2] += 1
        else:
            self._log_match_results[1] += 1

    @property
    def log_match_results(self) -> List[int, int, int]:
        """Retrieves the match results for the strategy."""
        return self._log_match_results

    @property
    def full_history(self) -> Tuple[List[str], List[str]]:
        """Retrieves the full history of actions."""
        return self._full_history, self._opponent_history


    def mutate(self, mutation_rate: float) -> None:
        """Applies mutation to the weights."""
        self._weights += self._rng.normal(0, mutation_rate, self.state_size)


    def crossover(self, other: EvoStrategy, strategy: str = "adaptive_weighted") -> Tuple[EvoStrategy, EvoStrategy]:
        """
        Produces two new EvoStrategy instances as offspring of this instance and another.

        :param other: Another EvoStrategy instance.
        :param strategy: Crossover strategy ("adaptive_weighted", "BLX-α", "random_subset").
        :return: Tuple of two new EvoStrategy instances (offspring1, offspring2).
        """

        # Determine the action_history_size based on the better parent
        best_parent = self if self.score > other.score else other
        action_history_size = best_parent.action_history_size

        # Correctly generate unique names using the class method
        offspring1_name = type(self)._generate_name()
        offspring2_name = type(self)._generate_name()

        offspring1 = EvoStrategy(name=offspring1_name, action_history_size=action_history_size)
        offspring2 = EvoStrategy(name=offspring2_name, action_history_size=action_history_size)

        # Crossover strategies
        if strategy == "adaptive_weighted":
            total_score = self.score + other.score
            alpha = self.score / total_score if total_score > 0 else 0.5
            offspring1.weights = alpha * self._weights + (1 - alpha) * other.weights
            offspring2.weights = (1 - alpha) * self._weights + alpha * other.weights
        elif strategy == "BLX-alpha":
            alpha = 0.5  # Adjustable parameter
            min_weights = np.minimum(self._weights, other.weights)
            max_weights = np.maximum(self._weights, other.weights)
            range_weights = max_weights - min_weights
            offspring1.weights = self._rng.uniform(min_weights - alpha * range_weights,
                                                  max_weights + alpha * range_weights)
            offspring2.weights = self._rng.uniform(min_weights - alpha * range_weights,
                                                  max_weights + alpha * range_weights)
        elif strategy == "random_subset":
            mask = self._rng.rand(self.state_size) > 0.5
            offspring1.weights = np.where(mask, self._weights, other.weights)
            offspring2.weights = np.where(mask, other.weights, self._weights)
        else:
            raise ValueError("Unsupported crossover strategy.")

        return offspring1, offspring2
    
    
    def save_strategy(self, filename) -> None:
        """Saves an EvoStrategy instance using pickle."""
        if not os.path.exists('strategies'):
            os.makedirs('strategies')
        filepath = f'strategies/{filename}_{self.name}.pkl'
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
