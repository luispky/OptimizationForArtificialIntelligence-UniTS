import copy
import numpy as np
from .strategies import Strategy
from typing import List, Tuple, Dict, Union, final
from math import ceil, log

PAYOFF_MATRIX = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

def set_seed(seed: Union[int, None] = 42) -> None:
    """Sets the seed for both numpy and Python's random module."""
    if seed is not None:
        np.random.seed(seed)

class Match:
    """Class to simulate a match between two players with optional noise and a probability of ending."""

    def __init__(self, players: List[Strategy], turns: int = 1, noise: float = 0, prob_end: float = 0, seed: Union[int, None] = 42):
        """
        Initializes the match between two players.

        :param players: List of two players participating in the match.
        :param turns: Maximum number of turns per match.
        :param noise: Probability of introducing noise in actions.
        :param prob_end: Probability of ending the match after each round.
        :param seed: Seed for reproducibility.
        """
        self.seed = seed
        set_seed(self.seed)
        
        self.players = players
        self.turns = turns
        self.noise = noise
        self.prob_end = prob_end
        self.moves = []
        self.winner = None
        self.final_scores = (0, 0)

    def _flip_action(self, action: str) -> str:
        """Flips the action with a probability defined by the noise."""
        if np.random.rand() < self.noise:
            return 'D' if action == 'C' else 'C'
        return action

    def _sample_length(self) -> int:
        """Samples the match length based on the ending probability."""
        if self.prob_end is None or self.prob_end <= 0:
            return self.turns
        if self.prob_end >= 1:
            return 1
        random_value = np.random.rand()
        return int(ceil(log(1 - random_value) / log(1 - self.prob_end)))

    def play(self) -> List[Tuple[str, str]]:
        """Simulates the match and returns the moves."""
        self.players[0].reset()
        self.players[1].reset()

        effective_turns = min(self.turns, self._sample_length())

        for _ in range(effective_turns):
            move1 = self._flip_action(self.players[0].play(self.players[1]))
            move2 = self._flip_action(self.players[1].play(self.players[0]))
            
            self.moves.append((move1, move2))
            self.players[0].history.append(move1)
            self.players[1].history.append(move2)

            score1, score2 = PAYOFF_MATRIX[(move1, move2)]
            self.players[0].score += score1
            self.players[1].score += score2

        self.final_scores = (self.players[0].score, self.players[1].score)
        self.winner = self._determine_winner()
        return self.moves

    def _determine_winner(self) -> str:
        """Determines the winner of the match based on final scores."""
        if self.players[0].score > self.players[1].score:
            return self.players[0].name
        elif self.players[0].score < self.players[1].score:
            return self.players[1].name
        return 'Tie'

class Tournament:
    """Class to simulate a round-robin tournament between multiple strategies."""

    def __init__(self, players: List[Strategy], turns: int, repetitions: int = 1, noise: float = 0.0, prob_end: float = 0., seed: Union[int, None] = 42, axelrod: bool = False):
        """
        Initializes the tournament.

        :param players: List of participating strategies.
        :param turns: Maximum number of turns per match.
        :param repetitions: Number of repetitions for the round-robin tournament.
        :param noise: Probability of introducing noise in actions.
        :param prob_end: Probability of ending a match after each turn.
        :param seed: Seed for reproducibility.
        """
        self.seed = seed
        set_seed(self.seed)
        
        self.players = players
        self.turns = turns
        self.repetitions = repetitions
        self.noise = noise
        self.prob_end = prob_end
        self.scores = {player.name: [] for player in players}
        self.axelrod = axelrod

    def play_match(self, player1: Strategy, player2: Strategy) -> Tuple[int, int]:
        """Plays a match between two players and returns their final scores."""
        match = Match([player1, player2], self.turns, self.noise, self.prob_end, self.seed)
        match.play()
        return match.final_scores

    def play(self) -> None:
        """Simulates the round-robin tournament."""
        for _ in range(self.repetitions):
            matches_scores = {player.name: [] for player in self.players}
            for i in range(len(self.players)):
                for j in range(i + 1, len(self.players)):
                    player1 = self.players[i]
                    player2 = self.players[j]
                    final_scores = self.play_match(player1, player2)
                    matches_scores[player1.name].append(final_scores[0])
                    matches_scores[player2.name].append(final_scores[1])
                if self.axelrod:
                    player1 = self.players[i]
                    player2 = copy.deepcopy(player1)
                    final_scores = self.play_match(player1, player2)
                    matches_scores[player1.name].append(final_scores[0])

            for player in self.players:
                self.scores[player.name].append(sum(matches_scores[player.name]))

    def get_ranked_results(self) -> Dict[str, float]:
        """Ranks players based on their accumulated scores."""
        if not self.scores:
            raise ValueError("The tournament has not been played yet.")
        
        ranked_results = {player: float(np.mean(scores)/self.repetitions) for player, scores in self.scores.items()}
        ranked_results = dict(sorted(ranked_results.items(), key=lambda item: item[1], reverse=True))
        return ranked_results
    
    def print_ranked_results(self) -> None:
        """Prints the ranked results of the tournament."""
        ranked_results = self.get_ranked_results()
        print("\nRanked results:")
        for i, (player, score) in enumerate(ranked_results.items()):
            print(f"{i + 1}. {player}: {score:.2f}")