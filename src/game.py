import copy
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict, Union
from math import ceil, log
from .strategies import Player
from .evo_strategy import EvoStrategy
from .utils import RandomNumberGenerator, ranked_scores_from_average_scores

PAYOFF_MATRIX = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

# Set global option for float formatting
pd.options.display.float_format = '{:.2f}'.format


class Match:
    """
    Class to simulate a match between two players with optional noise and 
    a probability of premature termination.

    The match consists of a series of rounds where each player chooses a move.
    The moves are potentially influenced by noise and the match length is 
    governed by either a fixed number of turns or a geometric distribution 
    (based on the probability of ending).
    """

    def __init__(self, players: List[Player], turns: int = 1, noise: float = 0, prob_end: float = 0, seed: Union[int, None] = None):
        """
        Initialize the match between two players.

        :param players: List of exactly two Player instances participating in the match.
        :param turns: Maximum number of turns per match (default: 1).
        :param noise: Probability of introducing noise in actions (default: 0).
        :param prob_end: Probability of ending the match after each round (default: 0).
        :param seed: Seed for reproducibility. If None, a random seed is used.
        :raises ValueError: If the number of players is not exactly 2.
        """
        if len(players) != 2:
            raise ValueError("The 'players' parameter must contain exactly two Player instances.")
        
        self.players = players  
        self.turns = turns  
        self.noise = noise  
        self.prob_end = prob_end  
        self.moves: List[Tuple[str, str]] = []  
        self.winner: Union[str, None] = None  
        self.final_scores: Tuple[int, int] = (0, 0)

        # Set the seed for reproducibility.
        self.set_seed(seed)

    def set_seed(self, seed: Union[int, None] = None) -> None:
        """
        Sets the random seed for the match.

        :param seed: Seed value for reproducibility. If None, a random seed is generated.
        """
        if seed is None:
            self._seed = np.random.randint(0, 2**32 - 1)
        else:
            self._seed = seed
        self._rng = RandomNumberGenerator(seed=self._seed)

    def _flip_action(self, action: str) -> str:
        """
        Flips the action with a probability defined by the noise parameter.

        :param action: The original action ('C' for Cooperate, 'D' for Defect).
        :return: The potentially flipped action.
        """
        if self._rng.random() < self.noise:
            return 'D' if action == 'C' else 'C'
        return action

    def _get_effective_turns(self) -> int:
        """
        Determines the effective number of turns for the match.

        If `prob_end` is specified, the match length is sampled from a geometric
        distribution. The resulting length is capped by the maximum number of turns.
        
        Here, the number of turns until the match ends is random variable.
        This variable follows a geometric distribution with parameter `prob_end`.
        The probability mass function (PMF) of the geometric distribution is given by:
        
        PMF(k) = (1 - prob_end)^(k - 1) * prob_end
        
        The expected value for the geometric distribution is given by:
        
        E[X] = 1 / prob_end
        
        A sample of the geometric distribution can be obtained by:
        
        X = ceil(log(1 - random_value) / log(1 - prob_end))
        
        where `random_value` is a random number in the interval [0, 1).
        
        The match length is the minimum between the sampled length and
        the maximum number of turns.

        :return: The effective number of turns for the match.
        """
        if self.prob_end is None or self.prob_end <= 0:
            # Default to the maximum number of turns if no premature ending is specified.
            return self.turns
        if self.prob_end >= 1:
            # If `prob_end` is 1, the match ends immediately after one turn.
            return 1
        
        # Sample the match length from a geometric distribution.
        random_value = self._rng.random()
        sample_length = int(ceil(log(1 - random_value) / log(1 - self.prob_end)))
        
        # Cap the match length by the maximum number of turns.
        effective_turns = min(self.turns, sample_length)
        return effective_turns

    def play(self) -> List[Tuple[str, str]]:
        """
        Simulates the match and returns the sequence of moves.

        Each player makes a move in every turn, potentially influenced by noise.
        After all turns, the final scores are computed, and the winner is determined.

        :return: A list of tuples representing the moves made in the match.
        """
        # Reset the players' states before starting the match.
        self.players[0].reset()
        self.players[1].reset()

        # Determine the effective number of turns based on `prob_end`.
        effective_turns = self._get_effective_turns()

        # Set unique seeds for stochastic players if applicable.
        for player in self.players:
            if player.stochastic:
                player.set_seed(self._rng.randint(0, 2**32 - 1))

        # Simulate each turn.
        for _ in range(effective_turns):
            # Get the moves for both players, potentially flipping them due to noise.
            move1 = self._flip_action(self.players[0].strategy(self.players[1]))
            move2 = self._flip_action(self.players[1].strategy(self.players[0]))
            
            # Record the moves.
            self.moves.append((move1, move2))
            
            # Retrieve the scores from the payoff matrix.
            score1, score2 = PAYOFF_MATRIX[(move1, move2)]
            
            # Update each player's state and score.
            self.players[0].update(move1, score1)
            self.players[1].update(move2, score2)

        # Store the final scores and determine the winner.
        self.final_scores = (self.players[0].score, self.players[1].score)
        self.winner = self._determine_winner()

        return self.moves

    def _determine_winner(self) -> str:
        """
        Determines the winner of the match based on the final scores.

        :return: The name of the winning player or 'Tie' if scores are equal.
        """
        if self.players[0].score > self.players[1].score:
            return self.players[0].name
        elif self.players[0].score < self.players[1].score:
            return self.players[1].name
        return 'Tie'

def play_match(player1: Player, player2: Player, turns: int, noise: float, prob_end: float, seed: Union[int, None] = None) -> Tuple[int, int]:
    """
    Plays a single match between two players and returns their final scores.

    :param player1: The first player (strategy).
    :param player2: The second player (strategy).
    :param turns: Maximum number of turns per match.
    :param noise: Probability of introducing noise in player actions.   
    :param prob_end: Probability of ending the match after each turn.
    :param seed: Seed for reproducibility.
    :return: A tuple containing the two players' final scores.
    """
    match = Match([player1, player2], turns, noise, prob_end, seed)
    match.play()
    return match.final_scores


def round_robin_tournament(players: List[Player], turns: int, noise: float = 0.0, prob_end: float = 0.0, seed: Union[int, None] = None, axelrod: bool = False) -> pd.DataFrame:

    # Shuffle players to randomize the match order.
    np.random.shuffle(players)

    # Dictionary to store raw scores from each match.
    scores = {player.name: [] for player in players}

    # Round-robin: each player vs. each other player (one match each).
    for i, player1 in enumerate(players):
        for player2 in players[i + 1:]:
            score1, score2 = play_match(player1, player2, turns, noise, prob_end, seed)
            scores[player1.name].append(score1)
            scores[player2.name].append(score2)

        # If Axelrod is True, each player also competes against itself.
        if axelrod:
            score1, _ = play_match(player1, copy.deepcopy(player1), turns, noise, prob_end, seed)
            scores[player1.name].append(score1)
    
    return scores


class Tournament:
    """Class to simulate a repeated round-robin tournament between multiple strategies.

    Each player in the tournament competes against every other player (and potentially
    against themselves in the case of Axelrod-style tournaments). The results are 
    ranked based on their accumulated scores over multiple repetitions of the round-robin process.
    """

    def __init__(self, players: List[Player], turns: int, repetitions: int = 1, noise: float = 0.0, prob_end: float = 0.0, seed: Union[int, None] = None):
        """
        Initialize the tournament.

        :param players: List of participating strategies (Player objects).
        :param turns: Maximum number of turns per match.
        :param repetitions: Number of repetitions for the round-robin tournament.
        :param noise: Probability of introducing noise in player actions.
        :param prob_end: Probability of a match ending prematurely after each turn.
        :param seed: Seed for random number generator (ensures reproducibility).
        """
        self.players = players  
        self.turns = turns  
        self.repetitions = repetitions  
        self.noise = noise  
        self.seed = seed  
        self.prob_end = prob_end  

        # Initialize scores dictionary to track scores for each player.
        self.scores: Dict[str, List[int]] = {player.name: [] for player in players}
        # DataFrame to store ranked results after tournament completion.
        self.ranked_results: pd.DataFrame = pd.DataFrame()

    def play(self, axelrod: bool = False) -> None:
        """
        Simulates the round-robin tournament with the specified number of repetitions.

        :param axelrod: If True, players compete against themselves as part of the tournament.
        """
        for _ in range(self.repetitions):
            matches_scores = round_robin_tournament(self.players, self.turns, self.noise, self.prob_end, self.seed, axelrod)
            
            # Aggregate scores for all players after one repetition.
            for player in self.players:
                self.scores[player.name].append(sum(matches_scores[player.name]))
        
        # Compute final average scores and rank players.
        total_players = len(self.players) + 1 * axelrod
        average_scores = {player: float(np.mean(scores) / total_players) for player, scores in self.scores.items()}
        
        self.ranked_results = ranked_scores_from_average_scores(average_scores)

    def get_ranked_results(self) -> pd.DataFrame:
        """
        Returns the ranked results of the tournament as a DataFrame.

        :return: DataFrame containing ranks, player names, and scores.
        :raises ValueError: If the tournament has not been played yet.
        """
        if self.ranked_results.empty:
            raise ValueError("The tournament has not been played yet.")
        return self.ranked_results

    def print_ranked_results(self) -> None:
        """
        Prints the ranked results of the tournament in tabular format.
        """
        print(self.get_ranked_results().to_string(index=False))

    def save_ranked_results(self, filename: str) -> None:
        """
        Saves the ranked results of the tournament to a CSV file.

        :param filename: Name of the file (without extension) to save the results.
        :raises ValueError: If filename is invalid or None.
        """
        if not filename:
            raise ValueError("The 'filename' parameter must be a valid file path.")
        os.makedirs("results", exist_ok=True)  # Ensure results directory exists.
        self.get_ranked_results().to_csv(f"results/{filename}.csv", index=False)


class EvolutionaryIterativePrisonersDilemma:
    """
    Simulates an evolutionary Iterated Prisoner's Dilemma (IPD) tournament
    that combines fixed (non-evolutionary) players and evolutionary players.

    The evolutionary players adapt over multiple generations. Each generation:
      - Runs a round-robin tournament (possibly Axelrod-style, i.e., 
        players may compete against themselves).
      - Ranks players by their average scores.
      - Selects the top proportion of evolutionary players and evolves them
        (via crossover and mutation).
      - Replaces the previous generation's evolutionary players with the 
        newly formed offspring.

    The process repeats for a specified number of generations, and a record of 
    each generation's top performers is maintained.
    """

    def __init__(
        self,
        fixed_players: List[Player],
        num_evo_players: int = 10,
        turns: int = 100,
        generations: int = 10,
        selected_proportion: float = 0.5,
        action_history_size: int = 10,
        mutation_rate: float = 0.1,
        noise: float = 0.0,
        prob_end: float = 0.0,
        seed: Union[int, None] = None
    ):
        """
        Initialize the evolutionary iterative tournament.

        :param fixed_players: List of fixed strategies (non-evolutionary).
        :param num_evo_players: Number of evolutionary players to include.
        :param turns: Maximum number of turns per match.
        :param generations: Number of generations of the evolutionary process.
        :param selected_proportion: Proportion of top evolutionary players that
                                    will be selected to produce the next generation.
        :param action_history_size: Size of the action-history window for evolutionary players.
        :param mutation_rate: Probability that a mutation occurs in an offspring strategy.
        :param noise: Probability of introducing noise in players' actions.
        :param prob_end: Probability of ending a match after each turn.
        :param seed: Seed for reproducibility (if None, a random seed is used).

        :raises ValueError: If the number of evolutionary players or the selected proportion
                           results in fewer than two survivors each generation.
        """
        if num_evo_players < 2 or num_evo_players * selected_proportion < 2:
            raise ValueError(
                "The number of evolutionary players and the selected proportion "
                "must ensure at least 2 survivors."
            )

        self.fixed_players = fixed_players
        self.num_evo_players = num_evo_players
        self.num_players = len(fixed_players) + num_evo_players
        self.num_evo_selected = max(2, int(num_evo_players * selected_proportion))
        self.turns = turns
        self.generations = generations
        self.noise = noise
        self.prob_end = prob_end
        self.seed = seed
        self.ranked_results: pd.DataFrame = pd.DataFrame()
        self.generations_results: pd.DataFrame = pd.DataFrame()
        self.mutation_rate = mutation_rate

        # Initialize evolutionary players, each with its own strategy parameters.
        self.evo_players = [
            EvoStrategy(action_history_size=action_history_size)
            for _ in range(num_evo_players)
        ]
        self._best_evo_player: Union[EvoStrategy, None] = None

    def _play_tournament(self, axelrod: bool = False) -> None:
        """
        Conducts a single round-robin tournament among all current players 
        (fixed and evolutionary).

        :param axelrod: If True, each player also plays against itself 
                        (Axelrod-style tournament).
        :return: None (updates `self.ranked_results` with new ranking).
        """
        players = self.fixed_players + self.evo_players

        scores = round_robin_tournament(players, self.turns, self.noise, self.prob_end, self.seed, axelrod)
        
        # Compute average scores per player, dividing total by the number of opponents faced.
        total_players = self.num_players + 1 * axelrod
        average_scores = {
            player.name: sum(scores[player.name]) / total_players for player in players
        }

        self.ranked_results = ranked_scores_from_average_scores(average_scores)

    def _evolve_population(self, crossover_strategy: str = "adaptive_weighted") -> None:
        """
        Evolves the current population of evolutionary players by selecting top performers
        and generating offspring via crossover and mutation.
        """
        # Filter out rows in `self.ranked_results` that belong to evolutionary players only.
        evo_scores = self.ranked_results[
            self.ranked_results["Player"].isin([p.name for p in self.evo_players])
        ]

        # Reorder self.evo_players according to the rank in evo_scores (best to worst).
        self.evo_players = [
            next(p for p in self.evo_players if p.name == player_name)
            for player_name in evo_scores["Player"]
        ]

        # Select the top-performing evolutionary players.
        selected_evo_players = self.evo_players[:self.num_evo_selected]

        # Generate offspring from the selected players.
        offspring = self._generate_offspring(selected_evo_players, crossover_strategy)

        # Update evolutionary players: top players + new offspring.
        self.evo_players = selected_evo_players + offspring

    def _generate_offspring(
        self, 
        selected_players: List[EvoStrategy], 
        crossover_strategy: str = "adaptive_weighted"
    ) -> List[EvoStrategy]:
        """
        Creates offspring to fill the evolutionary player pool back to `num_evo_players`.
        """
        offspring = []
        while len(offspring) < self.num_evo_players - self.num_evo_selected:
            parent1, parent2 = np.random.choice(selected_players, size=2, replace=False)
            child = parent1.crossover(parent2, crossover_strategy)
            child.mutate(self.mutation_rate)
            offspring.append(child)
        
        return offspring

    def train(self, axelrod: bool = False, crossover_strategy: str = "adaptive_weighted") -> EvoStrategy:
        """
        Runs the full evolutionary training for the specified number of generations.

        :param axelrod: If True, each player also competes against itself in each generation.
        :param crossover_strategy: Mechanism for combining two parent strategies.
        :return: The best evolutionary player after the final generation.
        """
        print(f"\nTraining {self.num_evo_players} evolutionary strategies...\n")

        # Prepare a list to accumulate generation-level results.
        results_data = []

        # Print a table header for intermediate results.
        print("-" * 90)
        print(
            f'{"Generation":<15}'
            f'{"EvoStrategy":<12}'
            f'{"(Rank/Players, Score)":<25}'
            f'{"Best Player":<25}'
            f'{"Best Score":<15}'
        )
        print("-" * 90)

        print_interval = max(1, self.generations // 10)
        for generation in range(self.generations):
            self._play_tournament(axelrod)
            self._evolve_population(crossover_strategy)

            # Extract the top evolutionary player in this generation.
            top_evo = self.ranked_results.loc[
                self.ranked_results["Player"].isin([p.name for p in self.evo_players])
            ].iloc[0]

            # Extract the top overall player in this generation.
            top_player = self.ranked_results.iloc[0]

            # Print/log intermediate results at defined intervals (or the final generation).
            if generation % print_interval == 0 or generation == self.generations - 1:
                results_data.append({
                    "Generation": generation + 1,
                    "Best Evo Player": top_evo["Player"],
                    "Evo Rank": int(top_evo["Rank"]),
                    "Evo Score": float(top_evo["Score"]),
                    "Best Player": top_player["Player"],
                    "Best Score": float(top_player["Score"])
                })
                print(
                    f'{f"[{generation + 1}/{self.generations}]":<15}'
                    f'{top_evo["Player"]:<12}'
                    f'{f"({top_evo['Rank']}/{self.num_players}, {top_evo['Score']:.2f})":<25}'
                    f'{top_player["Player"]:<25}'
                    f'{f"{top_player['Score']:.2f}":<15}'
                )

        print("-" * 90)
        print("\nEvolutionary training completed.\n")

        self.generations_results = pd.DataFrame(results_data)
        best_evo_name = self.ranked_results.loc[
            self.ranked_results["Player"].isin([p.name for p in self.evo_players])
        ].iloc[0]["Player"]
        self._best_evo_player = next(p for p in self.evo_players if p.name == best_evo_name)

        return self._best_evo_player
        
    def get_best_evo_player(self) -> EvoStrategy:
        """
        Returns the best evolutionary player found during training.

        :return: The top EvoStrategy instance at the end of the last generation.
        :raises ValueError: If `train()` has not been called yet (no best player identified).
        """
        if self._best_evo_player is None:
            raise ValueError("The evolutionary tournament has not been played yet.")
        return self._best_evo_player

    def get_generations_results(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing the results of each generation.

        The DataFrame columns include:
            - 'Generation': Generation index (1-based).
            - 'Best Evo Player': Name of top evolutionary player in that generation.
            - 'Evo Rank': Rank of the top evolutionary player among all players.
            - 'Evo Score': Score of the top evolutionary player.
            - 'Best Player': Name of the top overall player in that generation.
            - 'Best Score': Score of the top overall player.

        :return: Pandas DataFrame of size (#logged_generations x 6).
        :raises ValueError: If the training has not been run yet (DataFrame is empty).
        """
        if self.generations_results.empty:
            raise ValueError("The evolutionary tournament has not been played yet.")
        return self.generations_results

    def print_final_results(self) -> None:
        """
        Prints the final ranked results of the last generation's tournament 
        for all players (fixed and evolutionary).

        :raises ValueError: If the tournament has not been played yet (ranked_results empty).
        """
        if self.ranked_results.empty:
            raise ValueError("The evolutionary tournament has not been played yet.")
        print(self.ranked_results.to_string(index=False))        
        
