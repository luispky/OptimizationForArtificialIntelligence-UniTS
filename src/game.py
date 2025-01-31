import copy
import numpy as np
import pandas as pd
import pandas as pd
import os
from typing import List, Tuple, Dict, Union
from math import ceil, log
from .strategies import Player
from .evo_strategy import EvoStrategy
from .utils import RandomNumberGenerator

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
        self._rng = RandomNumberGenerator()
        self.set_seed(seed)

    def set_seed(self, seed: Union[int, None] = None) -> None:
        """
        Sets the random seed for the match.

        :param seed: Seed value for reproducibility. If None, a random seed is generated.
        """
        if seed is None:
            seed_ = np.random.randint(0, 2**32 - 1)
        else:
            seed_ = seed
        self._rng = RandomNumberGenerator(seed=seed_)

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
                self.set_seed()
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


def ranked_scores_from_average_scores(average_scores: dict[str, float]) -> pd.DataFrame:
    """
    Converts a dictionary of average scores to a DataFrame with ranks.
    
    :param average_scores: Dictionary with player names as keys and average scores as values.
    :return: DataFrame with ranks based on sorted scores.
    """
    
    ranked_results = pd.DataFrame(
        list(average_scores.items()), columns=["Player", "Score"]
    )
    ranked_results = ranked_results.sort_values(
        by="Score", ascending=False
    ).reset_index(drop=True)
    
    # Add a rank column based on sorted scores.
    ranked_results["Rank"] = ranked_results.index + 1
    ranked_results = ranked_results[["Rank", "Player", "Score"]]
    
    return ranked_results


def round_robin_tournament(players: List[Player], turns: int, noise: float = 0.0, prob_end: float = 0.0, seed: Union[int, None] = None, axelrod: bool = False) -> Dict[str, List[int]]:
    """
    Simulates a round-robin tournament between multiple players.

    Each player competes against every other player (and potentially against themselves
    in the case of Axelrod-style tournaments). The results are stored as raw scores.

    :param players: List of participating strategies (Player objects).
    :param turns: Maximum number of turns per match.
    :param noise: Probability of introducing noise in player actions.
    :param prob_end: Probability of ending a match prematurely after each turn.
    :param seed: Seed for random number generator (ensures reproducibility).
    :param axelrod: If True, players also compete against themselves.
    :return: Dictionary containing the raw scores for each player in each match.
    """
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
        total_players = len(self.players) - 1 * (not axelrod)
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


class GAIterativePrisonersDilemma:
    """
    Simulates an evolutionary Iterated Prisoner's Dilemma (IPD) tournament with both fixed
    (non-evolutionary) and evolutionary players. Over multiple generations, evolutionary
    players adapt by competing in round-robin tournaments, ranking by average score, and
    undergoing selection, crossover, and mutation to form the next generation.
    """

    def __init__(
        self,
        fixed_players: List[Player],
        num_evo_players: int = 10,
        noise: float = 0.0,
        prob_end: float = 0.0,
        turns: int = 100,
        generations: int = 10,
        elitism_proportion: float = 0.1,
        action_history_size: int = 10,
        mutation_rate: float = 0.1,
        crossover_probability: float = 0.95,
        mutation_probability: float = 0.1,
        seed: Union[int, None] = 42
    ):
        """
        Initializes the evolutionary IPD tournament.

        :param fixed_players: List of non-evolutionary (fixed) strategies.
        :param num_evo_players: Number of evolutionary players to include.
        :param noise: Probability of introducing noise in players' actions.
        :param prob_end: Probability of ending a match after each turn.
        :param turns: Maximum number of turns per match.
        :param generations: Number of generations for the evolutionary process.
        :param elitism_proportion: Fraction of top evolutionary players retained each generation.
        :param action_history_size: Max history size for evolutionary strategies' memory.
        :param mutation_rate: Standard deviation for Gaussian mutation noise.
        :param crossover_probability: Probability of performing crossover between parents.
        :param mutation_probability: Probability of mutating an offspring strategy.
        :param seed: Random seed for reproducibility (if None, random seed is used).
        :raises ValueError: If num_evo_players < 2 or elitism would produce fewer than two survivors.
        """
        if num_evo_players < 2:
            raise ValueError("The number of evolutionary players must be at least 2.")

        self.fixed_players = fixed_players
        self.num_evo_players = num_evo_players
        self.num_players = len(fixed_players) + num_evo_players
        self.num_elite_players = max(1, int(elitism_proportion * num_evo_players))
        self.turns = turns
        self.generations = generations
        self.noise = noise
        self.prob_end = prob_end
        self.seed = seed
        self.ranked_results: pd.DataFrame = pd.DataFrame()
        self.generations_results: pd.DataFrame = pd.DataFrame()

        # Crossover and mutation parameters
        self.xover_prob = crossover_probability
        self.mutation_probability = mutation_probability
        self.mutation_rate = mutation_rate
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(self.seed)

        # Initialize evolutionary players with random action_history_size
        self.evo_players: List[EvoStrategy] = [
            EvoStrategy(action_history_size=self.rng.integers(max(1, action_history_size // 2), action_history_size, dtype=int))
            for _ in range(num_evo_players)
        ]
        self._best_evo_player: Union[EvoStrategy, None] = None
        

    def _play_tournament(self, players: List[Player], axelrod: bool = False) -> pd.DataFrame:
        """
        Conducts a single round-robin tournament among the specified players (fixed + evolutionary).

        :param players: List of all players participating in the tournament.
        :param axelrod: If True, each player also competes against itself (Axelrod-style).
        :return: DataFrame with each player's average score, sorted by rank.
        """
        scores = round_robin_tournament(players, self.turns, self.noise, self.prob_end, self.seed, axelrod)
        total_opponents = len(players) - 1 * (not axelrod)  # Adjust if self-play is disabled
        average_scores = {
            player.name: sum(scores[player.name]) / total_opponents for player in players
        }
        return ranked_scores_from_average_scores(average_scores)


    def _extract_sorted_evo_scores(self, df_ranked: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and sorts evolutionary players' scores from a ranked results DataFrame.

        :param df_ranked: DataFrame containing ranked results of a tournament (all players).
        :return: DataFrame sorted by rank but filtered to only include evolutionary players.
        """
        evo_player_names = [p.name for p in self.evo_players]
        evo_scores = df_ranked[df_ranked["Player"].isin(evo_player_names)]
        return evo_scores.sort_values(by="Rank", ascending=True)
    
    
    def _evolve_population(
        self,
        sorted_evo_scores: pd.DataFrame,
        crossover_strategy: str = "adaptive_weighted"
    ) -> None:
        """
        Selects top evolutionary players (elitism), then generates offspring to replenish
        the evolutionary player pool.

        :param sorted_evo_scores: Ranked scores specific to evolutionary players.
        :param crossover_strategy: Crossover approach (e.g., 'adaptive_weighted', 'BLX-α').
        """
        # Reorder current evolutionary players by rank (best -> worst)
        sorted_names = sorted_evo_scores["Player"].tolist()
        name_to_player = {p.name: p for p in self.evo_players}
        self.evo_players = [name_to_player[n] for n in sorted_names]

        # Elitism: keep top performers
        elites = self.evo_players[:self.num_elite_players]

        # Generate offspring from the entire (sorted) evo player set
        offspring = self._generate_offspring(df_ranked=sorted_evo_scores, strategy=crossover_strategy)

        # Combine elites and newly produced offspring
        self.evo_players = elites + offspring


    def _generate_offspring(
        self,
        df_ranked: pd.DataFrame,
        crossover_strategy: str = "adaptive_weighted"
    ) -> List[EvoStrategy]:
        """
        Fills the evolutionary player pool up to num_evo_players by creating offspring.

        :param df_ranked: Overall ranked results used to get selection probabilities.
        :param crossover_strategy: How parents' genetic information is combined.
        :return: A list of newly created EvoStrategy offspring.
        """
        offspring = []
        required_offspring = self.num_evo_players - self.num_elite_players

        # Map names to evolutionary player instances and prepare selection probabilities
        player_dict = {p.name: p for p in self.evo_players}
        probabilities = df_ranked["Score"].values / df_ranked["Score"].sum()
        ordered_players = [player_dict[name] for name in df_ranked["Player"].values]

        while len(offspring) < required_offspring:
            # Select two parents based on their scores
            parent1, parent2 = self.rng.choice(
                ordered_players, size=2, replace=False, p=probabilities
            )

            # Crossover with probability xover_prob, otherwise copy parents
            if self.rng.random() < self.xover_prob:
                offspring_pair = parent1.crossover(parent2, strategy=crossover_strategy)
            else:
                offspring_pair = (parent1, parent2)

            for child in offspring_pair:
                # Mutate with probability mutation_probability
                if self.rng.random() < self.mutation_probability:
                    child.mutate(self.mutation_rate)
                offspring.append(child)
                if len(offspring) >= required_offspring:
                    break

        return offspring


    def train(
        self,
        axelrod: bool = False,
        crossover_strategy: str = "adaptive_weighted"
    ) -> EvoStrategy:
        """
        Executes the evolutionary training process for a specified number of generations.
        Each generation:
          - Runs a round-robin tournament (optionally Axelrod-style).
          - Ranks all players by average score.
          - Records top-performing evolutionary and overall players.
          - Evolves the evolutionary players (selection, crossover, mutation).

        :param axelrod: If True, each player also plays against itself.
        :param crossover_strategy: Strategy for combining parent weights (e.g., 'adaptive_weighted').
        :return: Reference to the best evolutionary player after final generation.
        """
        print(f"\nTraining {self.num_evo_players} evolutionary strategies...\n")
        results_data = []

        # Display table header for intermediate results
        print("-" * 90)
        print(
            f'{"Generation":<15}'
            f'{"Best Evo Player":<20}'
            f'{"Evo Score":<15}'
            f'{"Best Player":<20}'
            f'{"Best Score":<15}'
        )
        print("-" * 90)

        print_interval = max(1, self.generations // 10)
        for generation in range(1, self.generations + 1):
            # Play tournament among fixed + current evolutionary players
            players = self.fixed_players + self.evo_players
            self.ranked_results = self._play_tournament(players, axelrod)

            # Identify top evolutionary and top overall players
            top_evo = self.ranked_results.loc[
                self.ranked_results["Player"].isin([p.name for p in self.evo_players])
            ].iloc[0]
            top_player = self.ranked_results.iloc[0]

            # Log intermediate results at intervals
            if (generation == 1 or
                generation % print_interval == 0 or
                generation == self.generations
                ):
                results_data.append({
                    "Generation": generation,
                    "Best Evo Player": top_evo["Player"],
                    "Evo Rank": int(top_evo["Rank"]),
                    "Evo Score": float(top_evo["Score"]),
                    "Best Player": top_player["Player"],
                    "Best Score": float(top_player["Score"])
                })
                print(
                    f"[{generation}/{self.generations}]".ljust(15) +
                    f"{top_evo['Player']:<12}" +
                    f"({top_evo['Rank']}/{self.num_players}, {top_evo['Score']:.2f})".ljust(25) +
                    f"{top_player['Player']:<25}" +
                    f"{top_player['Score']:.2f}".ljust(15)
                )

            # Evolve the population unless we're at the final generation
            if generation < self.generations:
                sorted_evo_scores = self._extract_sorted_evo_scores(self.ranked_results)
                self._evolve_population(sorted_evo_scores, crossover_strategy)

        print("-" * 90)
        print("\nEvolutionary training completed.\n")

        # Store the generation-by-generation results
        self.generations_results = pd.DataFrame(results_data)

        # Identify the best evolutionary player overall
        best_evo_name = self.ranked_results.loc[
            self.ranked_results["Player"].isin([p.name for p in self.evo_players])
        ].iloc[0]["Player"]
        self._best_evo_player = next(p for p in self.evo_players if p.name == best_evo_name)

        return self._best_evo_player
    

    def get_best_evo_player(self) -> EvoStrategy:
        """
        Returns the best evolutionary player found at the end of training.

        :return: The best EvoStrategy instance.
        :raises ValueError: If training has not been run yet (no best player).
        """
        if self._best_evo_player is None:
            raise ValueError("No best evolutionary player found (train() not yet called).")
        return self._best_evo_player


    def get_generations_results(self) -> pd.DataFrame:
        """
        Retrieves a summary of the top performers per generation.

        Columns in the resulting DataFrame:
            Generation, Best Evo Player, Evo Rank, Evo Score, Best Player, Best Score

        :return: DataFrame of results (#logged_generations x 6).
        :raises ValueError: If training has not been run yet (no results).
        """
        if self.generations_results.empty:
            raise ValueError("No results available (train() not yet called).")
        return self.generations_results


    def print_final_results(self) -> None:
        """
        Prints the final ranked results of the last tournament.

        :raises ValueError: If no tournament has been played.
        """
        if self.ranked_results.empty:
            raise ValueError("No tournament data available (train() not yet called).")
        print(self.ranked_results.to_string(index=False))


class CoevolutionaryIterativePrisonersDilemma:
    """
    Simulates an evolutionary Iterated Prisoner's Dilemma (IPD) tournament with both fixed
    (non-evolutionary) and evolutionary players.

    Over multiple generations, evolutionary players:
      - Participate in a round-robin tournament (optionally Axelrod-style),
      - Are ranked by their average scores,
      - Undergo selection (with elitism), crossover, and mutation,
      - Replace the previous evolutionary population with newly formed offspring.

    A record of top performers is maintained. Additionally, an "absolute fitness" evaluation
    may be conducted at specified intervals (e.g., every Nth generation).
    """

    def __init__(
        self,
        fixed_players: List[Player],
        num_evo_players: int = 10,
        absolute_fitness_eval_interval: int = 10,
        noise: float = 0.0,
        prob_end: float = 0.0,
        turns: int = 200,
        generations: int = 10,
        elitism_proportion: float = 0.1,
        action_history_size: int = 10,
        mutation_rate: float = 0.1,
        mutation_probability: float = 0.8,
        crossover_probability: float = 0.95,
        seed: Union[int, None] = 42
    ):
        """
        Initializes the evolutionary IPD tournament.

        :param fixed_players: List of non-evolutionary (fixed) strategies.
        :param num_evo_players: Number of evolutionary players to include (>= 2).
        :param absolute_fitness_eval_interval: Interval (in generations) at which to perform
                                              "absolute fitness" evaluation.
        :param noise: Probability of introducing noise in players' actions.
        :param prob_end: Probability of ending a match after each turn.
        :param turns: Maximum number of turns per match.
        :param generations: Total number of generations in the evolutionary process.
        :param elitism_proportion: Fraction of top evolutionary players retained each generation.
        :param action_history_size: Memory size for the evolutionary strategies (history window).
        :param mutation_rate: Standard deviation for Gaussian mutation noise.
        :param mutation_probability: Probability that mutation is applied to an offspring.
        :param crossover_probability: Probability of performing crossover between parents.
        :param seed: Seed for random number generation (None => random seed).
        :raises ValueError: If num_evo_players < 2, or if elitism_proportion yields < 1 elite.
        """
        if num_evo_players < 2:
            raise ValueError("The number of evolutionary players must be at least 2.")

        self.fixed_players = fixed_players
        self.num_evo_players = num_evo_players
        self.num_elite_players = max(1, int(elitism_proportion * num_evo_players))
        self.turns = turns
        self.generations = generations
        self.absolute_fitness_eval_interval = absolute_fitness_eval_interval
        self.noise = noise
        self.prob_end = prob_end
        self.seed = seed
        self.ranked_results: pd.DataFrame = pd.DataFrame()
        self.generations_results: List[Dict] = []

        # Crossover and mutation parameters
        self.xover_prob = crossover_probability
        self.mutation_probability = mutation_probability
        self.mutation_rate = mutation_rate
        
        # Random number generator for reproducibility
        self.rng = np.random.default_rng(self.seed)

        # Initialize evolutionary players with random action_history_size
        self.evo_players: List[EvoStrategy] = [
            EvoStrategy(action_history_size=self.rng.integers(max(1, action_history_size // 2), action_history_size, dtype=int))
            for _ in range(num_evo_players)
        ]
        self._best_evo_player: Union[EvoStrategy, None] = None


    def _play_tournament(self, players: List[Player], axelrod: bool = False) -> pd.DataFrame:
        """
        Conducts a single round-robin tournament among the specified players (fixed + evolutionary).

        :param players: List of all players participating in the tournament.
        :param axelrod: If True, each player also competes against itself (Axelrod-style).
        :return: DataFrame with each player's average score, sorted by rank.
        """
        scores = round_robin_tournament(players, self.turns, self.noise, self.prob_end, self.seed, axelrod)
        total_opponents = len(players) - 1 * (not axelrod)  # Adjust if self-play is disabled
        average_scores = {
            player.name: sum(scores[player.name]) / total_opponents for player in players
        }
        return ranked_scores_from_average_scores(average_scores)

    def _evolve_population(
        self,
        sorted_evo_scores: pd.DataFrame,
        crossover_strategy: str = "adaptive_weighted"
    ) -> None:
        """
        Selects top evolutionary players (elitism), then uses them to produce offspring
        via crossover and mutation, restoring the population size.

        :param sorted_evo_scores: Ranked scores of evolutionary players (best -> worst).
        :param crossover_strategy: Strategy used for combining parent weights
                                  (e.g., 'adaptive_weighted', 'BLX-α').
        """
        # Reorder self.evo_players according to the sorted evo_scores (best -> worst).
        sorted_names = sorted_evo_scores["Player"].tolist()
        name_to_player = {p.name: p for p in self.evo_players}
        self.evo_players = [name_to_player[n] for n in sorted_names]

        # Select top-performing evolutionary players (elitism)
        elite_evo_players = self.evo_players[:self.num_elite_players]

        # Create the rest via offspring
        offspring = self._generate_offspring(sorted_evo_scores, crossover_strategy)

        # Replace old evolutionary population
        self.evo_players = elite_evo_players + offspring

    def _generate_offspring(
        self,
        df_ranked: pd.DataFrame,
        crossover_strategy: str = "adaptive_weighted"
    ) -> List[EvoStrategy]:
        """
        Produces enough offspring to fill the evolutionary population up to 'num_evo_players'.

        :param df_ranked: Ranked scores for evolutionary players.
        :param crossover_strategy: Crossover approach (e.g., 'adaptive_weighted').
        :return: List of newly created EvoStrategy offspring.
        """
        offspring = []
        required_offspring = self.num_evo_players - self.num_elite_players

        # Map player name -> EvoStrategy instance
        player_dict = {p.name: p for p in self.evo_players}
        probabilities = df_ranked["Score"].values / df_ranked["Score"].sum()
        ordered_players = [player_dict[name] for name in df_ranked["Player"].values]

        while len(offspring) < required_offspring:
            # Select two parents based on their scores
            parent1, parent2 = self.rng.choice(
                ordered_players, size=2, replace=False, p=probabilities
            )

            # Crossover with probability xover_prob, otherwise copy parents
            if self.rng.random() < self.xover_prob:
                offspring_pair = parent1.crossover(parent2, strategy=crossover_strategy)
            else:
                offspring_pair = (parent1, parent2)

            for child in offspring_pair:
                # Mutate with probability mutation_probability
                if self.rng.random() < self.mutation_probability:
                    child.mutate(self.mutation_rate)
                offspring.append(child)
                if len(offspring) >= required_offspring:
                    break

        return offspring


    def _evaluate_absolute_fitness(self, axelrod: bool, generation: int) -> pd.DataFrame:
        """
        Performs an 'absolute fitness' evaluation of each evolutionary player against
        all fixed players individually, logs best performers, and returns sorted scores.

        :param axelrod: If True, each player also plays against itself.
        :param generation: Current generation number (used for logging).
        :return: DataFrame sorted by Score, containing the evolutionary players' results.
        """
        evo_players_scores = []

        for evo_player in self.evo_players:
            # Evaluate evo_player against all fixed players only
            players = self.fixed_players + [evo_player]
            df_ranked = self._play_tournament(players, axelrod=axelrod)
            evo_player_info_tmp = df_ranked[df_ranked["Player"] == evo_player.name].iloc[0]
            evo_players_scores.append(evo_player_info_tmp)

        # Convert evolutionary players' results into a DataFrame, then sort
        evo_scores_df = pd.DataFrame(evo_players_scores).sort_values(by="Score", ascending=False)
        best_evo_name = evo_scores_df.iloc[0]["Player"]
        best_evo_player = next(p for p in self.evo_players if p.name == best_evo_name)
        
        players = self.fixed_players + [best_evo_player]
        df_ranked = self._play_tournament(players, axelrod=axelrod)
        
        top_player_info = df_ranked.iloc[0]
        top_evo_info = df_ranked[df_ranked["Player"] == best_evo_name].iloc[0]

        # Print intermediate results (best evolutionary player + best player overall)
        print(
            f'[{generation}/{self.generations}]'.ljust(15)
            + f'{top_evo_info["Player"]:<12}'
            + f'({top_evo_info["Rank"]}/{len(players)}, {top_evo_info["Score"]:.2f})'.ljust(25)
            + f'{top_player_info["Player"]:<25}'
            + f'{top_player_info["Score"]:.2f}'.ljust(15)
        )

        # Log results for the generation
        self.generations_results.append({
            "Generation": generation,
            "Best Evo Player": top_evo_info["Player"],
            "Evo Rank": int(top_evo_info["Rank"]),
            "Evo Score": float(top_evo_info["Score"]),
            "Best Player": top_player_info["Player"],
            "Best Score": float(top_player_info["Score"])
        })

        # Remove 'Rank' before returning, as we only need 'Score' to sort in subsequent steps
        sorted_evo_scores = evo_scores_df.drop(columns=["Rank"], errors="ignore")
        return sorted_evo_scores


    def train(self, axelrod: bool = False, crossover_strategy: str = "adaptive_weighted") -> EvoStrategy:
        """
        Conducts the evolutionary training over the specified number of generations.
        Depending on 'absolute_fitness_eval_interval', some generations perform an 'absolute'
        fitness evaluation, while others run a standard round-robin among evolutionary players only.

        :param axelrod: If True, each player also competes against itself in each generation.
        :param crossover_strategy: Strategy used to combine parent strategies (e.g., 'adaptive_weighted').
        :return: The best evolutionary player after all generations.
        """
        print(f"\nTraining {self.num_evo_players} evolutionary strategies...\n")
        print("-" * 90)
        print(
            f'{"Generation":<15}'
            f'{"Best Evo Player":<20}'
            f'{"Evo Score":<15}'
            f'{"Best Player":<20}'
            f'{"Best Score":<15}'
        )
        print("-" * 90)

        for generation in range(1, self.generations + 1):
            # Determine which type of evaluation to run
            # 1) Absolute: if at intervals or final generation
            # 2) Evolutionary: standard round-robin among evo_players
            if (
                generation == 1
                or generation % self.absolute_fitness_eval_interval == 0
                or generation == self.generations
            ):
                # Evaluate each evolutionary player's performance against fixed players
                df_ranked = self._evaluate_absolute_fitness(axelrod, generation)
            else:
                # Standard evolutionary tournament among evolutionary players only
                df_ranked = self._play_tournament(self.evo_players, axelrod)

            if generation < self.generations:
                self._evolve_population(df_ranked, crossover_strategy)

        # Final evaluation to identify the best evolutionary player
        self.ranked_results = self._evaluate_absolute_fitness(axelrod, self.generations + 1)

        print("-" * 90)
        print("\nEvolutionary training completed.\n")

        best_evo_name = self.ranked_results.iloc[0]["Player"]
        self._best_evo_player = next(p for p in self.evo_players if p.name == best_evo_name)
        return self._best_evo_player


    def get_best_evo_player(self) -> EvoStrategy:
        """
        Retrieves the best evolutionary player determined at the end of training.

        :return: Reference to the top EvoStrategy.
        :raises ValueError: If training hasn't been run and no best player is identified.
        """
        if self._best_evo_player is None:
            raise ValueError("No best evolutionary player found. Run 'train()' first.")
        return self._best_evo_player


    def get_generations_results(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing the stored results for each generation.

        Columns typically include:
          'Generation', 'Best Evo Player', 'Evo Rank', 'Evo Score', 'Best Player', 'Best Score'.

        :return: Pandas DataFrame with the logged generation results.
        :raises ValueError: If no results are available (training not done).
        """
        if not self.generations_results:
            raise ValueError("No results available. You must run 'train()' first.")
        return pd.DataFrame(self.generations_results)


    def print_final_results(self) -> None:
        """
        Prints the final ranked results from the most recent tournament evaluation.

        :raises ValueError: If the tournament has never been played.
        """
        if self.ranked_results.empty:
            raise ValueError("No tournament data available. Run 'train()' first.")
        print(self.ranked_results.to_string(index=False))
