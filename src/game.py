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
    Simulates an evolutionary Iterated Prisoner's Dilemma (IPD) tournament
    with evolutionary players adapting solely through mutation and the
    one-fifth rule for dynamic mutation rate adjustment.
    """

    def __init__(
        self,
        fixed_players: List[Player],
        num_evo_players: int = 10,
        turns: int = 100,
        generations: int = 10,
        action_history_size: int = 2,
        mutation_rate: float = 0.1,
        noise: float = 0.0,
        prob_end: float = 0.0,
        seed: Union[int, None] = None
    ):
        """
        Initializes the evolutionary iterative tournament.

        :param fixed_players: List of fixed strategy players.
        :param num_evo_players: Number of evolutionary players.
        :param turns: Number of turns per match.
        :param generations: Number of generations.
        :param action_history_size: Action-history size for evolutionary players.
        :param mutation_rate: Initial mutation rate.
        :param noise: Probability of noise in actions.
        :param prob_end: Probability of ending a match early.
        :param seed: Random seed.
        """
        if num_evo_players < 2:
            raise ValueError("The number of evolutionary players must be at least 2.")

        self.fixed_players = fixed_players
        self.num_evo_players = num_evo_players
        self.num_players = len(fixed_players) + num_evo_players
        self.turns = turns
        self.generations = generations
        self.noise = noise
        self.prob_end = prob_end
        self.seed = seed
        self.mutation_rate = mutation_rate

        # DataFrames for logging
        self.ranked_results: pd.DataFrame = pd.DataFrame()
        self.generations_results: pd.DataFrame = pd.DataFrame()

        # Initialize evolutionary players
        # Each EvoStrategy must have a unique name if you rely on 'name' for tracking
        self.evo_players = [
            EvoStrategy(action_history_size=action_history_size)
            for _ in range(num_evo_players)
        ]
        self._best_evo_player: Union[EvoStrategy, None] = None

    def _play_tournament(
        self, players: List[Player], axelrod: bool = False
    ) -> pd.DataFrame:
        """
        Conducts a single round-robin tournament among fixed + evo players.

        :param axelrod: If True, players also compete against themselves.
        :return: A DataFrame with columns ["Player", "Score", "Rank"], 
                 sorted by descending Score.
        """
        
        # Conduct the round-robin. This function should return
        # something like a dict: {player: [scores_against_each_opponent]} 
        scores_dict = round_robin_tournament(
            players, self.turns, self.noise, self.prob_end, self.seed, axelrod
        )

        # Example aggregator for average scores. This depends on how round_robin_tournament
        # organizes the returned data. Adjust to your actual aggregator.
        total_opponents = len(players) - 1 * (not axelrod)
        
        average_scores = {}
        for p in players:
            p_scores = scores_dict[p.name]  # or however you stored them
            average_scores[p.name] = sum(p_scores) / total_opponents

        # Convert average scores to a ranking DataFrame
        df_ranked = ranked_scores_from_average_scores(average_scores)
        # df_ranked has columns: ["Player", "Score", "Rank"]
        
        return df_ranked

    def _apply_one_fifth_rule(self, success_rate: float) -> None:
        """
        Adjusts the mutation rate based on the success rate using the one-fifth rule.
        """
        if success_rate > 0.2:
            self.mutation_rate *= 1.05
        elif success_rate < 0.2:
            self.mutation_rate *= 0.95

    def _mutate_population(self) -> List["EvoStrategy"]:
        """
        Creates offspring by mutating each evolutionary player exactly once.

        :return: List of mutated EvoStrategy instances, same length as self.evo_players.
        """
        offspring = []
        for parent in self.evo_players:
            child = copy.deepcopy(parent)
            # Use the current mutation rate
            child.mutate(self.mutation_rate)
            offspring.append(child)
        return offspring
    
    def train(self, axelrod: bool = False, one_fifth_interval: int = 10) -> "EvoStrategy":
        """
        Runs the evolutionary training for the specified number of generations.

        :param axelrod: If True, players also compete against themselves.
        :param one_fifth_interval: Interval for applying the one-fifth success rule.
        :return: The best evolutionary player after the final generation.
        """
        print(f"\nTraining {self.num_evo_players} evolutionary strategies...\n")

        results_data = []

        header_line = (
            f'{"Generation":<15}'
            f'{"Best Evo":<20}'
            f'{"(Rank, Score)":<20}'
            f'{"Overall Best":<20}'
            f'{"Score":<10}'
        )
        print("-" * len(header_line))
        print(header_line)
        print("-" * len(header_line))

        # --- Generation 0: Evaluate the initial population ---
        ranked_results_parents = self._play_tournament(self.fixed_players + self.evo_players, axelrod)
        self.ranked_results = ranked_results_parents
        
        # Identify best among parents
        evo_ranks = ranked_results_parents[
            ranked_results_parents["Player"].isin([p.name for p in self.evo_players])
        ].sort_values(by="Score", ascending=False, ignore_index=True)
        
        best_evo = evo_ranks.iloc[0]
        overall_best = ranked_results_parents.iloc[0]
        
        print(
            f'{0:<10}'
            f'{best_evo["Player"]:<20}'
            f'({int(best_evo["Rank"])}, {best_evo["Score"]:.2f})    '
            f'{overall_best["Player"]:<20}'
            f'{overall_best["Score"]:.2f}'
        )

        successes = 0
        print_interval = max(1, self.generations // 5)

        for generation in range(1, self.generations + 1):
            print(f"\nGeneration {generation}:")
            # 1) Create offspring
            offspring = self._mutate_population()

            # 2) Evaluate parents + offspring in the same tournament (plus selection).
            #    Or you could do two separate calls, but typically we want them all
            #    together to get consistent scores for selection.
            union = self.evo_players + offspring
            print(f"Playing tournament with {len(union)} players...")
            ranked_results_union = self._play_tournament(union, axelrod)

            # 3) Track how many offspring outperformed their parents
            #    Because this is 1-to-1, we can zip them in order
            parent_scores = {
                p.name: ranked_results_union.loc[
                    ranked_results_union["Player"] == p.name, "Score"
                ].values[0]
                for p in self.evo_players
            }
            offspring_scores = {
                c.name: ranked_results_union.loc[
                    ranked_results_union["Player"] == c.name, "Score"
                ].values[0]
                for c in offspring
            }

            gen_successes = 0
            for p, c in zip(self.evo_players, offspring):
                if offspring_scores[c.name] > parent_scores[p.name]:
                    gen_successes += 1
            
            successes += gen_successes

            # 4) Select the top num_evo_players from the union
            #    Sort by Score desc, keep first num_evo_players that are Evo players
            #    (since we only care about evolutionary players for selection)
            union_df = ranked_results_union[
                ranked_results_union["Player"].isin([x.name for x in union])
            ].copy()
            union_df.sort_values("Score", ascending=False, inplace=True, ignore_index=True)
            
            # Just pick the top self.num_evo_players
            selected_evo_names = union_df.head(self.num_evo_players)["Player"].values
            # Update self.evo_players
            # We rely on matching by name
            new_pop = []
            for ev in union:
                if ev.name in selected_evo_names:
                    new_pop.append(ev)
            self.evo_players = new_pop

            # 5) Possibly adjust mutation rate using 1/5 rule
            if generation % one_fifth_interval == 0:
                # success_rate = #successes / (N * interval)
                success_rate = successes / (self.num_evo_players * one_fifth_interval)
                self._apply_one_fifth_rule(success_rate)
                successes = 0  # reset

            # 6) Track best evolutionary player of the new population
            evo_ranks = union_df[
                union_df["Player"].isin([p.name for p in self.evo_players])
            ].head(self.num_evo_players)
            evo_ranks.sort_values("Score", ascending=False, inplace=True, ignore_index=True)
            top_evo = evo_ranks.iloc[0]
            
            # 7) Track overall best (including fixed players if you want)
            #    The union included only the old parents + new children,
            #    so if you want to see how they stack up against fixed players,
            #    do a separate full eval or keep the same approach:
            print(f"Playing full tournament with {len(self.fixed_players) + len(self.evo_players)} players...")
            full_ranked = self._play_tournament(self.fixed_players + self.evo_players, axelrod)
            best_player = full_ranked.iloc[0]  # overall best
            self.ranked_results = full_ranked

            # 8) Print / log results
            if (generation % print_interval == 0) or (generation == self.generations):
                print(
                    f'{generation:<10}'
                    f'{top_evo["Player"]:<20}'
                    f'({int(top_evo["Rank"])}, {top_evo["Score"]:.2f})    '
                    f'{best_player["Player"]:<20}'
                    f'{best_player["Score"]:.2f}'
                )
            
            results_data.append({
                "Generation": generation,
                "Best Evo Player": top_evo["Player"],
                "Evo Rank": int(top_evo["Rank"]),
                "Evo Score": float(top_evo["Score"]),
                "Best Player": best_player["Player"],
                "Best Score": float(best_player["Score"]),
                "Mutation Rate": self.mutation_rate
            })

        print("-" * len(header_line))
        print("\nEvolutionary training completed.\n")

        # Store results
        self.generations_results = pd.DataFrame(results_data)

        # Return the best evolutionary strategy from the final round
        # Re-evaluate the final pop vs. fixed players to get final ranks
        final_ranked = self._play_tournament(self.fixed_players + self.evo_players, axelrod)
        final_evo_ranks = final_ranked[
            final_ranked["Player"].isin([p.name for p in self.evo_players])
        ].copy()
        final_evo_ranks.sort_values("Score", ascending=False, inplace=True, ignore_index=True)
        best_evo_name = final_evo_ranks.iloc[0]["Player"]

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
        
