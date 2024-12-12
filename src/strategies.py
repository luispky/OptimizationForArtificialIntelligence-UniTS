import numpy as np
from typing import List, Tuple, Dict, Union
from scipy.stats import chisquare
import random

def random_choice(p: float = 0.5) -> str:
    if p == 0:
            return 'D'  
    elif p == 1:
        return 'C'  
    return 'C' if random.random() < p else 'D'

class Strategy:
    def __init__(self, name: str):
        self.name = name
        self.score = 0
        self.history = []

    def reset(self):
        self.history = []
        self.score = 0

    def play(self, opponent):
        raise NotImplementedError

class TitFortat(Strategy):
    def __init__(self) -> None:
        super().__init__("TitForTat")
        
    def play(self, opponent):
        if not self.history:
            return 'C'
        if opponent.history[-1] == 'D':
            return 'D'
        return 'C'

class Alternator(Strategy):
    def __init__(self) -> None:
        super().__init__("Alternator")
    
    def play(self, opponent):
        if len(self.history) == 0:
            return 'C'
        if self.history[-1] == 'C':
            return 'D'
        return 'C'

class Defector(Strategy):
    def __init__(self) -> None:
        super().__init__("Defector")
    
    def play(self, opponent):
        return 'D'

class Cooperator(Strategy):
    def __init__(self) -> None:
        super().__init__("Cooperator")
    
    def play(self, opponent):
        return 'C'
    
class Random(Strategy):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__("Random")
        self.p = p
    
    def play(self, opponent):
        return random_choice(self.p)
    
class FirstByAnonymous(Strategy):
    """
    A strategy that randomly cooperates with a probability uniformly distributed 
    between 30% and 70% each turn.
    """

    def __init__(self) -> None:
        super().__init__("Anonymous")

    def play(self, opponent: Strategy) -> str:
        # Randomly choose a cooperation probability between 30% and 70%
        r = random.uniform(3, 7) / 10
        # Use the random_choice function to decide whether to cooperate or defect
        return random_choice(r)

class FirstByJoss(Strategy):
    """
    Strategy from Johann Joss for Axelrod's first tournament.
    """

    def __init__(self, p: float = 0.9) -> None:
        """
        Parameters
        ----------
        p, float
            The probability of cooperating when the previous round was (C, C)
            or (D, C), i.e. the opponent cooperated.
        """
        super().__init__("Joss")
        self.p = p

    def play(self, opponent: Strategy) -> str:
        if not opponent.history:
            return 'C'  # Cooperate on the first move

        if opponent.history[-1] == 'D':
            return 'D'  # Always defect after a defection by the opponent

        # Cooperate with probability p if the opponent cooperated last time
        return random_choice(self.p)

class Grudger(Strategy):
    """
    A strategy that starts by cooperating, but will defect if at any point the
    opponent has defected.
    """

    def __init__(self) -> None:
        super().__init__("Grudger")
        self.opponent_defected = False

    def play(self, opponent: Strategy) -> str:
        if self.opponent_defected:
            return 'D'  # If opponent has defected at any point, defect forever.
        
        if 'D' in opponent.history:
            self.opponent_defected = True  # Mark if opponent defects
            return 'D'  # First time opponent defects, defect
        
        return 'C'  # Cooperate until the opponent defects
    
# FirstByGrofman strategy integrated with the Strategy base class
class FirstByGrofman(Strategy):
    """
    Strategy from Grofman for Axelrod's first tournament.
    """

    def __init__(self) -> None:
        super().__init__("Grofman")

    def play(self, opponent: Strategy) -> str:
        # If the history is empty or if the last moves are the same, always cooperate
        if len(self.history) == 0 or self.history[-1] == opponent.history[-1]:
            return 'C'
        # Otherwise, cooperate with probability 2/7
        return random_choice(2 / 7)

# FirstByNydegger strategy class integrated with Strategy base class
class FirstByNydegger(Strategy):
    """
    Strategy from Rudy Nydegger's implementation for Axelrod's first tournament.
    """

    def __init__(self) -> None:
        super().__init__("Nydegger")
        self.As = [1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61]
        self.score_map = {( 'C', 'C'): 0, ('C', 'D'): 2, ('D', 'C'): 1, ('D', 'D'): 3}

    def score_history(
        self, my_history: List[str], opponent_history: List[str], score_map: Dict[Tuple[str, str], int]
    ) -> int:
        """Implements the Nydegger formula A = 16 a_1 + 4 a_2 + a_3"""
        a = 0
        for i, weight in [(-1, 16), (-2, 4), (-3, 1)]:
            plays = (my_history[i], opponent_history[i])
            a += weight * score_map[plays]
        return a

    def play(self, opponent) -> str:
        if len(self.history) == 0:
            return 'C'  # Cooperate on the first move

        if len(self.history) == 1:
            # Tit for Tat for the first two moves
            return 'D' if opponent.history[-1] == 'D' else 'C'

        if len(self.history) == 2:
            # Special condition when the opponent defected only on the second move
            if opponent.history[0:2] == ['D', 'C']:
                return 'D'
            else:
                # Tit for Tat for the third move
                return 'D' if opponent.history[-1] == 'D' else 'C'

        # Calculate A based on the last three moves
        A = self.score_history(self.history[-3:], opponent.history[-3:], self.score_map)
        
        # Defect if A is one of the predefined values
        if A in self.As:
            return 'D'
        return 'C'

# FirstByTidemanAndChieruzzi strategy integrated with the Strategy base class
class FirstByTidemanAndChieruzzi(Strategy):
    """
    Strategy from Tideman and Chieruzzi for Axelrod's first tournament.
    """

    def __init__(self) -> None:
        super().__init__("Tideman and Chieruzzi")
        self.is_retaliating = False
        self.retaliation_length = 0
        self.retaliation_remaining = 0
        self.current_score = 0
        self.opponent_score = 0
        self.last_fresh_start = 0
        self.fresh_start = False
        self.remembered_number_of_opponent_defectioons = 0
        # Define a score map for game outcomes
        self.score_map = {('C', 'C'): (3, 3), ('C', 'D'): (0, 5), ('D', 'C'): (5, 0), ('D', 'D'): (1, 1)}

    def _decrease_retaliation_counter(self):
        """Lower the remaining owed retaliation count and flip to non-retaliate
        if the count drops to zero."""
        if self.is_retaliating:
            self.retaliation_remaining -= 1
            if self.retaliation_remaining == 0:
                self.is_retaliating = False

    def _fresh_start(self):
        """Give the opponent a fresh start by forgetting the past."""
        self.is_retaliating = False
        self.retaliation_length = 0
        self.retaliation_remaining = 0
        self.remembered_number_of_opponent_defectioons = 0

    def _score_last_round(self, opponent: Strategy):
        """Updates the scores for each player."""
        # Use predefined score map for action pairs
        last_round = (self.history[-1], opponent.history[-1])
        scores = self.score_map[last_round]
        self.current_score += scores[0]
        self.opponent_score += scores[1]

    def play(self, opponent: Strategy) -> str:
        if not opponent.history:
            return 'C'  # Cooperate on the first move

        if opponent.history[-1] == 'D':
            self.remembered_number_of_opponent_defectioons += 1

        # Calculate the scores based on the previous round's actions.
        self._score_last_round(opponent)

        # Check if we have recently given the strategy a fresh start.
        if self.fresh_start:
            self.fresh_start = False
            return 'C'  # Second cooperation after fresh start

        # Check conditions to give opponent a fresh start.
        current_round = len(self.history) + 1
        valid_fresh_start = current_round - self.last_fresh_start >= 20 if self.last_fresh_start != 0 else True

        if valid_fresh_start:
            valid_points = self.current_score - self.opponent_score >= 10
            valid_rounds = len(self.history) < 10  # There should be at least 10 rounds remaining
            opponent_is_cooperating = opponent.history[-1] == 'C'

            # Calculate the standard deviation for the 50-50 deviation check
            if valid_points and valid_rounds and opponent_is_cooperating:
                N = len(opponent.history)  # Total number of moves so far
                std_deviation = np.sqrt(N * 0.5 * 0.5)  # Binomial distribution std deviation for 50-50

                lower = N / 2 - 3 * std_deviation
                upper = N / 2 + 3 * std_deviation
                if (self.remembered_number_of_opponent_defectioons <= lower or
                    self.remembered_number_of_opponent_defectioons >= upper):
                    # Opponent deserves a fresh start
                    self.last_fresh_start = current_round
                    self._fresh_start()
                    self.fresh_start = True
                    return 'C'  # First cooperation after fresh start

        # Retaliation logic
        if self.is_retaliating:
            # Are we retaliating still?
            self._decrease_retaliation_counter()
            return 'D'

        if opponent.history[-1] == 'D':
            self.is_retaliating = True
            self.retaliation_length += 1
            self.retaliation_remaining = self.retaliation_length
            self._decrease_retaliation_counter()
            return 'D'

        return 'C'

class FirstByShubik(Strategy):
    """
    Strategy from Martin Shubik for Axelrod's first tournament.
    """

    def __init__(self) -> None:
        super().__init__("Shubik")
        self.is_retaliating = False
        self.retaliation_length = 0
        self.retaliation_remaining = 0

    def _decrease_retaliation_counter(self):
        """Lower the remaining owed retaliation count and flip to non-retaliate
        if the count drops to zero."""
        if self.is_retaliating:
            self.retaliation_remaining -= 1
            if self.retaliation_remaining == 0:
                self.is_retaliating = False

    def play(self, opponent: Strategy) -> str:
        if not opponent.history:
            return 'C'

        if self.is_retaliating:
            # Are we retaliating still?
            self._decrease_retaliation_counter()
            return 'D'

        if opponent.history[-1] == 'D' and self.history[-1] == 'C':
            # "If he uses his move 2 again after I have resumed using move 1,
            # then I will switch to move 2 for the k + 1 immediately subsequent
            # periods"
            self.is_retaliating = True
            self.retaliation_length += 1
            self.retaliation_remaining = self.retaliation_length
            self._decrease_retaliation_counter()
            return 'D'
        return 'C'

class FirstBySteinAndRapoport(Strategy):
    """
    Strategy from Stein and Rapoport for Axelrod's first tournament.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Parameters
        ----------
        alpha: float
            The significance level of p-value from chi-squared test with
            alpha == 0.05 by default.
        """
        super().__init__("Stein and Rapoport")
        self.alpha = alpha
        self.opponent_is_random = False

    def play(self, opponent: Strategy) -> str:
        round_number = len(self.history) + 1

        # First 4 moves: Cooperate
        if round_number < 5:
            return 'C'
        
        # For first 15 rounds: Tit for tat
        elif round_number < 15:
            return opponent.history[-1]

        # Every 15 moves: Check if opponent is playing randomly using chi-squared test
        if round_number % 15 == 0:
            # Count the cooperations and defections of the opponent manually
            cooperations = opponent.history.count('C')
            defections = opponent.history.count('D')
            
            # Perform chi-squared test to check for randomness
            p_value = chisquare([cooperations, defections]).pvalue
            self.opponent_is_random = p_value >= self.alpha

        if self.opponent_is_random:
            # Defect if opponent plays randomly
            return 'D'
        else:
            # Tit for tat if opponent plays non-randomly
            return opponent.history[-1]

class FirstByTullock(Strategy):
    """
    Strategy from Tullock for Axelrod's first tournament.
    """

    def __init__(self) -> None:
        """
        Initializes the strategy with the number of rounds to cooperate initially (11 rounds).
        """
        super().__init__("Tullock")
        self._rounds_to_cooperate = 11
        self.memory_depth = self._rounds_to_cooperate

    def play(self, opponent: Strategy) -> str:
        if len(self.history) < self._rounds_to_cooperate:
            return 'C'  # Cooperate for the first 11 rounds
        
        rounds = self._rounds_to_cooperate - 1
        # Count the number of times the opponent cooperated in the last 10 rounds
        cooperate_count = opponent.history[-rounds:].count('C')
        
        # Calculate the proportion of cooperation and reduce it by 10%
        prop_cooperate = cooperate_count / rounds
        prob_cooperate = max(0, prop_cooperate - 0.10)
        
        # Use the calculated probability to make a decision on whether to cooperate
        return random_choice(prob_cooperate)

class FirstByDavis(Strategy):
    """
    A strategy that cooperates for a set number of rounds and then plays Grudger,
    defecting if at any point the opponent has defected.
    """

    def __init__(self, rounds_to_cooperate: int = 10) -> None:
        """
        Parameters
        ----------
        rounds_to_cooperate: int, 10
           The number of rounds to cooperate initially.
        """
        super().__init__("Davis")
        self._rounds_to_cooperate = rounds_to_cooperate
        self.opponent_defected = False

    def play(self, opponent: Strategy) -> str:
        if len(self.history) < self._rounds_to_cooperate:
            return 'C'  # Cooperate for the first _rounds_to_cooperate moves
        
        if 'D' in opponent.history:
            self.opponent_defected = True  # Mark if opponent defects
            return 'D'  # Defect once opponent defects

        return 'C'  # Continue cooperating if opponent hasn't defected

class FirstByGraaskamp(Strategy):
    """
    Strategy from Graaskamp for Axelrod's first tournament.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Parameters
        ----------
        alpha: float
            The significance level of p-value from chi-squared test with
            alpha == 0.05 by default.
        """
        super().__init__("Graaskamp")
        self.alpha = alpha
        self.opponent_is_random = False
        self.next_random_defection_turn = None

    def play(self, opponent: Strategy) -> str:
        # First move: Cooperate
        if not self.history:
            return 'C'
        
        # First 50 rounds: Tit-for-tat (except for round 51)
        if len(self.history) < 50:
            return opponent.history[-1]

        # Defect on round 51
        if len(self.history) == 50:
            return 'D'

        # Next 5 rounds: Tit-for-tat
        if len(self.history) < 56:
            return opponent.history[-1]

        # Check if opponent plays randomly using chi-squared test
        p_value = chisquare([opponent.history.count('C'), opponent.history.count('D')]).pvalue
        self.opponent_is_random = (p_value >= self.alpha) or self.opponent_is_random

        if self.opponent_is_random:
            return 'D'

        # Check if opponent is playing Tit for Tat or a clone of itself
        if all(opponent.history[i] == self.history[i - 1] for i in range(1, len(self.history))) or opponent.history == self.history:
            return opponent.history[-1]  # Continue Tit-for-Tat

        # Randomly defect every 5 to 15 moves
        if self.next_random_defection_turn is None:
            self.next_random_defection_turn = random.randint(5, 15) + len(self.history)

        if len(self.history) == self.next_random_defection_turn:
            self.next_random_defection_turn = random.randint(5, 15) + len(self.history)
            return 'D'
        
        return 'C'

class FirstByDowning(Strategy):
    """
    Strategy from Downing for Axelrod's first tournament.
    """

    def __init__(self) -> None:
        super().__init__("Downing")
        self.number_opponent_cooperations_in_response_to_C = 0
        self.number_opponent_cooperations_in_response_to_D = 0

    def play(self, opponent: Strategy) -> str:
        round_number = len(self.history) + 1

        # Defect on the first round
        if round_number == 1:
            return 'D'
        
        # Defect on the second round if opponent cooperates
        if round_number == 2:
            if opponent.history[-1] == 'C':
                self.number_opponent_cooperations_in_response_to_C += 1
            return 'D'

        # Update cooperations based on the opponent's previous moves
        if self.history[-2] == 'C' and opponent.history[-1] == 'C':
            self.number_opponent_cooperations_in_response_to_C += 1
        if self.history[-2] == 'D' and opponent.history[-1] == 'C':
            self.number_opponent_cooperations_in_response_to_D += 1

        # Count the number of cooperations and defections by the player
        cooperations = self.history.count('C')
        defections = self.history.count('D')

        # Calculate alpha (P(C_o | C_s)) and beta (P(C_o | D_s))
        alpha = (self.number_opponent_cooperations_in_response_to_C /
                 (cooperations + 1))  # Adding 1 to avoid division by zero
        beta = (self.number_opponent_cooperations_in_response_to_D /
                max(defections, 2))  # Adding 2 to defections to avoid division by zero

        # Define the payoff values for the Prisoner's Dilemma
        R, P, S, T = 3, 1, 0, 5  # Typical payoff values for the PD game
        expected_value_of_cooperating = alpha * R + (1 - alpha) * S
        expected_value_of_defecting = beta * T + (1 - beta) * P

        # Compare expected utilities and decide whether to cooperate or defect
        if expected_value_of_cooperating > expected_value_of_defecting:
            return 'C'
        if expected_value_of_cooperating < expected_value_of_defecting:
            return 'D'

        # In case of a tie, flip the previous move
        if self.history[-1] == 'C':
            return 'D'
        return 'C'

class FirstByFeld(Strategy):
    """
    Strategy from Feld for Axelrod's first tournament.
    """

    def __init__(
        self,
        start_coop_prob: float = 1.0,
        end_coop_prob: float = 0.5,
        rounds_of_decay: int = 200,
    ) -> None:
        """
        Parameters
        ----------
        start_coop_prob, float
            The initial probability to cooperate
        end_coop_prob, float
            The final probability to cooperate
        rounds_of_decay, int
            The number of rounds to linearly decrease from start_coop_prob
            to end_coop_prob
        """
        super().__init__("Feld")
        self._start_coop_prob = start_coop_prob
        self._end_coop_prob = end_coop_prob
        self._rounds_of_decay = rounds_of_decay

    def _cooperation_probability(self) -> float:
        """Linearly decreases the probability of cooperation."""
        diff = self._end_coop_prob - self._start_coop_prob
        slope = diff / self._rounds_of_decay
        rounds = len(self.history)
        return max(self._start_coop_prob + slope * rounds, self._end_coop_prob)

    def play(self, opponent: Strategy) -> str:
        if not opponent.history:
            return 'C'  # Cooperate on the first move

        if opponent.history[-1] == 'D':
            return 'D'  # Defect if opponent defects

        # Calculate cooperation probability and make a random choice based on that
        p = self._cooperation_probability()
        return random_choice(p)