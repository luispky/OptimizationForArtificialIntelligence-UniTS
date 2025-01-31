from __future__ import annotations  # Automatically treats all type hints as forward references
from typing import List, Union
from abc import ABC, abstractmethod
from scipy.stats import chisquare
import numpy as np
from .utils import RandomNumberGenerator


class Player(ABC):
    def __init__(self, name: str):
        self.name = name
        self.score = 0
        self.history = []
        self.cooperations = 0
        self.defections = 0
        self.set_seed(None)
        self.stochastic = False
    
    def reset(self):
        """Reset the player's history, score, and cooperation/defection counts."""
        self.history = []
        self.score = 0
        self.cooperations = 0
        self.defections = 0
        self.set_seed(None)

    def set_seed(self, seed: Union[int, None] = None) -> None:
        if seed is None:
            self._seed = np.random.randint(0, 2**32 - 1)
        else:
            self._seed = seed   
        self._rng = RandomNumberGenerator(seed=self._seed)

    @abstractmethod
    def strategy(self, opponent: Player) -> str:
        "Return an action, 'C' or 'D', potentially based on the opponent's history."
        raise NotImplementedError
    
    def update(self, move: str, score: int) -> None:
        """Update the player's history, score, and cooperation/defection counts."""
        self.history.append(move)
        if move == 'C':
            self.cooperations += 1
        else:
            self.defections += 1
        self.score += score


class TitForTat(Player):
    """Cooperates on the first round and imitates its opponent's previous move thereafter."""
    def __init__(self) -> None:
        super().__init__("TitForTat")
        
    def strategy(self, opponent: Player):
        if len(opponent.history) == 0:
            return 'C'
        if opponent.history[-1] == 'D':
            return 'D'
        return 'C'


class Alternator(Player):
    """Starts by cooperating and then alternates between cooperation and defection according
    to the opponent's previous move."""
    def __init__(self) -> None:
        super().__init__("Alternator")
    
    def strategy(self, opponent: Player):
        if len(self.history) == 0:
            return 'C'
        if self.history[-1] == 'C':
            return 'D'
        return 'C'


class Defector(Player):
    """Always defects."""
    def __init__(self) -> None:
        super().__init__("Defector")
    
    def strategy(self, opponent: Player):
        return 'D'


class Cooperator(Player):
    """Always cooperates."""
    def __init__(self) -> None:
        super().__init__("Cooperator")
    
    def strategy(self, opponent: Player):
        return 'C'


class Random(Player):
    """Randomly chooses to cooperate or defect with a given probability."""
    def __init__(self, p: float = 0.5) -> None:
        super().__init__("Random")
        self.p = p
        self.stochastic = True
    
    def strategy(self, opponent: Player):
        return self._rng.random_choice(self.p)


class FirstByTidemanAndChieruzzi(Player):
    """
    This strategy begins with cooperation and follows a Tit-for-Tat (TFT) approach.
    After the opponent defects, it retaliates with defections for a series of moves,
    with the retaliation length increasing after each series of defections from the opponent.
    
    A "fresh start" is given to the opponent under the following conditions:
    - The opponent is at least 10 points behind.
    - It has been at least 20 moves since the last fresh start.
    - There are at least 10 moves remaining in the game.
    - The opponent’s number of defections deviates by more than 3 standard deviations from a random 50-50 distribution.

    When a fresh start occurs, the strategy forgets the past history, cooperates twice, and
    resumes as if the game had just started.
    """
    def __init__(self) -> None:
        super().__init__("Tideman and Chieruzzi")
        self.is_retaliating = False
        self.retaliation_length = 0
        self.retaliation_remaining = 0
        self.last_fresh_start = 0
        self.fresh_start = False
        self.remembered_number_of_opponent_defectioons = 0

    def _decrease_retaliation_counter(self):
        """Lower the remaining owed retaliation count and flip to non-retaliate
        if the count drops to zero."""
        if self.is_retaliating:
            self.retaliation_remaining -= 1
            if self.retaliation_remaining == 0:
                self.is_retaliating = False

    def _fresh_start(self):
        """Give the opponent a fresh start by forgetting the past"""
        self.is_retaliating = False
        self.retaliation_length = 0
        self.retaliation_remaining = 0
        self.remembered_number_of_opponent_defectioons = 0

    def strategy(self, opponent: Player) -> str:
        if not opponent.history:
            return 'C'  # Cooperate on the first move

        if opponent.history[-1] == 'D':
            self.remembered_number_of_opponent_defectioons += 1

        # Check if we have recently given the strategy a fresh start.
        if self.fresh_start:
            self.fresh_start = False
            return 'C'  # Second cooperation after fresh start

        # Check conditions to give opponent a fresh start.
        current_round = len(self.history) + 1
        valid_fresh_start = current_round - self.last_fresh_start >= 20 if self.last_fresh_start != 0 else True

        if valid_fresh_start:
            valid_points = self.score - opponent.score >= 10  # Opponent is 10 or more points behind
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


class FirstByNydegger(Player):
    """
    This strategy begins with Tit-for-Tat (TFT) for the first three moves. However, if it was the only player
    to cooperate on the first move and the only one to defect on the second move, it defects on the third move.
    After the third move, the strategy's decision is based on the outcomes of the previous three moves.

    The strategy calculates a value `A` using the outcomes of the last three moves as follows:
    - The value of `A` is computed as `A = 16a1 + 4a2 + a3`, where:
      - `a1`, `a2`, and `a3` represent the outcomes of the first, second, and third most recent moves, respectively.
      - If both strategies defect, `ai = 3`.
      - If the opponent defects and the strategy cooperates, `ai = 2`.
      - If the strategy defects and the opponent cooperates, `ai = 1`.

    The strategy defects only when the computed value `A` is one of the following:
    `{1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61}`

    Notably, if the last three moves are mutual defections, then `A = 63`, and the strategy will cooperate.
    """
    def __init__(self) -> None:
        super().__init__("Nydegger")
        self.As = [1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54, 55, 58, 61]
        self.score_map = {( 'C', 'C'): 0, ('C', 'D'): 2, ('D', 'C'): 1, ('D', 'D'): 3}

    def score_history(
        self, player_history: List[str], opponent_history: List[str]
        ) -> int:
        """Implements the Nydegger formula A = 16 a_1 + 4 a_2 + a_3"""
        a = 0
        for i, weight in [(-1, 16), (-2, 4), (-3, 1)]:
            plays = (player_history[i], opponent_history[i])
            a += weight * self.score_map[plays]
        return a

    def strategy(self, opponent: Player) -> str:
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
        A = self.score_history(self.history[-3:], opponent.history[-3:])
        
        # Defect if A is one of the predefined values
        if A in self.As:
            return 'D'
        return 'C'


class FirstByGrofman(Player):
    """
    If the players did different things on the previous move, this rule cooperates with probability 2/7.
    Otherwise this rule always cooperates.
    """
    def __init__(self) -> None:
        super().__init__("Grofman")
        self.stochastic = True

    def strategy(self, opponent: Player) -> str:
        # If the history is empty or if the last moves are the same, always cooperate
        if len(self.history) == 0 or self.history[-1] == opponent.history[-1]:
            return 'C'
        # Otherwise, cooperate with probability 2/7
        return self._rng.random_choice(2/7)


class FirstByShubik(Player):
    """
    The player cooperates, if when it is cooperating, the opponent defects it defects for k rounds.
    After k rounds it starts cooperating again and increments the value of k if the opponent defects again.
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

    def strategy(self, opponent: Player) -> str:
        if len(opponent.history) == 0:
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


class FirstBySteinAndRapoport(Player):
    """
    This rule plays tit for tat except that it cooperates on the first four moves, and every fifteen moves
    it checks to see if the opponent seems to be playing randomly. This check uses a chi-squared test of
    the other’s transition probabilities and also checks for alternating moves of CD and DC.
    """
    def __init__(self, alpha: float = 0.05) -> None:
        """
        Parameters
        ----------
        alpha: float
            The significance level of p-value from chi-squared test.
        """
        super().__init__("Stein and Rapoport")
        self.alpha = alpha
        self.opponent_is_random = False

    def strategy(self, opponent: Player) -> str:
        round_number = len(self.history) + 1

        # First 4 moves: Cooperate
        if round_number < 5:
            return 'C'
        
        # For first 15 rounds: Tit for tat
        elif round_number < 15:
            return opponent.history[-1]

        # Every 15 moves: Check if opponent is playing randomly using chi-squared test
        if round_number % 15 == 0:
            # Perform chi-squared test to check for randomness
            p_value = chisquare([opponent.cooperations, opponent.defections]).pvalue
            self.opponent_is_random = p_value >= self.alpha

        if self.opponent_is_random:
            # Defect if opponent plays randomly
            return 'D'
        else:
            # Tit for tat if opponent plays non-randomly
            return opponent.history[-1]


class Grudger(Player):
    """
    A player starts by cooperating however will defect if at any point the opponent has defected.
    """
    def __init__(self) -> None:
        super().__init__("Grudger")

    def strategy(self, opponent: Player) -> str:
        if opponent.defections > 0:
            return 'D'
        return 'C'


class FirstByDavis(Player):
    """
    A strategy that cooperates for a set number of rounds and then plays Grudger,
    defecting if at any point the opponent has defected.
    """

    def __init__(self, rounds_to_cooperate: int = 10) -> None:
        """
        Parameters
        ----------
        rounds_to_cooperate: int
           The number of rounds to cooperate initially.
        """
        super().__init__("Davis")
        self._rounds_to_cooperate = rounds_to_cooperate

    def strategy(self, opponent: Player) -> str:
        if len(self.history) < self._rounds_to_cooperate:
            return 'C'  # Cooperate for the first _rounds_to_cooperate moves
        
        if opponent.defections > 0:
            return 'D'
        return 'C'  # Continue cooperating if opponent hasn't defected


class FirstByGraaskamp(Player):
    """
    This rule plays tit for tat for 50 moves, defects on move 51, and then plays 5 more moves of tit for tat.
    A check is then made to see if the player seems to be RANDOM, in which case it defects from then on.
    A check is also made to see if the other is TIT FOR TAT, and its own twin, in which case it plays
    tit for tat. Otherwise it randomly defects every 5 to 15 moves, hoping that enough trust has been built up 
    so that the other player will not notice these defections.
    """
    def __init__(self, alpha: float = 0.05) -> None:
        """
        Parameters
        ----------
        alpha: float
            The significance level of p-value from chi-squared test.
        """
        super().__init__("Graaskamp")
        self.alpha = alpha
        self.opponent_is_random = False
        self.stochastic = True
        self.next_random_defection_turn = None

    def strategy(self, opponent: Player) -> str:
        # First move: Cooperate
        if len(self.history) == 0:
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
        p_value = chisquare([opponent.cooperations, opponent.defections]).pvalue
        self.opponent_is_random = (p_value >= self.alpha) or self.opponent_is_random

        if self.opponent_is_random:
            return 'D'

        # Check if opponent is playing Tit for Tat or a clone of itself
        if all(
            opponent.history[i] == self.history[i - 1] 
            for i in range(1, len(self.history))
        ) or opponent.history == self.history:
            return opponent.history[-1]  # Continue Tit-for-Tat

        # Randomly defect every 5 to 15 moves
        if self.next_random_defection_turn is None:
            self.next_random_defection_turn = self._rng.randint(5, 16) + len(self.history)

        if len(self.history) == self.next_random_defection_turn:
            self.next_random_defection_turn = self._rng.randint(5, 16) + len(self.history)
            return 'D'
        return 'C'


class FirstByDowning(Player):
    """
    This strategy, based on Downing's model, aims to maximize long-term payoff by estimating
    the opponent’s reaction probabilities based on prior moves. It tracks two conditional
    probabilities: P(C_o | C_s) (the likelihood the opponent cooperates after the player cooperates)
    and P(C_o | D_s) (the likelihood the opponent cooperates after the player defects). The strategy
    then calculates the expected payoffs for always cooperating (E_C) and always defecting (E_D)
    using these probabilities. It chooses cooperation if E_C > E_D, defection if E_C < E_D,
    and alternates moves if E_C = E_D.

    Initially, both probabilities are set to 0.5, leading the strategy to defect in the first two rounds,
    as no opponent behavior has been observed. Over time, the strategy updates its estimates based on
    the opponent's responses, refining its decisions for future moves.
    """
    def __init__(self) -> None:
        super().__init__("Downing")
        self.number_opponent_cooperations_in_response_to_C = 0
        self.number_opponent_cooperations_in_response_to_D = 0

    def strategy(self, opponent: Player) -> str:
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

        # Calculate alpha (P(C_o | C_s)) and beta (P(C_o | D_s))
        alpha = (self.number_opponent_cooperations_in_response_to_C /
                 (self.cooperations + 1))
        # Adds 1 on the assumption that the first opponent move being a response to a cooperation
        beta = (self.number_opponent_cooperations_in_response_to_D /
                max(self.defections, 2)) 
        # Adds 2 to defections on the assumption that the first two moves are defections

        # Define the payoff values for the Prisoner's Dilemma
        R, P, S, T = 3, 1, 0, 5  # Typical payoff values for the PD game
        expected_value_of_cooperating = alpha * R + (1 - alpha) * S
        expected_value_of_defecting = beta * T + (1 - beta) * P

        # Compare expected utilities and decide whether to cooperate or defect
        if expected_value_of_cooperating > expected_value_of_defecting:
            return 'C'
        if expected_value_of_cooperating < expected_value_of_defecting:
            return 'D'

        # If the expected values are equal, alternate between cooperation and defection
        if self.history[-1] == 'C':
            return 'D'
        return 'C'


class FirstByFeld(Player):
    """
    This strategy starts with Tit For Tat, cooperating when the opponent cooperates and defecting
    when the opponent defects. Over time, it gradually reduces its probability of cooperating after
    the opponent’s cooperation, decreasing linearly until the probability reaches 0.5 by the 200th move.
    If the opponent defects, the strategy always defects in response. The cooperation probability starts
    at 1.0 and decays to 0.5 over a set number of rounds (200).
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
        start_coop_prob
            The initial probability to cooperate
        end_coop_prob
            The final probability to cooperate
        rounds_of_decay
            The number of rounds to linearly decrease from start_coop_prob
            to end_coop_prob
        """
        super().__init__("Feld")
        self._start_coop_prob = start_coop_prob
        self._end_coop_prob = end_coop_prob
        self._rounds_of_decay = rounds_of_decay
        self.stochastic = True

    def _cooperation_probability(self) -> float:
        """Linearly decreases the probability of cooperation."""
        diff = self._end_coop_prob - self._start_coop_prob
        slope = diff / self._rounds_of_decay
        rounds = len(self.history)
        return max(self._start_coop_prob + slope * rounds, self._end_coop_prob)

    def strategy(self, opponent: Player) -> str:
        if len(self.history) == 0:
            return 'C'  # Cooperate on the first move

        if opponent.history[-1] == 'D':
            return 'D'  # Defect if opponent defects

        # Calculate cooperation probability and make a random choice based on that
        p = self._cooperation_probability()
        return self._rng.random_choice(p)


class FirstByJoss(Player):
    """
    This rule cooperates 90% of the time after a cooperation by the other. It always defects after a defection by the other.
    """
    def __init__(self, p: float = 0.9) -> None:
        """
        Parameters
        ----------
        p
            The probability of cooperating when the previous round was (C, C)
            or (D, C), i.e. the opponent cooperated.
        """
        super().__init__("Joss")
        self.p = p
        self.stochastic = True

    def strategy(self, opponent: Player) -> str:
        if len(opponent.history) == 0:
            return 'C'  # Cooperate on the first move
        
        p = 0
        
        if self.history[-1] == 'C' and opponent.history[-1] == 'C':
            p = self.p
        elif self.history[-1] == 'C' and opponent.history[-1] == 'D':
            p = 0
        elif self.history[-1] == 'D' and opponent.history[-1] == 'C':
            p = self.p
        elif self.history[-1] == 'D' and opponent.history[-1] == 'D':
            p = 0
        return self._rng.random_choice(p)


class FirstByTullock(Player):
    """
    Cooperates for the first 11 rounds then randomly cooperates 10% less often 
    than the opponent has in the previous 10 rounds.
    """
    def __init__(self) -> None:
        """
        Initializes the strategy with the number of rounds to cooperate initially (11 rounds).
        """
        super().__init__("Tullock")
        self._rounds_to_cooperate = 11
        self.stochastic = True

    def strategy(self, opponent: Player) -> str:
        if len(self.history) < self._rounds_to_cooperate:
            return 'C'  # Cooperate for the first 11 rounds
        
        rounds = self._rounds_to_cooperate - 1
        # Count the number of times the opponent cooperated in the last 10 rounds
        cooperate_count = opponent.history[-rounds:].count('C')
        # Calculate the proportion of cooperation and reduce it by 10%
        prob_cooperate = max(0, cooperate_count / rounds - 0.10)
        
        # Use the calculated probability to make a decision on whether to cooperate
        return self._rng.random_choice(prob_cooperate)


class FirstByAnonymous(Player):
    """
    A strategy that randomly cooperates with a probability uniformly distributed 
    between 30% and 70% each turn.
    """

    def __init__(self) -> None:
        super().__init__("Anonymous")

    def strategy(self, opponent: Player) -> str:
        # Randomly choose a cooperation probability between 30% and 70%
        r = self._rng.uniform(3, 7) / 10
        # Use the random_choice function to decide whether to cooperate or defect
        return self._rng.random_choice(r)
    