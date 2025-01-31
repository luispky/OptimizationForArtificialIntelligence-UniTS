from src.strategies import (
    TitForTat,
    Alternator,
    Defector,
    Cooperator,
    Random,
    FirstByTidemanAndChieruzzi,
    Player,
)
from src.game import Match, Tournament
from src.evo_strategy import EvoStrategy
from src.utils import plot_full_history_evo_player
import argparse


def match_against_evo(player: Player, turns: int, evo_action_history_size: int) -> None:
    """
    Test a player against an EvoStrategy player.

    Args:
        player (Player): The player to test.
        turns (int): Number of turns for the match.
        evo_action_history_size (int): Action history size for EvoStrategy.
    """
    print(f"\n{player.name} vs EvoStrategy")
    players = [player, EvoStrategy(name="Evo", action_history_size=evo_action_history_size, log_history=True)]
    match = Match(players, turns=turns)
    moves = match.play()
    print(f"Moves:\n{moves}")
    print(f"Winner: {match.winner}")
    print(f"Final scores: {match.final_scores}")
    plot_full_history_evo_player(players[1])
    print("\n" + "-" * 50)

def test_match(turns: int, evo_action_history_size: int) -> None:
    """
    Test the Match class with various players against an EvoStrategy.

    Args:
        turns (int): Number of turns for the match.
        evo_action_history_size (int): Action history size for EvoStrategy.
    """
    print("\nTesting Match class with one EvoStrategy player")
    # Showcase that the action history size can be different for each EvoStrategy player
    evo1 = EvoStrategy(name="Evo1", action_history_size=evo_action_history_size//2, log_history=True)
    test_players = [evo1, Alternator(), TitForTat(), Random(), FirstByTidemanAndChieruzzi()]

    for player in test_players:
        match_against_evo(player, turns, evo_action_history_size)


def test_tournament(
    action_history_sizes: range, turns: int, repetitions: int, prob_end: float
) -> None:
    """
    Test the Tournament class with various strategies and EvoStrategy.

    Args:
        action_history_sizes (range): Range of action history sizes for EvoStrategy.
        turns (int): Number of turns for the match.
        repetitions (int): Number of tournament repetitions.
        prob_end (float): Probability of ending a match.
    """
    print("\nTesting Tournament class with one EvoStrategy")
    for action_history_size in action_history_sizes:
        print(f"\nResults for EvoStrategy with action history size = {action_history_size}")
        players = [
            Cooperator(),
            Defector(),
            Alternator(),
            TitForTat(),
            Random(),
            EvoStrategy(name="Evo", action_history_size=action_history_size),
        ]
        tournament = Tournament(players, turns=turns, repetitions=repetitions, prob_end=prob_end)
        tournament.play(axelrod=True)
        tournament.print_ranked_results()
        plot_full_history_evo_player([p for p in players if p.name == "Evo"][0])
        print("\n" + "-" * 50)


def main() -> None:
    """
    Main function to parse command-line arguments and run the specified tests.
    """
    parser = argparse.ArgumentParser(
        description="Run tests for Match and Tournament classes with EvoStrategy players"
    )

    # General parameters
    parser.add_argument("--test", choices=["match", "tournament"], required=True, help="Specify which test to run.")

    # Shared parameters
    parser.add_argument("--turns", type=int, default=100, help="Number of turns for matches.")
    parser.add_argument("--evo_action_history_size", type=int, default=10, help="EvoStrategy action history size.")

    # Tournament-specific parameters
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions in the tournament.")
    parser.add_argument("--prob_end", type=float, default=0.01, help="Probability of ending a match in the tournament.")

    args = parser.parse_args()

    if args.test == "match":
        test_match(turns=args.turns, evo_action_history_size=args.evo_action_history_size)
    elif args.test == "tournament":
        test_tournament(
            action_history_sizes=range(0, args.evo_action_history_size, 2),
            turns=args.turns,
            repetitions=args.repetitions,
            prob_end=args.prob_end,
        )
    else:
        print("Invalid test type. Available options are: 'evo_match', 'match', or 'tournament'.")


if __name__ == "__main__":
    main()
