from src.strategies import (TitForTat,
                            Alternator,
                            Defector,
                            Cooperator,
                            Random,
                            FirstByTidemanAndChieruzzi,
                            FirstByNydegger,
                            FirstByGrofman,
                            FirstByShubik,
                            FirstBySteinAndRapoport,
                            Grudger,
                            FirstByDavis,
                            FirstByGraaskamp,
                            FirstByDowning,
                            FirstByFeld,
                            FirstByJoss,
                            FirstByTullock,
                            FirstByAnonymous,
                            Player
)
from src.game import Match, Tournament, EvolutionaryIterativePrisonersDilemma
from src.evo_strategy import EvoStrategy
from src.utils import (
    plot_full_history_evo_player,
    set_seeds,
    plot_best_players
)
import argparse

# import numpy as np
# SEED = np.random.randint(0, 1000)
SEED = 42
set_seeds(SEED)

def match_against_evo(player: Player,
                       turns: int,
                       evo_action_history_size: int):
    """Test a player against an EvoStrategy player"""
    
    print(f"\n{player.name} vs EvoStrategy")
    players = [player, EvoStrategy(name='Evo', action_history_size=evo_action_history_size)]
    match = Match(players, turns=turns)
    moves = match.play()
    print(f"Moves: \n {moves}")
    print(f"Winner: {match.winner}")
    print(f"Final scores: {match.final_scores}")
    
    plot_full_history_evo_player(players[1])

    print(f"\n{'-'*50}")

def test_evo_match():
    """Test the Match class with EvoStrategy players"""
    
    # Test EvoStrategy vs EvoStrategy
    print("\nTesting Match class with two EvoStrategy players")
    turns = 100
    evo_action_history_size = 5
    evo1 = EvoStrategy(name='Evo1', action_history_size=evo_action_history_size)
    
    match_against_evo(evo1, turns, evo_action_history_size)

def test_match():
    """Test the Match class including the EvoStrategy"""
    
    print("\nTesting Match class with one EvoStrategy player")

    turns = 100
    evo_action_history_size = 0
    
    # Alternator vs EvoStrategy
    match_against_evo(Alternator(), turns,  evo_action_history_size)
    
    # TitForTat vs EvoStrategy
    match_against_evo(TitForTat(), turns,  evo_action_history_size)
    
    # Random vs EvoStrategy
    match_against_evo(Random(), turns,  evo_action_history_size)
    
    # FirstByTidemanAndChieruzzi vs EvoStrategy
    match_against_evo(FirstByTidemanAndChieruzzi(), turns,  evo_action_history_size)

def test_tournament():
    """Test the Tournament class with TitForTat, Alternator, Defector, Cooperator, Random, and EvoStrategy"""
    print("\nTesting Tournament class with one EvoStrategy")
    
    # List of actions history sizes to test the EvoStrategy
    action_history_sizes = list(range(0, 12, 2))
    
    for action_history_size in action_history_sizes:
        print(f"\nResults for EvoStrategy with action history size = {action_history_size}")
        players = [
            Cooperator(),
            Defector(),
            Alternator(),
            TitForTat(),
            Random(),
            EvoStrategy(name='Evo', action_history_size=0),
        ]
        tournament = Tournament(players, turns=25, repetitions=3, prob_end=0.01)
        tournament.play(axelrod=True)

        tournament.print_ranked_results()
        
        plot_full_history_evo_player([p for p in players if p.name == 'Evo'][0])
        
    print(f"\n{'-' * 50}")

def evolutionary_axelrod_tournament(num_evo_players_: int = 5,
                                    action_history_size_:int = 10,
                                    generations_:int = 100,
                                    experiment: int = 1, 
                                    save_fig_: bool = True, 
                                    crossover_strategy_: str = "average"):
    """Test the EvolutionaryIterativePrisonersDilemma class"""
    
    print("\nTesting EvolutionaryIterativePrisonersDilemma class")
    
    players = [
        TitForTat(),
        FirstByTidemanAndChieruzzi(),
        FirstByNydegger(),
        FirstByGrofman(),
        FirstByShubik(),
        FirstBySteinAndRapoport(),
        Grudger(),
        FirstByDavis(),
        FirstByGraaskamp(),
        FirstByDowning(),
        FirstByFeld(),
        FirstByJoss(),
        FirstByTullock(),
        FirstByAnonymous(),
        Random(),
    ]

    suffix = f"nep-{num_evo_players_}_ahs-{action_history_size_}_gens-{generations_}" \
            f"_ovr-{crossover_strategy_}_exp-{experiment}"

    evolutionary_ipp = EvolutionaryIterativePrisonersDilemma(players,
                                                             turns=200,
                                                             num_evo_players=num_evo_players_, 
                                                             action_history_size=action_history_size_,
                                                             selected_proportion=0.5,   
                                                             mutation_rate=0.1,
                                                             generations=generations_,
                                                             prob_end=0.01,
                                                             )

    best_evo_player = evolutionary_ipp.train(axelrod=True, crossover_strategy=crossover_strategy_)
    evolutionary_ipp.print_final_results()

    print(f"\n{'-'*50}")

    print("\nTesting Tournament class with the best EvoStrategy")    
    # Run multiple times and plot the full history of the best EvoStrategy player to see the actual behavior
    best_evo_player.reset_full_history()
    tournament = Tournament(players + [best_evo_player],
                            turns=200,
                            repetitions=5, 
                            prob_end=0.01,
                            )
    
    tournament.play(axelrod=True)
    tournament.print_ranked_results()
    
    plot_full_history_evo_player(best_evo_player, save_fig=save_fig_, 
                                 filename=f"best_evo_player_history_{suffix}")
    
    generations_results = evolutionary_ipp.get_generations_results()
    plot_best_players(generations_results, save_fig=save_fig_, filename=f"best_players_{suffix}")
    
    print(f"\n{'-' * 50}")

# Main function to handle argument parsing and running specific tests
def main(args_):
    """
    Main function to run different tests based on user input.
    
    Arguments:
    - args_: Parsed command-line arguments.
    """
    if args_.test == 'match':
        test_match()
    elif args_.test == 'evo_match':
        test_evo_match()
    elif args_.test == 'tournament':
        test_tournament()
    elif args_.test == 'evo_axelrod_tournament':
        evolutionary_axelrod_tournament()
    else:
        print("Invalid test type. Available options are: 'match', 'tournament', 'axelrod_tournament'")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run tests for Match and Tournament classes with EvoStrategy players")
    parser.add_argument(
        '--test', 
        choices=['match', 'evo_match', 'tournament', 'evo_axelrod_tournament'],
        required=True, 
        help="Specify which test to run"
    )
    
    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)