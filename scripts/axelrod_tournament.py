from src.strategies import (TitFortat,
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
)
from src.game import Match, Tournament
import numpy as np
import axelrod as axl
import argparse

def test_match():
    """Test the Match class"""
    print("\nTesting Match class")
    players = [Alternator(), TitFortat()]
    match = Match(players, turns=6, prob_end=0.1, seed=42)
    moves = match.play()
    print(f"\nMoves: \n {moves}")
    print(f"Winner: {match.winner}")
    print(f"Final scores: {match.final_scores}")
    
    print(f"\n{"-"*50}")
    
def test_match_library():
    """Test the Match class from Axelrod library"""
    
    print("\nTesting Match class from Axelrod library")
    players = [axl.Alternator(), axl.TitForTat()]
    match = axl.Match(players, turns=6, prob_end=0.1, seed=42)
    moves = match.play()
    print(f"\nMoves: \n {moves}")
    print(f"Winner: {match.winner()}")
    print(f"Final scores: {match.final_score()}")
    
    print(f"\n{"-"*50}")
    
def test_tournament():
    """Test the Tournament class"""
    print("\nTesting Tournament class")
    players = [
        Cooperator(),
        Defector(),
        TitFortat(),
        Alternator(), 
        # Random(p=0.5)
    ]
    tournament = Tournament(players, turns=10, repetitions=5, noise=0.0, prob_end=0.0, seed=42)
    tournament.play()
    
    print(f"\nRanked results:")
    for player, score in tournament.get_ranked_results().items():
        print(f"{player}: {score}")
        
    print(f"\n{"-"*50}")
    
def test_tournament_library():
    """Test the Tournament class from Axelrod library"""
    print("\nTesting Tournament class from Axelrod library")
    players = [
        axl.Cooperator(),
        axl.Defector(),
        axl.TitForTat(),
        axl.Alternator(), 
        # axl.Random(p=0.5)
    ]
    tournament = axl.Tournament(players, turns=10, repetitions=5, noise=0.0, prob_end=0.0, seed=42)
    results = tournament.play()

    mean_results = np.array(results.scores).mean(axis=1)
    sorted_mean_results = {name: mean_results[i] for name,i in zip(results.ranked_names, results.ranking)}
    for name, score in sorted_mean_results.items():
        print(f"{name}: {score:.2f}")
        
    print(f"\n{"-"*50}")

def axelrod_tournament():
    """Run Axelrod tournament with custom implementations"""
    print("\nRunning Axelrod tournament with custom implementations")
    players = [
        TitFortat(),
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
    
    tournament = Tournament(players, turns=200, repetitions=5, noise=0.0, prob_end=0.0, seed=42)
    tournament.play()
    
    print(f"\nRanked results:")
    for player, score in tournament.get_ranked_results().items():
        print(f"{player}: {score}")
        
    print(f"\n{"-"*50}")
    
def axelrod_tournament_library():
    """Run Axelrod tournament with Axelrod library implementations"""
    print("\nRunning Axelrod tournament with Axelrod library implementations")
    players = [
        axl.TitForTat(),
        axl.FirstByTidemanAndChieruzzi(),
        axl.FirstByNydegger(),
        axl.FirstByGrofman(),
        axl.FirstByShubik(),
        axl.FirstBySteinAndRapoport(),
        axl.Grudger(),
        axl.FirstByDavis(),
        axl.FirstByGraaskamp(),
        axl.FirstByDowning(),
        axl.FirstByFeld(),
        axl.FirstByJoss(),
        axl.FirstByTullock(),
        axl.FirstByAnonymous(),
        axl.Random()
    ]
    # Initialize the axelrod Tournament with the specified parameters
    tournament = axl.Tournament(players, turns=200, repetitions=5, noise=0.0, prob_end=0.0, seed=42)
    
    # Play the tournament and capture the results
    results = tournament.play()
    
    # Calculate the mean results for each player
    mean_results = np.array(results.scores).mean(axis=1)
    
    # Sort and print the players and their average scores
    sorted_mean_results = {name: mean_results[i] for name, i in zip(results.ranked_names, results.ranking)}
    for name, score in sorted_mean_results.items():
        print(f"{name}: {score:.2f}")
    
    # Plot the results using a boxplot
    plot = axl.Plot(results)
    p = plot.boxplot()
    p.show()
    input("Press Enter to close the plot")
    
    print(f"\n{"-"*50}")

# Main function to handle argument parsing and running specific tests
def main(args):
    """
    Main function to run different tests based on user input.
    
    Arguments:
    - args: Parsed command-line arguments.
    """
    if args.test == 'match':
        test_match()
        test_match_library()
    elif args.test == 'tournament':
        test_tournament()
        test_tournament_library()
    elif args.test == 'axelrod_tournament':
        axelrod_tournament()
        axelrod_tournament_library()
    else:
        print("Invalid test type. Available options are: 'match', 'tournament', 'axelrod_tournament'")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run tests for Match and Tournament classes")
    parser.add_argument(
        '--test', 
        choices=['match', 'tournament', 'axelrod_tournament'],
        required=True, 
        help="Specify which test to run"
    )
    
    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)