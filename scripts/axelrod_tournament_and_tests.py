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
)
from src.game import Match, Tournament
import numpy as np
import axelrod as axl
import argparse

def test_match():
    """Test the Match class"""
    print("\nTesting Match class")
    
    print(f"\nTitForTat vs Alternator")
    players = [TitForTat(), Alternator()]
    match = Match(players, turns=6, prob_end=0.01, seed=42)
    moves = match.play()
    print(f"\nMoves: \n {moves}")
    print(f"Winner: {match.winner}")
    print(f"Final scores: {match.final_scores}")
    

    print(f"\n TitForTat vs Random")
    players = [TitForTat(), Random()]
    match = Match(players, turns=6, prob_end=0.01, seed=42)
    moves = match.play()
    print(f"\nMoves: \n {moves}")
    print(f"Winner: {match.winner}")
    print(f"Final scores: {match.final_scores}")
    
    print(f"\n{'-'*75}")


def test_match_axl_library():
    """Test the Match class from Axelrod library"""
    print("\nMatch class from Axelrod library")
    
    print(f"\nTitForTat vs Alternator")
    players = [axl.TitForTat(), axl.Alternator()]
    match = axl.Match(tuple(players), turns=6, prob_end=0.01, seed=42)
    moves = match.play()
    print(f"\nMoves: \n {moves}")
    print(f"Winner: {match.winner()}")
    print(f"Final scores: {match.final_score()}")


    print(f"\nTitForTat vs Random")
    players = [axl.TitForTat(), axl.Random()]
    match = axl.Match(tuple(players), turns=6, prob_end=0.01, seed=42)
    moves = match.play()
    print(f"\nMoves: \n {moves}")
    print(f"Winner: {match.winner()}")
    print(f"Final scores: {match.final_score()}")
    
    print(f"\n{'-'*75}")


def test_tournament():
    """Test the Tournament class"""
    print("\nTesting Tournament class")

    repetitions = 5 # Makes sense only with stochastic strategies and fixed seed
    turns = 200
    seed = 42

    players = [
        Cooperator(),
        Defector(),
        TitForTat(),
        Alternator(),
        Random(), 
    ]
    tournament = Tournament(players, turns=turns, repetitions=repetitions, seed=seed)
    tournament.play()
    
    tournament.print_ranked_results()
    
    print("\n Per Match Results")
    for player, scores in tournament.scores.items():
        print(f"\n{player}: {scores}")
    
    print(f"\n{'-'*75}")


def test_tournament_axl_library():
    """Test the Tournament class from Axelrod library"""
    print("\nTesting Tournament class from Axelrod library")
    
    repetitions = 5
    turns = 200
    seed = 42
    
    players = [
        axl.Cooperator(),
        axl.Defector(),
        axl.TitForTat(),
        axl.Alternator(), 
        axl.Random(),
    ]
    tournament = axl.Tournament(players, turns=turns, repetitions=repetitions, seed=seed)
    results = tournament.play(progress_bar=False)
    
    # Calculate the mean results
    mean_results = np.array(results.scores).mean(axis=1)/(len(players)-1)
    mean_results = np.sort(mean_results)[::-1]
    
    # Create a dictionary of sorted mean results
    sorted_mean_results = {
        name: mean_results[i] 
        for i, name in enumerate(results.ranked_names)
    }
    
    print(f'{"Rank":>5} {"Name":>45} {"Score":>8}')
    for i, (name, score) in enumerate(sorted_mean_results.items(), start=1):
        print(f"{i:>5} {name:>45} {score:>8.2f}")
    
    print("-" * 75)


def axelrod_tournament():
    """Run Axelrod tournament with custom implementations"""
    print("\nRunning Axelrod tournament with custom implementations")
    turns = 200
    repetitions = 5
    prob_end = 0.01
    seed = 42
    
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
    
    tournament = Tournament(players, turns=turns, repetitions=repetitions, prob_end=prob_end, seed=seed)
    tournament.play(axelrod=True)
    
    tournament.print_ranked_results()
        
    print(f"\n{'-'*75}")
    
def axelrod_tournament_axl_library():
    """Run Axelrod tournament with Axelrod library implementations"""
    print("\nRunning Axelrod tournament with Axelrod library implementations")
    
    turns = 200
    repetitions = 5
    prob_end = 0.01
    seed = 42
    
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
    tournament = axl.Tournament(players, turns=turns, repetitions=repetitions, prob_end=prob_end, seed=seed)
    
    # Play the tournament and capture the results
    results = tournament.play(progress_bar=False)
    
    # Calculate the mean results
    mean_results = np.array(results.scores).mean(axis=1)/(len(players)-1)
    mean_results = np.sort(mean_results)[::-1]
    
    # Create a dictionary of sorted mean results
    sorted_mean_results = {
        name: mean_results[i] 
        for i, name in enumerate(results.ranked_names)
    }
    
    print(f'{"Rank":>5} {"Name":>45} {"Score":>8}')
    for i, (name, score) in enumerate(sorted_mean_results.items(), start=1):
        print(f"{i:>5} {name:>45} {score:>8.2f}")
    
    # Plot the results using a boxplot
    plot = axl.Plot(results)
    p = plot.boxplot()
    p.show()
    
    print(f"\n{'-'*75}")

    input("\nPress Enter to close the plot")


# Main function to handle argument parsing and running specific tests
def main(args_):
    """
    Main function to run different tests based on user input.
    
    Arguments:
    - args_: Parsed command-line arguments.
    """
    if args_.test == 'match':
        test_match()
        test_match_axl_library()
    elif args_.test == 'tournament':
        test_tournament()
        test_tournament_axl_library()
    elif args_.test == 'axelrod_tournament':
        axelrod_tournament()
        axelrod_tournament_axl_library()
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