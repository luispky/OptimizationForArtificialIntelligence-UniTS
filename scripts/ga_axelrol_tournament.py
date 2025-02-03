import argparse
from typing import Union
from src.strategies import (
    TitForTat,
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
from src.game import Tournament, GAIterativePrisonersDilemma
from src.utils import (
    plot_full_history_evo_player,
    plot_best_players, 
    set_seeds, 
    print_statistics_evo_player
)
import numpy as np


def ga_axelrod_tournament(
    num_evo_players: int = 5,
    action_history_size: int = 10,
    generations: int = 100,
    experiment: int = 1,
    save_fig: bool = True,
    crossover_strategy: str = "adaptive_weighted",
    elitism_proportion: float = 0.1,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.1,
    mutation_rate: float = 0.1,
    noise: float = 0.0,
    prob_end: float = 0.0,
    turns: int = 100,
    test_repetitions: int = 10,
    seed: Union[int, None] = 42,
) -> None:
    """
    Run an Evolutionary Axelrod Tournament.

    Args:
        num_evo_players (int): Number of EvoStrategy players.
        action_history_size (int): History size for EvoStrategy actions.
        generations (int): Number of generations for the evolutionary process.
        experiment (int): Experiment identifier for saving files.
        save_fig (bool): Whether to save plots as files.
        crossover_strategy (str): Strategy used for crossover ('adaptive_weighted', 'BLX-α', 'random_subset').
        elitism_proportion (float): Proportion of top evolutionary players to retain each generation.
        mutation_rate (float): Standard deviation for Gaussian mutation noise.
        mutation_probability (float): Probability of a mutation occurring in an offspring strategy.
        noise (float): Probability of introducing noise in players' actions.
        prob_end (float): Probability of ending a match after each turn.
        turns (int): Maximum number of turns per match.
        test_repetitions (int): Number of repetitions for testing the best EvoStrategy.
        seed (int): Seed for reproducibility (if None, a random seed is used).
    """
    # Set seeds for reproducibility
    # seed = np.random.randint(0, 2**32 - 1) 
    # print(f"Seed: {seed}")
    set_seeds(seed)
    
    # Initialize fixed (non-evolutionary) players
    fixed_players = [
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

    # Create a suffix for file naming based on experiment parameters
    suffix = (
        f"GA"
        f"_exp-{experiment}"
        f"_seed-{seed}"
        f"_nevo-{num_evo_players}"
        f"_ahs-{action_history_size}"
        f"_gens-{generations}"
        f"_xover-{crossover_strategy}"
        f"_xprob-{crossover_probability}"
        f"_mprob-{mutation_probability}"
        f"_mrate-{mutation_rate}"
    )  

    # Initialize the evolutionary tournament with updated parameters
    evolutionary_ipp = GAIterativePrisonersDilemma(
        fixed_players=fixed_players,                
        num_evo_players=num_evo_players,
        elitism_proportion=elitism_proportion,         
        action_history_size=action_history_size,
        crossover_probability=crossover_probability,      
        mutation_probability=mutation_probability,     
        mutation_rate=mutation_rate,             
        turns=turns,                                   
        generations=generations,
        prob_end=prob_end,                             
        noise=noise,                                   
        seed=seed,                                     
    )

    # Train the evolutionary population
    best_evo_player = evolutionary_ipp.train(
        axelrod=True,
        crossover_strategy=crossover_strategy
    )
    best_evo_player.save_strategy(suffix)
    
    # Print the best evolutionary player's final genome
    print(f"\n {best_evo_player.name} genome:\n" 
          f"{np.round(best_evo_player.weights, 3)}\n")

    # Print final ranked results
    evolutionary_ipp.print_final_results()

    print("\nTesting Tournament with the best EvoStrategy")
    best_evo_player.reset_full_history()
    best_evo_player.log_history = True
    
    # Initialize a tournament including the best evolutionary player
    tournament = Tournament(
        players=fixed_players + [best_evo_player],
        turns=turns,
        repetitions=test_repetitions,
        prob_end=prob_end,
    )
    tournament.play(axelrod=True)
    tournament.print_ranked_results()
    tournament.save_ranked_results(suffix)

    # Plotting the full history of the best evolutionary player
    plot_full_history_evo_player(
        best_evo_player,
        save_fig=save_fig,
        filename=f"best_evo_player_history_{suffix}"
    )
    
    print_statistics_evo_player(best_evo_player)

    # Retrieve and plot the generation-by-generation results
    generations_results = evolutionary_ipp.get_generations_results()
    plot_best_players(
        generations_results,
        save_fig=save_fig,
        filename=f"best_players_{suffix}"
    )

    print("\n" + "-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evolutionary Axelrod Tournament")

    # Evolutionary parameters
    parser.add_argument("--num_evo_players", type=int, default=5, help="Number of EvoStrategy players")
    parser.add_argument("--action_history_size", type=int, default=10, help="EvoStrategy history size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--elitism_proportion", type=float, default=0.1, help="Proportion of top evolutionary players to retain each generation")
    parser.add_argument("--crossover_strategy", type=str, default="adaptive_weighted", help="Crossover strategy ('adaptive_weighted', 'BLX-α', 'random_subset')")
    parser.add_argument("--crossover_probability", type=float, default=0.95, help="Probability of crossover occurring between two parents")
    parser.add_argument("--mutation_probability", type=float, default=0.1, help="Probability of a mutation occurring in an offspring strategy")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Standard deviation for Gaussian mutation noise")

    # Tournament parameters
    parser.add_argument("--turns", type=int, default=200, help="Maximum number of turns per match")
    parser.add_argument("--prob_end", type=float, default=0.0, help="Probability of ending a match after each turn")
    parser.add_argument("--noise", type=float, default=0.0, help="Probability of introducing noise in players' actions")

    # Experiment parameters
    parser.add_argument("--experiment", type=int, default=1, help="Experiment identifier for saving files")
    parser.add_argument("--test_repetitions", type=int, default=10, help="Number of repetitions for testing the best EvoStrategy")
    parser.add_argument("--save_fig", type=bool, default=True, help="Save figures as files")
    parser.add_argument("--seed", type=int, default=3487522376, help="Seed for reproducibility (if None, a random seed is used)")

    args = parser.parse_args()

    ga_axelrod_tournament(
        num_evo_players=args.num_evo_players,
        action_history_size=args.action_history_size,
        generations=args.generations,
        experiment=args.experiment,
        save_fig=args.save_fig,
        crossover_strategy=args.crossover_strategy,
        elitism_proportion=args.elitism_proportion,
        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,
        mutation_rate=args.mutation_rate,
        test_repetitions=args.test_repetitions,
        noise=args.noise,
        prob_end=args.prob_end,
        turns=args.turns,
        seed=args.seed,
    )
