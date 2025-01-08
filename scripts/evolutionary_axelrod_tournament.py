import argparse
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
from src.game import Tournament, EvolutionaryIterativePrisonersDilemma
from src.utils import plot_full_history_evo_player, plot_best_players

def evolutionary_axelrod_tournament(
    num_evo_players: int = 5,
    action_history_size: int = 10,
    generations: int = 100,
    experiment: int = 1,
    save_fig: bool = True,
    crossover_strategy: str = "average",
) -> None:
    """
    Run an Evolutionary Axelrod Tournament.

    Args:
        num_evo_players (int): Number of EvoStrategy players.
        action_history_size (int): History size for EvoStrategy actions.
        generations (int): Number of generations for the evolutionary process.
        experiment (int): Experiment identifier for saving files.
        save_fig (bool): Whether to save plots as files.
        crossover_strategy (str): Strategy used for crossover ('average', 'random', etc.).
    """
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

    suffix = (
        f"nep-{num_evo_players}_ahs-{action_history_size}_gens-{generations}"
        f"_ovr-{crossover_strategy}_exp-{experiment}"
    )

    evolutionary_ipp = EvolutionaryIterativePrisonersDilemma(
        players,
        turns=200,
        num_evo_players=num_evo_players,
        action_history_size=action_history_size,
        selected_proportion=0.5,
        mutation_rate=0.1,
        generations=generations,
        prob_end=0.01,
    )

    best_evo_player = evolutionary_ipp.train(axelrod=True, crossover_strategy=crossover_strategy)
    evolutionary_ipp.print_final_results()

    print("\nTesting Tournament with the best EvoStrategy")
    best_evo_player.reset_full_history()
    tournament = Tournament(
        players + [best_evo_player],
        turns=200,
        repetitions=5,
        prob_end=0.01,
    )
    tournament.play(axelrod=True)
    tournament.print_ranked_results()
    tournament.save_ranked_results(suffix)

    plot_full_history_evo_player(
        best_evo_player, save_fig=save_fig, filename=f"best_evo_player_history_{suffix}"
    )
    
    generations_results = evolutionary_ipp.get_generations_results()
    plot_best_players(generations_results, save_fig=save_fig, filename=f"best_players_{suffix}")
    print("\n" + "-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evolutionary Axelrod Tournament")
    parser.add_argument("--num_evo_players", type=int, default=5, help="Number of EvoStrategy players")
    parser.add_argument("--action_history_size", type=int, default=10, help="EvoStrategy history size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--experiment", type=int, default=1, help="Experiment identifier")
    parser.add_argument("--save_fig", type=bool, default=True, help="Save figures as files")
    parser.add_argument("--crossover_strategy", type=str, default="adaptive_weighted", help="Crossover strategy")
    args = parser.parse_args()

    evolutionary_axelrod_tournament(
        num_evo_players=args.num_evo_players,
        action_history_size=args.action_history_size,
        generations=args.generations,
        experiment=args.experiment,
        save_fig=args.save_fig,
        crossover_strategy=args.crossover_strategy,
    )
