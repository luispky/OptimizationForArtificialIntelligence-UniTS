import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tabulate import tabulate


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)

class RandomNumberGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)

    def random_choice(self, p: float = 0.5):
        if p == 0:
                return 'D'
        elif p == 1:
            return 'C'  
        return 'C' if self.rng.random() < p else 'D'
    
    def random(self, size: int = None):
        """Generate uniform random number(s) in [0,1)."""
        return self.rng.random(size)

    def randint(self, low: int, high: int, size: int = None):
        """Generate random integer(s) in range [low, high)."""
        return self.rng.integers(low, high, size)

    def uniform(self, low: float, high: float, size: int = None):
        """Generate uniform random number(s) in range [low, high)."""
        return self.rng.uniform(low, high, size)

    def normal(self, mean: float, std: float, size: int):
        """Generate normal-distributed random number(s) with mean 0 and std 1."""
        return self.rng.normal(mean, std, size)

    def shuffle(self, x: np.ndarray):
        """Shuffle an array in-place."""
        self.rng.shuffle(x)

    def choice(self, x, size: int = None, replace=True, p=None):
        """Randomly choose element(s) from a given array."""
        return self.rng.choice(x, size=size, replace=replace, p=p)


def get_binary_histories(player):
    """Returns the logged history of an EvoStrategy player and its opponents as binary values.
    
    The actions are converted using the mapping: 'C' -> 0 and 'D' -> 1.
    The '*' symbols (indicating the start of a match) are removed from the histories
    but their positions are recorded and returned.
    """
    action_map = {'C': 0, 'D': 1}
    full_history, opponents_full_history = player.full_history
    
    # Find positions of the '*' markers (match start) in each history
    start_positions = [i for i, action in enumerate(full_history) if action == '*']
    start_positions_opponents = [i for i, action in enumerate(opponents_full_history) if action == '*']
    
    # Remove all '*' markers from the histories
    full_history = [action for action in full_history if action != '*']
    opponents_full_history = [action for action in opponents_full_history if action != '*']
    
    # Convert actions to binary values
    history_binary = [action_map[action] for action in full_history]
    opponents_history_binary = [action_map[action] for action in opponents_full_history]
    
    return history_binary, opponents_history_binary, start_positions, start_positions_opponents


def adjust_xticks(ax, history_length):
    """
    Adjusts the x-ticks of an axis to show a maximum of 8 evenly spaced ticks.
    """
    if history_length > 8:
        step = max(1, history_length // 7)
        xticks = list(range(0, history_length, step))
        # Ensure the last tick is included if it's not too close to the previous tick
        if (history_length - 1) not in xticks:
            if history_length - 1 - xticks[-1] > step // 2:
                xticks.append(history_length - 1)
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(range(history_length))


def add_match_start_lines(ax, positions):
    """
    Adds vertical dashed lines to the axis at each position specified in positions.
    """
    for pos in positions:
        ax.axvline(x=pos, color='black', linestyle='--', alpha=0.5)


def plot_full_history_evo_player(player, save_fig=False, filename=None):
    """
    Plots the binary history of a player's and its opponent's actions as scatter plots 
    in a 2x1 figure. Vertical dashed lines indicate the start of each match.

    - The player's history is plotted in the top subplot (red markers).
    - The opponent's history is plotted in the bottom subplot (blue markers).
    """
    # Get binary histories and match-start positions
    history_binary, opponents_history_binary, start_positions, start_positions_opponents = get_binary_histories(player)
    
    # Create a 2x1 figure with shared x-axis
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # --- Top Subplot: Player's History ---
    ax_player = axes[0]
    ax_player.scatter(range(len(history_binary)), history_binary, color='red', label="Player Actions", s=50)
    add_match_start_lines(ax_player, start_positions)
    adjust_xticks(ax_player, len(history_binary))
    
    ax_player.set_title(f"EvoStrategy {player.name} Actions: ", fontsize=14, pad=15)
    ax_player.set_ylabel("Actions", fontsize=12)
    ax_player.set_yticks([0, 1])
    ax_player.set_yticklabels(['Cooperate', 'Defect'], fontsize=10)
    ax_player.grid(axis='x', linestyle='--', alpha=0.7)
    ax_player.legend(loc='upper right', fontsize=10)
    
    # --- Bottom Subplot: Opponent's History ---
    ax_opponent = axes[1]
    ax_opponent.scatter(range(len(opponents_history_binary)), opponents_history_binary, color='blue', label="Opponent Actions", s=50)
    add_match_start_lines(ax_opponent, start_positions_opponents)
    adjust_xticks(ax_opponent, len(opponents_history_binary))
    
    ax_opponent.set_title("Opponent Actions", fontsize=14, pad=15)
    ax_opponent.set_xlabel("Turns", fontsize=12)
    ax_opponent.set_ylabel("Actions", fontsize=12)
    ax_opponent.set_yticks([0, 1])
    ax_opponent.set_yticklabels(['Cooperate', 'Defect'], fontsize=10)
    ax_opponent.grid(axis='x', linestyle='--', alpha=0.7)
    ax_opponent.legend(loc='upper right', fontsize=10)
    
    # Adjust layout for readability
    plt.tight_layout()
    
    # Save or display the figure
    if save_fig:
        if not filename:
            raise ValueError("Filename must be provided when save_fig is True")
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{filename}.png", bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def print_statistics_evo_player(player) -> None:
    """
    Compute and print EvoStrategy player statistics using tabulate.
    
    Parameters:
        player: EvoStrategy instance.
    """
    history_binary, opponents_history_binary, _, _ = get_binary_histories(player)
    
    # Calculate player's statistics
    cooperation_rate = history_binary.count(0) / len(history_binary) * 100
    defection_rate = history_binary.count(1) / len(history_binary) * 100
    cooperation_defection_ratio = (
        cooperation_rate / defection_rate if defection_rate > 0 else cooperation_rate
    )
    wins, losses, ties = player.log_match_results
    matches = wins + losses + ties
    
    # Calculate opponent's statistics
    cooperation_rate_opponent = opponents_history_binary.count(0) / len(opponents_history_binary) * 100
    defection_rate_opponent = opponents_history_binary.count(1) / len(opponents_history_binary) * 100
    cooperation_defection_ratio_opponent = (
        cooperation_rate_opponent / defection_rate_opponent if defection_rate_opponent > 0 else cooperation_rate_opponent
    )
    
    # Prepare data for tabulate
    player_data = [
        ["Wins", wins],
        ["Losses", losses],
        ["Ties", ties],
        ["Matches", matches],
        ["Cooperation Rate", f"{cooperation_rate:.2f}%"],
        ["Defection Rate", f"{defection_rate:.2f}%"],
        ["Cooperation-Defection Ratio", f"{cooperation_defection_ratio:.2f}"]
    ]
    
    opponent_data = [
        ["Cooperation Rate", f"{cooperation_rate_opponent:.2f}%"],
        ["Defection Rate", f"{defection_rate_opponent:.2f}%"],
        ["Cooperation-Defection Ratio", f"{cooperation_defection_ratio_opponent:.2f}"]
    ]
    
    print(f"\nPlayer {player.name} Statistics in the Tournament:")
    print(tabulate(player_data, tablefmt="github"))
    print("\nOpponents Statistics:")
    print(tabulate(opponent_data, tablefmt="github"))



def plot_best_players(generation_results, save_fig=False, filename=None):
    """
    Plots a scatter plot of the best evolutionary player's rank (lower x-axis)
    and the best player's rank (constant line at 1), with dotted lines for ranks,
    ranks decreasing (inverted y-axis), integer ranks, and a single generation value 
    displayed near the lower x-axis.
    
    :param generation_results: Pandas DataFrame containing columns:
        - 'Generation'
        - 'Best Evo Player'
        - 'Evo Rank'
        - 'Evo Score'
        - 'Best Player'
        - 'Best Score'
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Maximum rank to set y-axis limit
    max_rank = generation_results['Evo Rank'].max()

    # Scatter plot for Evo Player ranks
    ax1.scatter(
        generation_results['Generation'],
        generation_results['Evo Rank'],
        label='Best Evo Player Rank',
        marker='o',
        color='blue',
        zorder=3
    )

    # Line plot with dotted style for Evo Player ranks
    ax1.plot(
        generation_results['Generation'],
        generation_results['Evo Rank'],
        linestyle=':',
        color='blue',
        alpha=0.8
    )

    # Add a constant line at Rank = 1 for the Best Player
    ax1.axhline(y=1, color='orange', linestyle='--', label='Best Player (Rank = 1)', zorder=2)
        
    # Annotate lower x-axis with generation numbers near the tick labels
    for idx, row in generation_results.iterrows():
        ax1.text(
            row['Generation'], 0 + 0.25,  # Slightly above the x-axis
            f"G{row['Generation']}",
            color='black', fontsize=12, ha='center', va='top'
        )

    # Setup the y-axis
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Rank', fontsize=12)
    ax1.set_ylim(max_rank + 1, 0)  # Inverted Y-axis: Rank decreases upward
    ax1.set_yticks(range(1, max_rank + 1))  # Integer ticks only
    ax1.set_title('Rank of Best Players Across Generations', fontsize=12)

    # Set the lower x-axis with Best Evo Player names
    ax1.set_xticks(generation_results['Generation'])
    ax1.set_xticklabels(generation_results['Best Evo Player'], rotation=45, ha='right', fontsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a twin axis for the upper x-axis (Best Player names)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # Synchronize the upper x-axis with the lower
    ax2.set_xticks(generation_results['Generation'])
    ax2.set_xticklabels(generation_results['Best Player'], rotation=45, ha='left', fontsize=12)
    ax2.tick_params(axis='x', labelsize=12)

    # Add a legend for clarity
    ax1.legend(loc='center right', fontsize=12)
    
    # Fix layout
    plt.tight_layout()

    if save_fig:
        if not filename:
            raise ValueError("Filename must be provided when save_fig is True")
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{filename}.png", bbox_inches='tight')
    
    # Show the plot
    plt.show() if not save_fig else plt.close()
