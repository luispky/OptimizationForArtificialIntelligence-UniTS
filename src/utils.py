import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    
    def random(self):
        return self.rng.random()
    
    def randint(self, low: int, high: int):
        return self.rng.integers(low, high)
    
    def uniform(self, low: float, high: float):
        return self.rng.uniform(low, high)
    
    def rand(self, size: int):
        return self.rng.normal(0, 1, size=size)

    def shuffle(self, x: np.ndarray):
        self.rng.shuffle(x)
    
    def choice(self, x, size: int, replace=True, p=None):
        return self.rng.choice(x, size=size, replace=replace, p=p)


def plot_full_history_evo_player(player, save_fig=False, filename=None):
    """Plots the binary history of a player's actions in the last match as a scatter plot."""
    
    # Map actions to binary values
    action_map = {'C': 0, 'D': 1}
    full_history = player.get_full_history()
    history_binary = [action_map[action] for action in full_history]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(range(len(history_binary)), history_binary, color='red', label="Actions", s=50)

    # Adjust x-ticks: show max 8 evenly spaced ticks
    if len(history_binary) > 8:
        step = max(1, len(history_binary) // 7)  # Divide into 7 intervals for 8 ticks
        xticks = list(range(0, len(history_binary), step))  # Generate ticks
        
        # Ensure the last tick is included if it's not too close to the previous tick
        if len(history_binary) - 1 not in xticks:
            if len(history_binary) - 1 - xticks[-1] > step // 2:  # Add only if far enough
                xticks.append(len(history_binary) - 1)
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(range(len(history_binary)))

    # Add labels and title
    ax.set_title(f"Player Action History: {player.name}", fontsize=14, pad=15)
    ax.set_xlabel("Turns", fontsize=12)
    ax.set_ylabel("Actions", fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Cooperate', 'Defect'], fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add grid for readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add a legend
    ax.legend(loc='upper right', fontsize=10)

    # Fix layout
    plt.tight_layout()

    if save_fig:
        if not filename:
            raise ValueError("Filename must be provided when save_fig is True")
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{filename}.png", bbox_inches='tight')
    
    # Show the plot
    plt.show() if not save_fig else plt.close()


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
    

def ranked_scores_from_average_scores(average_scores: dict[str, float]) -> pd.DataFrame:
    """
    Converts a dictionary of average scores to a DataFrame with ranks.
    
    :param average_scores: Dictionary with player names as keys and average scores as values.
    :return: DataFrame with ranks based on sorted scores.
    """
    
    ranked_results = pd.DataFrame(
        list(average_scores.items()), columns=["Player", "Score"]
    )
    ranked_results = ranked_results.sort_values(
        by="Score", ascending=False
    ).reset_index(drop=True)
    
    # Add a rank column based on sorted scores.
    ranked_results["Rank"] = ranked_results.index + 1
    ranked_results = ranked_results[["Rank", "Player", "Score"]]
    
    return ranked_results
