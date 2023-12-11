import pandas as pd
import matplotlib.pyplot as plt

def plot_game_data():
    """
    Plots the Q-Values over episodes for different players and colors.

    Reads the game data from a CSV file and plots the MCTS and NN values for each player and color combination.
    The plot is divided into four subplots, each representing a player and color combination.

    Args:
        None

    Returns:
        None
    """
    df = pd.read_csv('game_data.csv')

    df['sequential_move'] = df['episode'] * df['iteration'].max() * df['move'].max() + df['iteration'] * df['move'] + df['move']
    df = df.sort_values(by=['sequential_move'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Q-Values over episodes')

    # Top Left: MCTS and NN values for white player learning
    white_learning_df = df[(df['player'] == 'learning') & (df['color'] == 'white')]
    axes[0, 0].plot(white_learning_df['sequential_move'], white_learning_df['mcts_value'], label='MCTS Values')
    axes[0, 0].plot(white_learning_df['sequential_move'], white_learning_df['nn_value'], label='NN Values')
    axes[0, 0].set_title('White Player Learning')
    axes[0, 0].set_xlabel('Move Number')
    axes[0, 0].set_ylabel('Values')
    axes[0, 0].legend()

    # Top Right: MCTS and NN values for black player stockfish
    black_stockfish_df = df[(df['player'] == 'stockfish') & (df['color'] == 'black')]
    axes[0, 1].plot(black_stockfish_df['sequential_move'], black_stockfish_df['mcts_value'], label='MCTS Values')
    axes[0, 1].plot(black_stockfish_df['sequential_move'], black_stockfish_df['nn_value'], label='NN Values')
    axes[0, 1].set_title('Black Player Stockfish')
    axes[0, 1].set_xlabel('Move Number')
    axes[0, 1].set_ylabel('Values')
    axes[0, 1].legend()

    # Bottom Left: White player stockfish
    white_stockfish_df = df[(df['player'] == 'stockfish') & (df['color'] == 'white')]
    axes[1, 0].plot(white_stockfish_df['sequential_move'], white_stockfish_df['mcts_value'], label='MCTS Values')
    axes[1, 0].plot(white_stockfish_df['sequential_move'], white_stockfish_df['nn_value'], label='NN Values')
    axes[1, 0].set_title('White Player Stockfish')
    axes[1, 0].set_xlabel('Move Number')
    axes[1, 0].set_ylabel('Values')
    axes[1, 0].legend()

    # Bottom Right: Black player learning
    black_learning_df = df[(df['player'] == 'learning') & (df['color'] == 'black')]
    axes[1, 1].plot(black_learning_df['sequential_move'], black_learning_df['mcts_value'], label='MCTS Values')
    axes[1, 1].plot(black_learning_df['sequential_move'], black_learning_df['nn_value'], label='NN Values')
    axes[1, 1].set_title('Black Player Learning')
    axes[1, 1].set_xlabel('Move Number')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'q_values_over_episodes.png')

def plot_loss_data():
    """
    Plots the loss over epochs.

    Reads the loss data from a CSV file and plots the loss values against the sequential move.
    The sequential move is calculated as the product of the episode and the maximum epoch value,
    plus the epoch value. The data is sorted by the sequential move before plotting.

    Saves the plot as 'loss_over_epochs.png'.
    """
    df = pd.read_csv('loss_data.csv')

    df['sequential_move'] = df['episode'] * df['epoch'].max() + df['epoch']
    df = df.sort_values(by=['sequential_move'])

    # plot loss over sequential move
    plt.figure()
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs ordered by episodes")
    plt.ylabel("Loss")
    plt.plot(df['sequential_move'], df['loss'])
    plt.savefig(f'loss_over_epochs.png')

if __name__ == "__main__":
    plot_game_data()
    plot_loss_data()