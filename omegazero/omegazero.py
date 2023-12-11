from algorithms.mcts import MCTS
from stages.play import PlayStage
from stages.learn import LearnStage
from stages.evaluate import EvaluateStage
from entities.player import LearningPlayer

import config
import math
import sys

from tqdm import tqdm
import pandas as pd

fen_string = "8/8/3k4/8/3K4/8/8/7R w - - 0 1"
#fen_string = "r2k3r/8/8/8/8/8/8/3K4 w - - 0 1"


def save_game_data(episode, values):
    """
    Save the game data to a CSV file.

    Parameters:
    episode (int): The episode number.
    values (list): A list of tuples containing the game data.

    Returns:
    None
    """
    columns = ['episode', 'iteration', 'move', 'chess_move', 'player', 'color', 'mcts_value', 'nn_value', 'done_found', 'action_prob']

    # add episode to each row
    values = [tuple([episode] + list(row)) for row in values]
    df = pd.DataFrame(values, columns=columns)

    # check if game_data.csv exists
    try:
        game_data = pd.read_csv("game_data.csv")
    except FileNotFoundError:
        game_data = pd.DataFrame(columns=columns)

    game_data = pd.concat([game_data, df], ignore_index=True)
    game_data.to_csv("game_data.csv", index=False)

def main(episode):
    """
    Run the OmegaZero algorithm for a given episode.

    Args:
        episode (int): The episode number.

    Returns:
        None
    """
    print(f"Running omegazero with episode={episode}")

    play_stage = PlayStage(fen_string, episode)
    play_stage.play(n=config.NUM_OF_PLAY_GAMES)
    brain = play_stage.getOutput()
    save_game_data(episode, play_stage.all_move_values)

    learn_stage = LearnStage()
    learn_stage.setInput(brain)
    learn_stage.learn()
    brain = learn_stage.getOutput()

    evaluate_stage = EvaluateStage(fen_string, episode)
    evaluate_stage.evaluate(n=config.NUM_OF_EVAL_GAMES)
    metrics = evaluate_stage.getOutput()
    print(f"Episode {episode} - {metrics}")
    
    # save metrics to log file
    with open("log.txt", "a") as f:
        f.write(f"Episode {episode} - {metrics}\n")


if __name__ == "__main__":
    episode = int(sys.argv[1])
    main(episode)
