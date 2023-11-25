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

fen_string = "4k3/8/8/8/8/8/R7/3K3R w - - 0 1"
#fen_string = "r2k3r/8/8/8/8/8/8/3K4 w - - 0 1"

def exploration_decay(exploration_prob, episode, episodes, decay_factor=0.1):
    decayed_prob = exploration_prob * math.exp(-decay_factor * episode / episodes)
    return decayed_prob

def save_game_data(episode, values):
    columns = ['episode', 'iteration', 'move', 'player', 'color', 'mcts_value', 'nn_value']

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

def main(episode, exploration_prob):
    print(f"Running omegazero with episode={episode} and exploration_prob={exploration_prob}")

    play_stage = PlayStage(fen_string, exploration_prob, episode)
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
    exploration_prob = float(sys.argv[2])
    main(episode, exploration_prob)
