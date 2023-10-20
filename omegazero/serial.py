from stages.play import PlayStage
from stages.learn import LearnStage
from stages.evaluate import EvaluateStage
from entities.player import LearningPlayer

import config
import math
from tqdm import tqdm

fen_string = "r3b2k/4r1b1/P5p1/6N1/2p1P1BP/2q2R2/2Q2RK1/8 w - - 6 52"

def main():
    print("Running omegazero in serial mode")

    episodes = config.EPISODES
    exploration_prob = config.INITIAL_EXPLORATION
    is_best_player = True
    best_player = LearningPlayer(exploration_prob, is_best_player)

    for episode in tqdm(range(episodes)):
        play_stage = PlayStage(fen_string, exploration_prob)
        play_stage.setInput(best_player)
        play_stage.play(n=config.NUM_OF_PLAY_GAMES)
        memory = play_stage.getOutput()

        learn_stage = LearnStage(episode)
        learn_stage.setInput(memory)
        learn_stage.learn()
        new_player = learn_stage.getOutput()

        evaluate_stage = EvaluateStage(fen_string)
        evaluate_stage.setInput(new_player, best_player)
        evaluate_stage.evaluate(n=config.NUM_OF_EVAL_GAMES, win_quote=config.WIN_QUOTE)
        best_player, metrics = evaluate_stage.getOutput()
        print(f"Episode {episode} - {metrics}")

        exploration_prob = exploration_prob * math.exp(-(1 - episode / episodes) * episode)

if __name__ == "__main__":
    main()
