from stages.play import PlayStage
from stages.learn import LearnStage
from stages.evaluate import EvaluateStage
from entities.player import LearningPlayer

import config
import math
from tqdm import tqdm

fen_string = "4k3/3ppp2/8/8/8/8/2PPPPP1/4K3 w - - 0 1"

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

        evaluate_stage = EvaluateStage(fen_string, episode)
        evaluate_stage.setInput(new_player, best_player)
        evaluate_stage.evaluate(n=config.NUM_OF_EVAL_GAMES, win_quote=config.WIN_QUOTE)
        best_player, metrics = evaluate_stage.getOutput()
        print(f"Episode {episode} - {metrics}")
        
        # save metrics to log file
        with open("log.txt", "a") as f:
            f.write(f"Episode {episode} - {metrics}\n")

        exploration_prob = exploration_prob * math.exp(-(1 - episode / episodes) * episode)

if __name__ == "__main__":
    main()
