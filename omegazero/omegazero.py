from algorithms.mcts import MCTS
from stages.play import PlayStage
from stages.learn import LearnStage
from stages.evaluate import EvaluateStage
from entities.player import LearningPlayer

import psutil
import gc
import config
import math
from tqdm import tqdm

fen_string = "4kn2/8/8/8/8/8/4P3/R3K3 w - - 0 1"

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()

    print(f"Memory usage: {memory_info.rss / (1024 * 1024 * 1024):.2f} GB")

def exploration_decay(exploration_prob, episode, episodes, decay_factor=0.1):
    decayed_prob = exploration_prob * math.exp(-decay_factor * episode / episodes)
    return decayed_prob

def main():
    print("Running omegazero in serial mode")

    episodes = config.EPISODES
    exploration_prob = config.INITIAL_EXPLORATION

    for episode in tqdm(range(episodes)):
        play_stage = PlayStage(fen_string, exploration_prob)
        play_stage.play(n=config.NUM_OF_PLAY_GAMES)
        brain = play_stage.getOutput()

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

        exploration_prob = exploration_decay(exploration_prob, episode, episodes)
        print(f"Exploration probability: {exploration_prob}")

        # free memory
        print("Freeing memory...")
        print(f"Before garbage collection: {gc.get_count()}")
        print_memory_usage()

        gc.collect()

        print(f"After garbage collection: {gc.get_count()}")
        print_memory_usage()


if __name__ == "__main__":
    main()
