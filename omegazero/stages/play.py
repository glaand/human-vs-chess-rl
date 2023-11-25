from entities.game import Game
from entities.player import LearningPlayer, StockfishPlayer
from entities.brain import Brain

from tqdm import tqdm
import random

class PlayStage:
    def __init__(self, initial_state: str, exploration_prob: float, episode: int):
        self.initial_state = initial_state
        self.exploration_prob = exploration_prob
        self.brain = Brain(episode)
        self.all_move_values = []
        if self.initial_state is None or self.initial_state == "":
            self.initial_state = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        print("")
        print("=====================")
        print("=    PLAY STAGE     =")
        print("=====================")
        print("► Input: Brain with old memories")
        print("► Output: Brain with new memories")
        print(f"► Initial state: {self.initial_state}")
        print("---------------------")

    def play(self, n):
        print(f"- Playing {n} games against stockfish, please wait...")

        self.stockfish_player = StockfishPlayer(self.brain)
        self.learning_player = LearningPlayer(self.brain, self.exploration_prob)

        # @todo: parallelize this
        for i in tqdm(range(n)):
            game = Game(self.initial_state, i)
            white_name = "OmegaZero"
            black_name = "Stockfish"

            # choose random player colors
            if random.random() < 0.5:
                game.setWhitePlayer(self.learning_player)
                game.setBlackPlayer(self.stockfish_player)
            else:
                game.setWhitePlayer(self.stockfish_player)
                game.setBlackPlayer(self.learning_player)
                white_name = "Stockfish"
                black_name = "OmegaZero"

            game.playUntilFinished()
            self.all_move_values.extend(game.move_values)
            self.brain.memory.commit_ltmemory()
            #game.savePGN("play", white_name, black_name)

    def getOutput(self):
        return self.brain