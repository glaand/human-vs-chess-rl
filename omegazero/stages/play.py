from entities.game import Game
from entities.player import LearningPlayer, StockfishPlayer

from tqdm import tqdm
import random

class PlayStage:
    def __init__(self, initial_state: str, exploration_prob: float):
        self.initial_state = initial_state
        self.exploration_prob = exploration_prob
        if self.initial_state is None or self.initial_state == "":
            self.initial_state = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        print("")
        print("=====================")
        print("=    PLAY STAGE     =")
        print("=====================")
        print("► Input: Best player")
        print("► Output: Game State Tensor")
        print(f"► Initial state: {self.initial_state}")
        print("---------------------")

    def play(self, n):
        print(f"- Playing {n} games against stockfish, please wait...")

        self.stockfish_player = StockfishPlayer()
        self.learning_player = LearningPlayer(self.exploration_prob)

        # @todo: parallelize this
        for i in tqdm(range(n)):
            game = Game(self.initial_state)

            # choose random player colors
            if random.random() < 0.5:
                game.setWhitePlayer(self.learning_player)
                game.setBlackPlayer(self.stockfish_player)
            else:
                game.setWhitePlayer(self.stockfish_player)
                game.setBlackPlayer(self.learning_player)

            game.playUntilFinished()
            self.learning_player.brain.memory.commit_ltmemory()
            game.savePGN()

    def setInput(self, best_player):
        self.best_player = best_player

    def getOutput(self):
        return self.learning_player.brain.memory