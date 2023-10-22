
import random
from tqdm import tqdm

from entities.game import Game

class EvaluateStage:
    def __init__(self, initial_state: str, episode):
        self.best_player = None
        self.metrics = None
        self.initial_state = initial_state
        self.episode = episode
        print("")
        print("=====================")
        print("=  EVALUATE STAGE   =")
        print("=====================")
        print("► Input: New Player")
        print("► Output: Best player & metrics")
        print("---------------------")

    def evaluate(self, n, win_quote):
        print(f"- Evaluating by playing {n} games against best player, please wait...")

        self.metrics = {
            "wins": 0,
            "losses_or_draw": 0,
        }

        # @todo: parallelize this
        for i in tqdm(range(n)):
            game = Game(self.initial_state)

            # choose random player colors
            if random.random() < 0.5:
                game.setWhitePlayer(self.best_player)
                game.setBlackPlayer(self.new_player)
            else:
                game.setWhitePlayer(self.new_player)
                game.setBlackPlayer(self.best_player)

            game.playUntilFinished()
            game.savePGN(f"eval_{self.episode}")

            # get the result
            result = game.gameState.board.result()
            # check if the result is a win
            if result == "1-0" and self.new_player.id == game.whitePlayer.id:
                self.metrics["wins"] += 1
            elif result == "0-1" and self.new_player.id == game.blackPlayer.id:
                self.metrics["wins"] += 1
            else:
                self.metrics["losses_or_draw"] += 1

        if self.metrics["wins"] / n > win_quote:
            print(f"- New player is better than best player, replacing best player")
            self.best_player = self.new_player
            self.best_player.setAsBestPlayer()
    
    def setInput(self, new_player, best_player):
        self.new_player = new_player
        self.best_player = best_player

    def getOutput(self):
        return self.best_player, self.metrics
