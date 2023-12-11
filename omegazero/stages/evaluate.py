
import random
from tqdm import tqdm

from entities.game import Game
from entities.player import StockfishPlayer, LearningPlayer
from entities.brain import Brain

class EvaluateStage:
    def __init__(self, initial_state: str, episode):
        """
        Initializes the EvaluateStage object.

        Args:
            initial_state (str): The initial state of the game.
            episode: The episode number.

        Attributes:
            metrics (dict): A dictionary to store the evaluation metrics.
            initial_state (str): The initial state of the game.
            episode: The episode number.
        """
        self.metrics = None
        self.initial_state = initial_state
        self.episode = episode
        print("")
        print("=====================")
        print("=  EVALUATE STAGE   =")
        print("=====================")
        print("► Input: New trained brain")
        print("► Output: Metrics")
        print("---------------------")

    def evaluate(self, n):
        """
        Evaluates the trained brain by playing a specified number of games against Stockfish.

        Args:
            n (int): The number of games to play.

        Returns:
            None
        """
        print(f"- Evaluating by playing {n} games against stockfish, please wait...")

        self.metrics = {
            "wins": 0,
            "losses": 0,
            "draw": 0,
        }

        new_trained_brain = Brain(episode=self.episode, is_play_stage=False)

        # @todo: parallelize this
        for i in tqdm(range(n)):
            game = Game(self.initial_state, i)

            exploration_prob = 0
            game.setWhitePlayer(LearningPlayer(new_trained_brain, exploration_prob))
            game.setBlackPlayer(StockfishPlayer(new_trained_brain))

            game.playUntilFinished()
            
            # save if last game
            if i == n-1:
                game.savePGN(f"eval_ep_{self.episode}", "OmegaZero", "Stockfish")

            # get the result
            result = game.gameState.board.result()
            # check if the result is a win
            if result == "1-0":
                self.metrics["wins"] += 1
            elif result == "0-1":
                self.metrics["losses"] += 1
            elif result == "1/2-1/2":
                self.metrics["draw"] += 1
            else:
                self.metrics["losses"] += 1

    def getOutput(self):
        """
        Returns the evaluation metrics.

        Returns:
            dict: The evaluation metrics.
        """
        return self.metrics
