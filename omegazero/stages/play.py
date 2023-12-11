from entities.game import Game
from entities.player import LearningPlayer, StockfishPlayer
from entities.brain import Brain
import config

from tqdm import tqdm
import random
import math

class PlayStage:
    """
    Represents the play stage of the game, where OmegaZero plays against Stockfish.

    Attributes:
        initial_state (str): The initial state of the game.
        exploration_prob (float): The exploration probability for OmegaZero.
        decay_factor (float): The decay factor for the exploration probability.
        brain (Brain): The brain object used by OmegaZero.
        all_move_values (list): A list to store all move values during the games.

    Methods:
        exploration_decay(current_game, num_of_games): Calculates the decayed exploration probability.
        play(n): Plays n games against Stockfish.
        getOutput(): Returns the brain object.

    """

    def __init__(self, initial_state: str, episode: int):
        self.initial_state = initial_state
        self.exploration_prob = config.INITIAL_EXPLORATION
        self.decay_factor = config.DECAY_FACTOR
        self.brain = Brain(episode=episode, is_play_stage=True)
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

    def exploration_decay(self, current_game, num_of_games):
        """
        Calculates the decayed exploration probability based on the current game and total number of games.

        Args:
            current_game (int): The current game number.
            num_of_games (int): The total number of games.

        Returns:
            float: The decayed exploration probability.

        """
        decayed_prob = self.exploration_prob * math.exp(-self.decay_factor * current_game / num_of_games)
        return decayed_prob

    def play(self, n):
        """
        Plays n games against Stockfish.

        Args:
            n (int): The number of games to play.

        """
        print(f"- Playing {n} games against stockfish, please wait...")

        self.stockfish_player = StockfishPlayer(self.brain)
        self.learning_player = LearningPlayer(self.brain, self.exploration_prob)

        # @todo: parallelize this
        for i in tqdm(range(n)):
            self.exploration_prob = self.exploration_decay(i, n)
            self.learning_player.exploration_prob = self.exploration_prob

            game = Game(self.initial_state, i)
            game.setWhitePlayer(self.learning_player)
            game.setBlackPlayer(self.stockfish_player)

            white_name = "OmegaZero"
            black_name = "Stockfish"

            game.playUntilFinished()
            self.all_move_values.extend(game.move_values)
            self.brain.memory.commit_ltmemory()

            #game.savePGN("play", white_name, black_name)

    def getOutput(self):
        """
        Returns the brain object.

        Returns:
            Brain: The brain object.

        """
        return self.brain
