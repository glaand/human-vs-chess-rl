from stockfish import Stockfish
import chess
import os
import random
import numpy as np

from .brain import Brain

artifacts_path = os.path.join(os.path.dirname(__file__), "..", "artifacts")

class Player:
    def makeMove(self, game):
        pass

class LearningPlayer(Player):
    """
    Represents a learning player in the game.

    Attributes:
        type (str): The type of the player.
        exploration_prob (float): The exploration probability for selecting explorative moves.
        brain: The brain object used for making moves.

    Methods:
        getExplorativeMove(game): Returns an explorative move based on the current game state.
        getExploitativeMove(game): Returns an exploitative move based on the current game state.
        makeMove(game): Makes a move based on the current game state.
    """

    def __init__(self, brain=None, exploration_prob=0.9):
        """
        Initializes a new instance of the LearningPlayer class.

        Args:
            brain: The brain object used for making moves.
            exploration_prob (float): The exploration probability for selecting explorative moves.
        """
        self.type = "learning"
        self.exploration_prob = exploration_prob
        self.brain = brain

    def getExplorativeMove(self, game):
        """
        Returns an explorative move based on the current game state.

        Args:
            game: The current game object.

        Returns:
            tuple: A tuple containing the explorative move, MCTS value, NN value, doneFound flag, and action probability.
        """
        tau = 1
        action, pi, MCTS_value, NN_value, doneFound, action_prob = self.brain.act(game.gameState, tau)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return game.gameState.getAllowedActionByIndex(action), MCTS_value, NN_value, doneFound, action_prob
    
    def getExploitativeMove(self, game):
        """
        Returns an exploitative move based on the current game state.

        Args:
            game: The current game object.

        Returns:
            tuple: A tuple containing the exploitative move, MCTS value, NN value, doneFound flag, and action probability.
        """
        tau = 0
        action, pi, MCTS_value, NN_value, doneFound, action_prob = self.brain.act(game.gameState, tau)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return game.gameState.getAllowedActionByIndex(action), MCTS_value, NN_value, doneFound, action_prob

    def makeMove(self, game):
        """
        Makes a move based on the current game state.

        Args:
            game: The current game object.

        Returns:
            tuple: A tuple containing the move, MCTS value, NN value, doneFound flag, and action probability.
        """
        if random.random() < self.exploration_prob:
            move, MCTS_value, NN_value, doneFound, action_prob = self.getExplorativeMove(game)
        else:
            move, MCTS_value, NN_value, doneFound, action_prob = self.getExploitativeMove(game)
        return chess.Move.from_uci(move), MCTS_value, NN_value, doneFound, action_prob

class StockfishPlayer(Player):
    """
    A player that uses the Stockfish chess engine for move evaluation and selection.

    Attributes:
        stockfish_binary_path (str): The path to the Stockfish binary.
        type (str): The type of player, set to "stockfish".
        stockfish_engine (Stockfish): The Stockfish engine instance.
        brain (Brain): The brain associated with the player.
    """

    stockfish_binary_path = os.path.join(os.path.dirname(__file__), "..", "stockfish.bin")

    def __init__(self, brain):
        """
        Initializes a new instance of the StockfishPlayer class.

        Args:
            brain (Brain): The brain associated with the player.
        """
        self.type = "stockfish"
        self.stockfish_engine = Stockfish(self.stockfish_binary_path)
        self.stockfish_engine.set_depth(10)
        self.brain = brain
        self.brain.stockfish_player = self

    def lichess_sigmoid(self, centipawns):
        """
        Applies the Lichess sigmoid function to the given centipawns value.

        Args:
            centipawns (float): The centipawns value.

        Returns:
            float: The transformed value.
        """
        y = (50 + 50 * (2 / (1 + np.exp(-0.00368208 * np.abs(centipawns))) - 1)) / 100
        return np.sign(centipawns) * y

    def evaluate_position(self, state):
        """
        Evaluates the given state using the Stockfish engine.

        Args:
            state (State): The state to evaluate.

        Returns:
            float: The evaluation value.
        """
        # Convert the state to a FEN string and set it in the Stockfish engine
        fen = state.id
        self.stockfish_engine.set_fen_position(fen)

        # Get the evaluation from Stockfish
        evaluation = self.stockfish_engine.get_evaluation()

        if evaluation["type"] == "mate":
            value = np.sign(evaluation["value"])
        else:
            value = self.lichess_sigmoid(evaluation["value"])

        return value

    def makeMove(self, game):
        """
        Makes a move using the Stockfish engine.

        Args:
            game (Game): The game instance.

        Returns:
            tuple: A tuple containing the move, MCTS value, NN value, doneFound flag, and action probability.
        """
        self.stockfish_engine.set_fen_position(game.gameState.board.fen())
        move = self.stockfish_engine.get_best_move()
        action, pi, MCTS_value, NN_value, doneFound, action_prob = self.brain.stockfish_act(game.gameState, move)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return chess.Move.from_uci(move), MCTS_value, NN_value, doneFound, action_prob