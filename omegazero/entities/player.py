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
    def __init__(self, brain = None, exploration_prob = 0.9):
        self.type = "learning"
        self.exploration_prob = exploration_prob
        self.brain = brain

    def getExplorativeMove(self, game):
        tau = 1
        action, pi, MCTS_value, NN_value, doneFound = self.brain.act(game.gameState, tau)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return game.gameState.getAllowedActionByIndex(action), MCTS_value, NN_value, doneFound
    
    def getExploitativeMove(self, game):
        tau = 0
        action, pi, MCTS_value, NN_value, doneFound = self.brain.act(game.gameState, tau)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return game.gameState.getAllowedActionByIndex(action), MCTS_value, NN_value, doneFound

    def makeMove(self, game):
        if random.random() < self.exploration_prob:
            move, MCTS_value, NN_value, doneFound = self.getExplorativeMove(game)
        else:
            move, MCTS_value, NN_value, doneFound = self.getExploitativeMove(game)
        return chess.Move.from_uci(move), MCTS_value, NN_value, doneFound

class StockfishPlayer(Player):
    stockfish_binary_path = os.path.join(os.path.dirname(__file__), "..", "stockfish.bin")
    def __init__(self, brain):
        self.type = "stockfish"
        self.stockfish_engine = Stockfish(self.stockfish_binary_path)
        self.stockfish_engine.set_depth(10)
        self.brain = brain
        self.brain.stockfish_player = self

    def lichess_sigmoid(self, centipawns):
        y = (50 + 50 * (2 / (1 + np.exp(-0.00368208 * np.abs(centipawns))) - 1)) / 100
        return np.sign(centipawns) * y

    def evaluate_position(self, state):
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
        self.stockfish_engine.set_fen_position(game.gameState.board.fen())
        move = self.stockfish_engine.get_best_move()
        action, pi, MCTS_value, NN_value, doneFound = self.brain.stockfish_act(game.gameState, move)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return chess.Move.from_uci(move), MCTS_value, NN_value, doneFound