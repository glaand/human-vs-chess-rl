from stockfish import Stockfish
import chess
import os
import random

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
        action, pi, MCTS_value, NN_value = self.brain.act(game.gameState, tau)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return game.gameState.getAllowedActionByIndex(action), MCTS_value, NN_value
    
    def getExploitativeMove(self, game):
        tau = 0
        action, pi, MCTS_value, NN_value = self.brain.act(game.gameState, tau)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return game.gameState.getAllowedActionByIndex(action), MCTS_value, NN_value

    def makeMove(self, game):
        if random.random() < self.exploration_prob:
            move, MCTS_value, NN_value = self.getExplorativeMove(game)
        else:
            move, MCTS_value, NN_value = self.getExploitativeMove(game)
        return chess.Move.from_uci(move), MCTS_value, NN_value

class StockfishPlayer(Player):
    stockfish_binary_path = os.path.join(os.path.dirname(__file__), "..", "stockfish.bin")
    def __init__(self, brain):
        self.type = "stockfish"
        self.stockfish_engine = Stockfish(self.stockfish_binary_path)
        self.brain = brain

    def makeMove(self, game):
        self.stockfish_engine.set_fen_position(game.gameState.board.fen())
        move = self.stockfish_engine.get_best_move()
        tau = 0
        action, pi, MCTS_value, NN_value = self.brain.stockfish_act(game.gameState, move, tau)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return chess.Move.from_uci(move), MCTS_value, NN_value