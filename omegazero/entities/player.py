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
    def __init__(self, exploration_prob = 0.9, is_best_player = False):
        self.id = str(random.randint(0, 1000000))
        self.exploration_prob = exploration_prob
        self.is_best_player = is_best_player
        self.brain = Brain(self)

    def setAsBestPlayer(self):
        self.is_best_player = True
        new_player_path = os.path.join(artifacts_path, "new_player_nn_%s.pth.tar" % self.player.id)
        best_player_path = os.path.join(artifacts_path, "best_player.pth.tar")
        if os.path.isfile(new_player_path):
            os.rename(new_player_path, best_player_path)

    def getExplorativeMove(self, game):
        action, pi, MCTS_value, NN_value = self.brain.act(game.gameState, 0)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return game.gameState.getAllowedActionByIndex(action)
    
    def getExploitativeMove(self, game):
        action, pi, MCTS_value, NN_value = self.brain.act(game.gameState, 1)
        self.brain.memory.commit_stmemory(game.identities, game.gameState, pi, MCTS_value)
        return game.gameState.getAllowedActionByIndex(action)

    def makeMove(self, game):
        if random.random() < self.exploration_prob:
            move = self.getExplorativeMove(game)
        else:
            move = self.getExploitativeMove(game)
        return chess.Move.from_uci(move)

class StockfishPlayer(Player):
    stockfish_binary_path = os.path.join(os.path.dirname(__file__), "..", "stockfish.bin")
    def __init__(self):
        self.stockfish_engine = Stockfish(self.stockfish_binary_path)

    def makeMove(self, game):
        self.stockfish_engine.set_fen_position(game.gameState.board.fen())
        move = self.stockfish_engine.get_best_move()
        return chess.Move.from_uci(move)