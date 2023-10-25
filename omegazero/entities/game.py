from typing import List
import numpy as np
import chess
import chess.pgn
import os
import config

from .player import Player

class GameState:

    def __init__(self, fen=None):
        self.action_space = np.zeros(shape=(64, 64))
        if fen is not None:
            self.board = chess.Board(fen)

    def getAllowedActionByIndex(self, index):
        action_tuple = [index // 64, index % 64]
        moves = [[x.from_square, x.to_square, x.uci()] for x in self.board.generate_legal_moves()]
        for move in moves:
            if move[0] == action_tuple[0] and move[1] == action_tuple[1]:
                return move[2]
        return None

    def takeAction(self, action):

        if self.isGameOver:
            if self.board.is_checkmate():
                value = 1
            else:
                value = -1
            done = 1
            return (self, value, done)

        move = self.getAllowedActionByIndex(action)
        if move is None:
            raise Exception(f"Invalid action: {action}")
        newBoard = self.board.copy()
        newBoard.push_uci(move)

        newState = GameState(newBoard.fen())

        value = 0
        done = 0

        if newState.isGameOver:
            if newState.board.is_checkmate():
                value = 1
            else:
                value = -1
            done = 1

        return (newState, value, done)

    @property
    def id(self):
        return self.board.fen()

    @property
    def turn(self):
        return self.board.turn
    
    @property
    def allowedActions(self):
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        self.action_space = np.zeros((64, 64), dtype=int)
        if len(moves) > 0:    
            from_squares, to_squares = zip(*moves)
            self.action_space[from_squares, to_squares] = 1
        return self.action_space

    @property
    def allowedActionsIndexes(self):
        # Reshape the 64x64 matrix to a 4096x1 array
        allowedActions = self.allowedActions.reshape(4096)
        # Get indexes of allowed actions
        allowedActionIndexes = np.where(allowedActions == 1)[0]
        return allowedActionIndexes
    
    def as_tensor(self):
        # Define a dictionary to map piece types to channel indices
        piece_to_index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        # Initialize the tensor with zeros
        ALL_PIECES = 32
        
        tensor = np.zeros((8, 8, ALL_PIECES), dtype=np.int8)

        # Loop through the board and fill the tensor
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)

                if piece is not None:
                    # Determine the color of the piece
                    color = int(piece.color)
                    # Calculate the index for the piece-color combination
                    index = piece_to_index[piece.piece_type] + (color * 6)
                    # Set the corresponding channel to 1
                    tensor[rank][file][index] = 1

        # You can add additional information to the tensor, such as whose turn it is, castling rights, etc.

        return tensor
    
    @property
    def isGameOver(self):
        return self.board.is_game_over()

class Game:
    def __init__(self, fen: str):
        self.fen = fen
        self.gameState = GameState(fen)

    def setWhitePlayer(self, player: Player):
        self.whitePlayer = player

    def setBlackPlayer(self, player: Player):
        self.blackPlayer = player

    def playUntilFinished(self):
        current_move = 0
        max_moves = config.MAX_NUMBER_OF_MOVES
        while not self.gameState.isGameOver and current_move < max_moves:
            if self.gameState.turn == chess.WHITE:
                move = self.whitePlayer.makeMove(self)
            else:
                move = self.blackPlayer.makeMove(self)

            newBoard = self.gameState.board.copy()
            newBoard.push(move)
            self.gameState = GameState()
            self.gameState.board = newBoard
            current_move += 1

    def savePGN(self, name="game"):
        # Initialize a counter to add to the filename if it already exists
        counter = 1
        file_name = f"games/{name}.pgn"

        # Create the "games" folder if it doesn't exist
        os.makedirs("games", exist_ok=True)

        # Check if the file already exists, and if it does, add a number to the filename
        while os.path.exists(file_name):
            file_name = f"games/{name}_{counter}.pgn"
            counter += 1

        pgn_game = chess.pgn.Game.from_board(self.gameState.board)

        # Write the game as a PGN file
        with open(file_name, "w", encoding="utf-8") as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            pgn_game.accept(exporter)
    
    def getGameState(self):
        return self.gameState
    
    def identities(self, state, actionValues):
        # convert actionValues 4096 vector into 64x64 matrix
        actionValues = actionValues.reshape((64, 64))
        identities = [(state, actionValues)]
        return identities
