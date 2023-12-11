from typing import List
import numpy as np
import chess
import chess.pgn
import os
import config

from .player import Player

class GameState:
    """
    Represents the state of a chess game.

    Attributes:
        action_space (numpy.ndarray): The action space of the game.
        board (chess.Board): The chess board.
    """

    def __init__(self, fen=None):
        """
        Initializes a new instance of the GameState class.

        Args:
            fen (str, optional): The FEN notation of the chess board. Defaults to None.
        """
        self.action_space = np.zeros(shape=(64, 64))
        if fen is not None:
            self.board = chess.Board(fen)

    def getAllowedActionByIndex(self, index):
        """
        Gets the allowed action at the specified index.

        Args:
            index (int): The index of the action.

        Returns:
            str: The UCI notation of the allowed move, or None if the move is not allowed.
        """
        action_tuple = [index // 64, index % 64]
        moves = [[x.from_square, x.to_square, x.uci()] for x in self.board.generate_legal_moves()]
        for move in moves:
            if move[0] == action_tuple[0] and move[1] == action_tuple[1]:
                return move[2]
        return None
    
    def getIndexOfAllowedMove(self, move):
        """
        Gets the index of the allowed move.

        Args:
            move (str): The UCI notation of the move.

        Returns:
            int: The index of the allowed move.

        Raises:
            Exception: If the move is invalid.
        """
        found = False
        from_square = None
        to_square = None

        for legal_move in self.board.generate_legal_moves():
            if move == legal_move.uci():
                found = True
                from_square, to_square = legal_move.from_square, legal_move.to_square

        if not found:
            raise Exception(f"Invalid move: {move}")

        index = from_square * 64 + to_square
        return index

    def takeAction(self, action):
        """
        Takes an action and returns the new state, value, and done flag.

        Args:
            action (int): The index of the action.

        Returns:
            tuple: A tuple containing the new state, value, and done flag.
        """
        value = 0
        done = 0
        
        if self.board.is_game_over():
            if self.board.is_checkmate():
                value = 1
                if newState.board.turn == chess.BLACK:
                    value = -1
            done = 1
            return (self, value, done)

        move = self.getAllowedActionByIndex(action)
        newBoard = self.board.copy()
        newBoard.push_uci(move)

        newState = GameState(newBoard.fen())

        if newState.board.is_game_over():
            if newState.board.is_checkmate():
                value = 1
                if newState.board.turn == chess.BLACK:
                    value = -1
            done = 1

        return (newState, value, done)

    @property
    def id(self):
        """
        Gets the FEN notation of the chess board.

        Returns:
            str: The FEN notation of the chess board.
        """
        return self.board.fen()

    @property
    def turn(self):
        """
        Gets the current turn.

        Returns:
            bool: True if it's white's turn, False if it's black's turn.
        """
        return self.board.turn
    
    @property
    def allowedActions(self):
        """
        Gets the allowed actions.

        Returns:
            numpy.ndarray: The allowed actions.
        """
        moves = [[x.from_square, x.to_square] for x in self.board.generate_legal_moves()]
        self.action_space = np.zeros((64, 64), dtype=int)
        if len(moves) > 0:    
            from_squares, to_squares = zip(*moves)
            self.action_space[from_squares, to_squares] = 1
        return self.action_space

    @property
    def allowedActionsIndexes(self):
        """
        Gets the indexes of the allowed actions.

        Returns:
            numpy.ndarray: The indexes of the allowed actions.
        """
        allowedActions = self.allowedActions.reshape(4096)
        allowedActionIndexes = np.where(allowedActions == 1)[0]
        return allowedActionIndexes
    
    def as_tensor(self):
        """
        Converts the chess board to a tensor representation.

        Returns:
            numpy.ndarray: The tensor representation of the chess board.
        """
        piece_to_index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        ALL_PIECES = 32
        
        tensor = np.zeros((8, 8, ALL_PIECES), dtype=np.int8)

        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)

                if piece is not None:
                    color = int(piece.color)
                    index = piece_to_index[piece.piece_type] + (color * 6)
                    tensor[rank][file][index] = 1

        if self.board.turn == chess.WHITE:
            tensor = np.append(tensor, np.ones((8, 8, 1)), axis=2)
        else:
            tensor = np.append(tensor, np.zeros((8, 8, 1)), axis=2)

        return tensor

class Game:
    def __init__(self, fen: str, iteration: int):
        """
        Initializes a new instance of the Game class.

        Args:
            fen (str): The FEN representation of the initial board position.
            iteration (int): The iteration number of the game.
        """
        self.fen = fen
        self.iteration = iteration
        self.gameState = GameState(fen)
        self.move_values = []

    def setWhitePlayer(self, player: Player):
        """
        Sets the white player in the game.

        Args:
            player (Player): The white player.
        """
        self.whitePlayer = player

    def setBlackPlayer(self, player: Player):
        """
        Sets the black player in the game.

        Args:
            player (Player): The black player.
        """
        self.blackPlayer = player

    def playUntilFinished(self):
        """
        Plays the game until it is finished.

        The game is played by alternating moves between the white and black players until the game is over or the maximum number of moves is reached.
        The move values are recorded during the game.
        """
        current_move = 0
        max_moves = config.MAX_NUMBER_OF_MOVES
        while not self.gameState.board.is_game_over() and current_move < max_moves:
            if self.gameState.turn == chess.WHITE:
                move, MCTS_value, NN_value, doneFound, action_prob = self.whitePlayer.makeMove(self)
                if self.whitePlayer.type == "learning":
                    self.move_values.append((self.iteration, current_move, move, "learning", "white", MCTS_value, NN_value[0], doneFound, action_prob))
                else:
                    self.move_values.append((self.iteration, current_move, move, "stockfish", "white", MCTS_value, NN_value[0], doneFound, action_prob))
            else:
                move, MCTS_value, NN_value, doneFound, action_prob = self.blackPlayer.makeMove(self)
                if self.blackPlayer.type == "learning":
                    self.move_values.append((self.iteration, current_move, move, "learning", "black", MCTS_value, NN_value[0], doneFound, action_prob))
                else:
                    self.move_values.append((self.iteration, current_move, move, "stockfish", "black", MCTS_value, NN_value[0], doneFound, action_prob))

            newBoard = self.gameState.board.copy()
            newBoard.push(move)
            self.gameState = GameState()
            self.gameState.board = newBoard
            current_move += 1

    def savePGN(self, name, white_name="OmegaZero", black_name="Stockfish"):
        """
        Saves the game as a PGN file.

        Args:
            name (str): The name of the game.
            white_name (str, optional): The name of the white player. Defaults to "OmegaZero".
            black_name (str, optional): The name of the black player. Defaults to "Stockfish".
        """
        # Initialize a counter to add to the filename if it already exists
        counter = 1
        file_name = f"games/{name}_{counter}.pgn"

        # Create the "games" folder if it doesn't exist
        os.makedirs("games", exist_ok=True)

        # Check if the file already exists, and if it does, add a number to the filename
        while os.path.exists(file_name):
            file_name = f"games/{name}_{counter}.pgn"
            counter += 1

        pgn_game = chess.pgn.Game.from_board(self.gameState.board)
        pgn_game.headers["Event"] = name
        pgn_game.headers["White"] = white_name
        pgn_game.headers["Black"] = black_name

        # Write the game as a PGN file
        with open(file_name, "w", encoding="utf-8") as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            pgn_game.accept(exporter)
    
    def getGameState(self):
        """
        Returns the current state of the game.

        Returns:
            GameState: The current state of the game.
        """
        return self.gameState
    
    def identities(self, state, actionValues):
        """
        Converts the action values into a list of state-action pairs.

        Args:
            state: The current state of the game.
            actionValues: The action values as a 4096-dimensional vector.

        Returns:
            list: A list of state-action pairs.
        """
        # convert actionValues 4096 vector into 64x64 matrix
        actionValues = actionValues.reshape((64, 64))
        identities = [(state, actionValues)]
        return identities
