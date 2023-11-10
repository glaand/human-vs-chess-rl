
import numpy as np
import chess
from config import state_space_size, action_space_size



def board_to_input_array(board):
    """
    Converts a chess board object to a 3D numpy array that can be used as input to a neural network.

    Args:
        board (chess.Board): The chess board object to convert.

    Returns:
        numpy.ndarray: A 3D numpy array of shape (8, 8, 12) representing the board state.
            The first two dimensions represent the board coordinates, and the third dimension
            represents the type of piece at that location (0-5: black rook, knight, bishop, queen, king, pawn;
            6-11: white rook, knight, bishop, queen, king, pawn). The values are 0 for an empty square,
            1 for a black piece, and 2 for a white piece.
    """
    board_array = np.zeros((8, 8, 12), dtype=np.uint8)
    piece_mapping = {'r': 0, 'n': 1, 'b': 2, 'q': 3, 'k': 4, 'p': 5, 'R': 6, 'N': 7, 'B': 8, 'Q': 9, 'K': 10, 'P': 11}

    for square, piece in board.piece_map().items():
        piece_type = piece_mapping[piece.symbol()]
        color = int(piece.color)
        board_array[square // 8, square % 8, piece_type] = color + 1
    return board_array

def state_to_index(board):
    """
    Converts the given board state to an index in the state space.

    Args:
        board (list): The board state to convert.

    Returns:
        int: The index in the state space corresponding to the given board state.
    """
    board_array = np.array(board_to_input_array(board))
    return hash(board_array.tostring()) % state_space_size[0]

def move_to_output_array(move, legal_moves):
    """
    Converts a given move to a one-hot encoded numpy array of legal moves.

    Args:
        move (int): The index of the move to be converted.
        legal_moves (set): A set of legal moves.

    Returns:
        numpy.ndarray: A one-hot encoded numpy array of legal moves.
    """
    output_array = np.zeros(action_space_size)
    move_index = list(legal_moves).index(move)
    output_array[move_index] = 1
    return output_array

def count_pieces_by_color(board, color):
    """ function to count the number of pieces of a given color on the board after the game is finished"""
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    return sum(len(board.pieces(piece_type, color)) for piece_type in piece_types)

def normalize_input(board):
    """
    Normalizes the input board array by dividing it by 12.0.

    Args:
        board (list): A list representing the current state of the chess board.

    Returns:
        numpy.ndarray: A normalized numpy array representing the current state of the chess board.
    """
    board_array = np.array(board_to_input_array(board), dtype=np.float16)
    board_array /= 12.0
    return board_array