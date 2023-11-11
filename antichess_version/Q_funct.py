from board_function import state_to_index, board_to_input_array
import chess
import numpy as np
import random
from collections import deque
from config import learning_rate, discount_factor
import numpy as np

experience_replay_buffer = deque(maxlen=1000000)

def get_exploration_rate(initial_rate, decay_rate, min_rate, num_games):
    return max(initial_rate - decay_rate * num_games, min_rate)


def update_q_table(state, action, reward, next_state, model):
    """
    Updates the Q-table using the given state, action, reward, next state, and model.

    Args:
        state (chess.Board): The current state of the game.
        action (chess.Move): The action taken in the current state.
        reward (float): The reward received for taking the action in the current state.
        next_state (chess.Board): The resulting state after taking the action.
        model (keras.Model): The Q-value neural network model.

    Returns:
        None
    """
    state_index = state_to_index(state)
    next_state_index = state_to_index(next_state)
    action_index = list(state.legal_moves).index(action)

    total_reward = reward
    experience_replay_buffer.append((state_index, action_index, total_reward, next_state_index))
    batch_size = min(len(experience_replay_buffer), 8)
    if batch_size > 0:
        batch = np.array(random.sample(experience_replay_buffer, batch_size))
        states = np.array([board_to_input_array(chess.Board(fen=chess.STARTING_FEN)) for _ in batch[:, 0]])
        next_states = np.array([board_to_input_array(chess.Board(fen=chess.STARTING_FEN)) for _ in batch[:, 3]])
        q_values = model.predict(states)
        next_q_values = model.predict(next_states)
        
        for i in range(batch_size):
            action_idx = int(batch[i, 1])
            q_values[i, action_idx] += learning_rate * (batch[i, 2] + discount_factor * np.max(next_q_values[i]) - q_values[i, action_idx])
        
        model.train_on_batch(states, q_values)
        

def calculate_reward(board, ai_color):
    """
    Calculates the reward for a given chess board state.

    Args:
        board (chess.Board): The chess board state to calculate the reward for.

    Returns:
        float: The reward for the given board state.
    """
    reward = 0
    piece_count = len(board.piece_map())

    # Penalize for having more pieces
    reward -= (32 - piece_count) * 0.1

    # Check for game over conditions
    if board.is_game_over():
        result = board.result()
        if (result == '0-1' and ai_color == chess.WHITE) or (result == '1-0' and ai_color == chess.BLACK):
            # AI lost all pieces (goal achieved)
            reward += 1000  # Large positive reward
        elif (result == '1-0' and ai_color == chess.WHITE) or (result == '0-1' and ai_color == chess.BLACK):
            # AI won in the traditional sense (not the goal)
            reward -= 1000  # Large negative reward
        else:
            # Draw or stalemate
            reward -= 5  # Moderate negative reward
    else:
        # Penalize for certain non-winning conditions
        if board.is_stalemate() or board.is_insufficient_material():
            reward -= 5
        elif board.is_fivefold_repetition() or board.is_seventyfive_moves():
            reward -= 5

    return reward


def choose_action(board, model, exploration_prob):
    """
    Chooses an action to take given the current board state and a trained Q-function model.

    Args:
        board: A chess.Board object representing the current board state.
        model: A trained Q-function model that takes in a board state and outputs Q-values for each possible action.

    Returns:
        A chess.Move object representing the chosen action.
    """
    if np.random.rand() < exploration_prob:
        return np.random.choice(list(board.legal_moves))
    else:
        state_index = state_to_index(board)
        legal_moves_list = list(board.legal_moves)
        if not legal_moves_list:
            return chess.Move.null()
        q_values = model.predict(np.array([board_to_input_array(board)]))[0]
        best_move_index = np.argmax(q_values)
        best_move_uci = legal_moves_list[min(best_move_index, len(legal_moves_list)-1)].uci()
        return chess.Move.from_uci(best_move_uci)
