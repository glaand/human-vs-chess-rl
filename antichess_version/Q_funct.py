from board_function import state_to_index, board_to_input_array
import chess
import numpy as np
import random

experience_replay_buffer = deque(maxlen=10000)


def update_q_table(state, action, reward, next_state, model):
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
        

def calculate_reward(board):
    reward = 0
    piece_count = len(board.piece_map())
    reward -= (32 - piece_count) * 0.1

    if board.is_stalemate() or board.is_insufficient_material():
        reward -= 5
    elif board.is_fivefold_repetition() or board.is_seventyfive_moves():
        reward -= 5
    return reward

