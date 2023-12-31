import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import chess.variant
import chess.svg
from board_function import board_to_input_array  # Assuming this is defined in your board_function

# Load the model
best_player_model = load_model("antichess_version/model/best_player_antichess.h5")


    
# Initialize session state
if 'board' not in st.session_state:
    st.session_state.board = chess.variant.GiveawayBoard()
if 'player_color' not in st.session_state:
    st.session_state.player_color = None

# Display function
def display_chess_board(board):
    return st.image(chess.svg.board(board=board, size=200), use_column_width=True)

# Function to reset the game
def reset_game():
    st.session_state.board = chess.variant.GiveawayBoard()
    st.session_state.player_color = None

def choose_action(board, model):
    """
    Chooses an action for giveaway chess given the current board state and a trained Q-function model.
    Enforces the rule that if a capture is available, it must be chosen.

    Args:
        board: A chess.Board object representing the current board state.
        model: A trained Q-function model for chess.

    Returns:
        A chess.Move object representing the chosen action.
    """
    legal_moves_list = list(board.legal_moves)
    if not legal_moves_list:
        return chess.Move.null()

    # Check if any capturing moves are available
    capturing_moves = [move for move in legal_moves_list if board.is_capture(move)]

    # If capturing moves are available, only consider those
    if capturing_moves:
        legal_moves_list = capturing_moves

    # Predict Q-values for each possible action
    q_values = model.predict(np.array([board_to_input_array(board)]))[0]

    # Select the best move from the filtered list
    best_move_index = np.argmax(q_values)
    best_move_uci = legal_moves_list[min(best_move_index, len(legal_moves_list)-1)].uci()
    return chess.Move.from_uci(best_move_uci)

def choose_color():
    color = st.radio("Choose your color", ('White', 'Black'))
    if st.button("Confirm Color"):
        st.session_state.player_color = 'white' if color == 'White' else 'black'
        # If the player chooses black, the bot makes the first move
        if st.session_state.player_color == 'black':
            bot_move = choose_action(st.session_state.board, best_player_model)
            st.session_state.board.push(bot_move)
        st.experimental_rerun()

def handle_player_move(move_input):
    try:
        move = chess.Move.from_uci(move_input)
        if move not in st.session_state.board.legal_moves:
            return "Illegal move"
        
        capturing_moves = [m for m in st.session_state.board.legal_moves if st.session_state.board.is_capture(m)]
        if capturing_moves and move not in capturing_moves:
            return "A capture is available, and you must capture."

        st.session_state.board.push(move)
        return None
    except ValueError:
        return "Invalid move"


def main():
    if st.session_state.player_color is None:
        choose_color()
    else:
        # Player's move input
        move_input = st.text_input("Enter your move:")

        # Process player's move
        if st.button("Make Move"):
            error_message = handle_player_move(move_input)

            if error_message:
                st.error(error_message)
            else:
                # Bot's move
                if not st.session_state.board.is_game_over() and st.session_state.board.turn != (st.session_state.player_color == 'white'):
                    bot_move = choose_action(st.session_state.board, best_player_model)
                    st.session_state.board.push(bot_move)

        # Update the display after each move
        display_chess_board(st.session_state.board)

        # Print result and reset button when the game is over
        if st.session_state.board.is_game_over():
            st.write(f"Game over: {st.session_state.board.result()}")
            if st.button("Reset Game"):
                reset_game()
                st.experimental_rerun()
                
                
if __name__ == "__main__":
    main()