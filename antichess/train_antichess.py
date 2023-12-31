
import random
import numpy as np
import chess
import chess.variant
import tensorflow as tf
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential, load_model
from IPython.display import display, HTML
import chess.svg
from config import state_space_size, action_space_size, learning_rate, discount_factor
from Q_funct import update_q_table, calculate_reward, get_exploration_rate
from board_function import state_to_index, board_to_input_array
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, RandomUniform
import chess.pgn
from tqdm import tqdm  # Import the tqdm function

pretrain_path = "antichess_version/pretraining_games/lichess_swiss_2023.04.23_a2vcYLBJ_swiss-fight.pgn"

def pretrain_model(model, pgn_data, batch_size=1028):
    print("Pre-training on PGN data...")
    total_pgn_games = len(pgn_data)
    history = {'loss': []}  # Initialize a dictionary to store loss values

    input_batch = []
    output_batch = []

    for game_idx, (input_array, output_array) in enumerate(tqdm(pgn_data, desc="Processing", ncols=100), start=1):
        input_batch.append(input_array)
        output_batch.append(output_array)

        # Check if the current batch is the right size
        if len(input_batch) == batch_size or game_idx == total_pgn_games:
            input_batch = np.array(input_batch)
            output_batch = np.array(output_batch)

            # Train on the batch
            loss = model.train_on_batch(input_batch, output_batch)
            history['loss'].append(loss)

            # Clear the current batch
            input_batch = []
            output_batch = []

    # Save to csv
    history_df = pd.DataFrame(history)
    history_df.to_csv('pretrain_history.csv', index=False)

    print("Pretrained model")
    return history


    

def load_pgn_data(pgn_file_path):
    print("Loading PGN data from {}...",pgn_file_path)
    pgn_data = []
    i = 0
    pretrain_games = random.randint(1000,25000)
    print("pretrain_games: ", pretrain_games)
    with open(pgn_file_path) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            i += 1
            print("game number: ", i,"off",pretrain_games," processed")
            #random number between 10000 and 25000

            if i >pretrain_games or game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                input_array = board_to_input_array(board)
                output_array = move_to_output_array(move, board.legal_moves)
                pgn_data.append((input_array, output_array))
                board.push(move)
    return pgn_data

# Function to convert a move into an output array
def move_to_output_array(move, legal_moves):
    output_array = np.zeros(action_space_size)
    move_index = list(legal_moves).index(move)
    output_array[move_index] = 1
    return output_array



def create_new_model():
    # Randomly choosing initializers add he initializers
    init_choices = ['glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform', 'random_normal', 'random_uniform']
    conv_initializer = random.choice(init_choices)
    dense_initializer = random.choice(init_choices)

    # Enhanced Neural Network Model with random initializers
    input_layer = Input(shape=state_space_size)

    conv1 = Conv2D(64, (3, 3), padding='same', kernel_initializer=conv_initializer)(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    activation1 = Activation('relu')(batch_norm1)

    conv2 = Conv2D(128, (5, 5), padding='same', kernel_initializer=conv_initializer)(activation1)
    batch_norm2 = BatchNormalization()(conv2)
    activation2 = Activation('relu')(batch_norm2)

    conv3 = Conv2D(256, (5, 5), padding='same', kernel_initializer=conv_initializer)(activation2)
    batch_norm3 = BatchNormalization()(conv3)
    activation3 = Activation('relu')(batch_norm3)

    flatten_layer = Flatten()(activation3)

    dense1 = Dense(256, activation='relu', kernel_initializer=dense_initializer)(flatten_layer)
    dense2 = Dense(128, activation='relu', kernel_initializer=dense_initializer)(dense1)
    dense3 = Dense(64, activation='relu', kernel_initializer=dense_initializer)(dense2)

    output_layer = Dense(action_space_size, activation='softmax', kernel_initializer=dense_initializer)(dense3)

    new_model = Model(inputs=input_layer, outputs=output_layer)
    new_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                      loss=['categorical_crossentropy','huber_loss'],
                      metrics=['mse'])
    
    print("New model created with the following initializers:")
    print("conv_initializer: ", conv_initializer)
    print("dense_initializer: ", dense_initializer)
    print("--------------------")
    print(new_model.summary())
    
    pgn_data = load_pgn_data(pretrain_path)
    pretrain_model(new_model, pgn_data, batch_size=1028)

    return new_model


def train_model_self_play(num_games, model, exploration_prob=0.2):
    for _ in range(num_games):
        play_game(model, model,exploration_prob)

def play_game(model1, model2, exploration_prob):
    board = chess.variant.GiveawayBoard()
    while not board.is_game_over():
        current_state = board.copy()  # Capture the current state

        if board.turn == chess.WHITE:
            move = choose_action(board, model1, exploration_prob)
        else:
            move = choose_action(board, model2, exploration_prob)

        board.push(move)  # Make the move
        next_state = board.copy()  # Capture the next state
        reward = calculate_reward(board, board.turn)  # Calculate the reward

        # Update the Q-table with the new information
        update_q_table(current_state, move, reward, next_state, model1)
        update_q_table(current_state, move, reward, next_state, model2)

    return board.result()

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


def train_new_player(best_player_model, new_player_model, threshold_win_rate=0.51, exploration_prob=0.2):
    new_player_wins = 0
    total_games_played = 0
    win_rate = 0
    

    while True:
        total_games_played += 1
        if random.choice([True, False]):
            exploration_prob = get_exploration_rate(initial_exploration_rate, exploration_decay_rate, min_exploration_rate, total_games_played)
            result = play_game(new_player_model, best_player_model, exploration_prob)
            if result == "1-0":
                new_player_wins += 1
                print("number of games played: ", total_games_played)
            else:
                print("player lost")
        else:
            result = play_game(best_player_model, new_player_model, exploration_prob)
            if result == "0-1":
                new_player_wins += 1
                print("number of games played: ", total_games_played)
            else:
                print("player lost")

        win_rate = new_player_wins / total_games_played
        print(f"Game {total_games_played}. New player win rate: {win_rate}")

        if win_rate >= threshold_win_rate and total_games_played >= 100:
            print(f"New player has achieved a win rate of {win_rate}. It becomes the best player.")
            id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            new_player_model.save("model/best_player.h5")
            return new_player_model, win_rate, id, total_games_played

        #condition for too many games played
        
        if total_games_played > 1000:
            id = "player remains after 2.5k games"
            print(f"New player has achieved a win rate of {win_rate}. It did not become the best player.")
            return new_player_model, win_rate, id, total_games_played

            

def main():


    # Load or create initial best player model
    try:
        best_player_model = load_model("model/best_player.h5")
    except IOError:
        print("No initial model found. Training a new model.")
        best_player_model = create_new_model()
        train_model_self_play(1000, best_player_model)

    # Main training and updating loop
    # Initial Hyperparameters
    initial_exploration_rate = 0.95
    exploration_decay_rate = 0.001
    min_exploration_rate = 0.01
    num_games_played = 0


    #check if results.csv exists
    try:
        df = pd.read_csv('results.csv')
    except IOError:
        df = pd.DataFrame(columns=['winrate', 'id', 'num_games_played'])
        df.to_csv('results.csv', index=False)


    # Main training and updating loop
    while True:
        # Rest of the training loop
        new_player_model = create_new_model()
        print("new challenger")
        print("--------------------")
        best_player_model, winrate, id, num_games_played = train_new_player(best_player_model, new_player_model)

        # Write the results to a new row in the results df
        df.loc[len(df)] = [winrate, id, num_games_played]
        df.to_csv('results.csv', index=False)

        best_player_model.save("model/best_player.h5")


if __name__ == "__main__":
    main()

