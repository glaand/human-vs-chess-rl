
import random
import chess
import chess.variant
import tensorflow as tf
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential, load_model
from IPython.display import display, HTML
import chess.svg
from config import state_space_size, action_space_size, learning_rate, discount_factor
from board_function import board_to_input_array, state_to_index, move_to_output_array,  count_pieces_by_color, normalize_input
from Q_funct import update_q_table, choose_action, calculate_reward, get_exploration_rate
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, RandomUniform






def create_new_model():
    # Randomly choosing initializers add he initializers
    init_choices = [RandomNormal(mean=0.0, stddev=0.05), RandomUniform(minval=-0.05, maxval=0.05), 'glorot_uniform', 'glorot_normal', 'he_normal', 'he_uniform']
    conv_initializer = random.choice(init_choices)
    dense_initializer = random.choice(init_choices)

    # Enhanced Neural Network Model with random initializers
    input_layer = Input(shape=state_space_size)

    conv1 = Conv2D(64, (3, 3), padding='same', kernel_initializer=conv_initializer)(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    activation1 = Activation('relu')(batch_norm1)

    conv2 = Conv2D(128, (3, 3), padding='same', kernel_initializer=conv_initializer)(activation1)
    batch_norm2 = BatchNormalization()(conv2)
    activation2 = Activation('relu')(batch_norm2)

    conv3 = Conv2D(256, (3, 3), padding='same', kernel_initializer=conv_initializer)(activation2)
    batch_norm3 = BatchNormalization()(conv3)
    activation3 = Activation('relu')(batch_norm3)

    flatten_layer = Flatten()(activation3)

    dense1 = Dense(256, activation='relu', kernel_initializer=dense_initializer)(flatten_layer)
    dense2 = Dense(128, activation='relu', kernel_initializer=dense_initializer)(dense1)
    dense3 = Dense(64, activation='relu', kernel_initializer=dense_initializer)(dense2)

    output_layer = Dense(action_space_size, activation='softmax', kernel_initializer=dense_initializer)(dense3)

    new_model = Model(inputs=input_layer, outputs=output_layer)
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

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


def train_new_player(best_player_model, new_player_model, threshold_win_rate=0.55, exploration_prob=0.2):
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
            result = play_game(best_player_model, new_player_model, exploration_prob)
            if result == "0-1":
                new_player_wins += 1
                print("number of games played: ", total_games_played)

        win_rate = new_player_wins / total_games_played
        print(f"Game {total_games_played}. New player win rate: {win_rate}")

        if win_rate >= threshold_win_rate:
            print(f"New player has achieved a win rate of {win_rate}. It becomes the best player.")
            id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            new_player_model.save("model/best_player.h5")
            return new_player_model, win_rate, id, total_games_played

        #condition for too many games played
        
        if total_games_played > 2500:
            id = "player remains after 2.5k games"
            print(f"New player has achieved a win rate of {win_rate}. It did not become the best player.")
            return new_player_model, win_rate, id, total_games_played

            




# Load or create initial best player model
try:
    best_player_model = load_model("model/best_player.h5")
except IOError:
    print("No initial model found. Training a new model.")
    best_player_model = create_new_model()
    train_model_self_play(100, best_player_model)

# Main training and updating loop
# Initial Hyperparameters
initial_exploration_rate = 0.8
exploration_decay_rate = 0.001
min_exploration_rate = 0.01
num_games_played = 0

df = pd.DataFrame(columns=['win_rate', 'id', 'num_games_played'])

# Main training and updating loop
while True:


    # Rest of the training loop
    new_player_model = create_new_model()
    best_player_model, winrate, id, num_games_played = train_new_player(best_player_model, new_player_model)

    # Write the results to a new row in the results df
    df.loc[len(df)] = [winrate, id, num_games_played]
    df.to_csv('results.csv', index=False)

    best_player_model.save("best_player.h5")



