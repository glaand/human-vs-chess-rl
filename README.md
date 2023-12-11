<div align="center">
<h1 align="center">
Chessbot RL</h1>
<h2>André Glatzl, Benito Rusconi</h2>
<h3>◦ Code together, thrive forever!</h3>
<h3>◦ Developed with the software and tools below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat-square&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=flat-square&logo=TensorFlow&logoColor=white" alt="TensorFlow" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Keras-D00000.svg?style=flat-square&logo=Keras&logoColor=white" alt="Keras" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy" />
</p>
</div>

---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📦 Features](#-features)
- [📂 Repository Structure](#-repository-structure)
- [⚙️ Modules](#modules)
- [🚀 Getting Started](#-getting-started)
    - [🔧 Installation](#-installation)
    - [🤖 Running ](#-running-)

---


## 📍 Overview

The repository contains code for an Antichess game with various components like GUI, model training, exploration, and AlphaZero algorithm implementation. The project provides a user-friendly interface for playing Antichess against a trained Q-function model. It also includes functionalities for training an advanced AI model using self-play techniques and implementing the AlphaZero algorithm. This codebase enables users to explore and analyze data related to Antichess and provides various functionalities for game development, machine learning, and chess engine integration.

The Omegazero component is a simplification of the Alphazero project. Instead of playing against itself, it plays against stockfish.

---

## 📦 Features

Our project consists of two approaches: Deep Q-Learning and Deep Q-Learning with Monte Carlo Tree search. The component *antichess* uses a deep q-learning approach and the *omegazero* uses the one with Monte Carlo Tree search.

---


## 📂 Repository Structure

```sh
└── /
    ├── antichess/
    │   ├── .GIT_KEEP
    │   ├── Q_funct.py
    │   ├── board_function.py
    │   ├── config.py
    │   ├── gui.py
    │   ├── model/
    │   └── train_antichess.py
    ├── exploration/
    │   ├── .GIT_KEEP
    │   ├── Antichess_Exploration.ipynb
    │   └── First_Exploration.ipynb
    ├── gui/
    │   ├── examples/*.pgn
    │   ├── games_bk/*.pgn
    │   └── gui.py
    ├── omegazero/
    │   ├── Makefile
    │   ├── algorithms/
    │   │   ├── mcts.py
    │   │   └── nn.py
    │   ├── artifacts/
    │   │   └── .GIT_KEEP
    │   ├── config.py
    │   ├── entities/
    │   │   ├── brain.py
    │   │   ├── game.py
    │   │   ├── memory.py
    │   │   └── player.py
    │   ├── evaluation.py
    │   ├── omegazero.py
    │   ├── quellen.txt
    │   ├── run_episodes.sh
    │   ├── stages/
    │   │   ├── evaluate.py
    │   │   ├── learn.py
    │   │   └── play.py
    │   └── stockfish.bin
    └── requirements.txt

```

---


## ⚙️ Modules

<details><summary>Root</summary>

| File                            | Summary                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                             | ---                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [requirements.txt]({file_path}) | The code in the requirements.txt file specifies the dependencies for a project. It includes various libraries such as gymnasium, gym-chess, keras, numpy, chess, pandas, matplotlib, tensorflow, stockfish, and cairosvg. These libraries are required for the project to run successfully and provide functionalities like game development, machine learning, data manipulation, visualization, and chess engine integration. |

</details>

<details><summary>Antichess</summary>

| File                              | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---                               | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [gui.py]({file_path})             | The code is a Streamlit app that allows users to play giveaway chess against a trained Q-function model. It loads a pre-trained model and displays a chessboard using the SVG format. Users can choose their color, make moves, and the app will validate the moves and update the board accordingly. If it's the bot's turn, it uses the Q-function model to choose the best move. The app also handles game over scenarios and provides a reset option.                                                                                                                                                                                                                                                                                                                                                                        |
| [board_function.py]({file_path})  | The code in the `board_function.py` file provides several functionalities for working with a chess board. These functionalities include:1. `board_to_input_array(board)`: Converts a chess board object into a 3D numpy array that represents the board state. The array has dimensions (8, 8, 12) and each element represents a square on the board and the type of piece at that location.2. `state_to_index(board)`: Converts a given board state into an index in the state space.3. `move_to_output_array(move, legal_moves)`: Converts a given move into a one-hot encoded numpy array of legal moves.4. `count_pieces_by_color(board, color)`: Counts the number of pieces of a given color on the board after the game is finished.5. `normalize_input(board)`: Normalizes the input board array by dividing it by 12.0. |
| [train_antichess.py]({file_path}) | The code is for training an advanced AI model to play the game of anti-chess. It uses a Q-function model to make decisions and uses the self-play technique to improve its performance over time. The code includes functions for pre-training the model on historical game data, creating a new model with random initializers, playing games between different versions of the model to determine the best player, and updating the best player based on its win rate. The code also saves the best player model and logs the win rate and number of games played.                                                                                                                                                                                                                                                             |
| [config.py]({file_path})          | The code in `antichess/config.py` defines several variables that are used for configuring the antichess game. These variables include the learning rate, discount factor, state space size, action space size, and experience replay buffer size. These configurations are important for training and playing the antichess game.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [Q_funct.py]({file_path})         | The code provides functionalities for updating the Q-table used in the reinforcement learning of an anti-chess AI. It includes functions for calculating the exploration rate, updating the Q-table values based on rewards and model predictions, and calculating rewards for a given chess board state. The code also manages an experience replay buffer and saves the training history to a CSV file.                                                                                                                                                                                                                                                                                                                                                                                                                        |

</details>

<details><summary>Gui</summary>

| File                  | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---                   | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| [gui.py]({file_path}) | The code is a GUI application that allows users to play the game of Antichess against a trained bot. It uses the tkinter library for creating the graphical interface. The core functionalities of the code are as follows:-The application allows users to choose their color (white or black) and play against the bot.-The application displays the chessboard and the current state of the game using a graphical representation.-Users can make moves by entering them in a text field and clicking the Make Move button.-The application validates the move entered by the user and displays an error message if the move is invalid.-After the user makes a move, the bot automatically responds with its move.-The application updates the display to reflect the new state of the game after each move.-If the game is over (checkmate or stalemate), the application displays a message indicating the result and offers the option to reset the game.-The application also allows users to navigate through the moves of a pre-recorded game by clicking the Previous Move and Next Move buttons.-Users can load multiple pre-recorded games (in PGN format) from a selected folder, and the application displays the boards of these games in separate tabs. |

</details>

<details><summary>Exploration</summary>

| File                                       | Summary                                                                                                                                                                                                                                                                                                                                                                                            |
| ---                                        | ---                                                                                                                                                                                                                                                                                                                                                                   |
| [First_Exploration.ipynb]({file_path})     | The code in the First_Exploration.ipynb notebook explores a Python codebase's directory structure using a depth-first search algorithm. It imports the deque and random modules and does not contain any code snippets or outputs.                                                                                                                                                                 |
| [Antichess_Exploration.ipynb]({file_path}) | The code is part of a directory structure that includes several folders and files. In particular, the code is located in the file Antichess_Exploration.ipynb in the exploration folder. It is written in Python and is likely used for exploring and analyzing data related to the game of Antichess. The code is organized into cells, with the specific functionality of the code not provided. |

</details>

<details><summary>Omegazero</summary>

| File                           | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---                            | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [omegazero.py]({file_path})    | The code is a script for running the OmegaZero component, which is used for playing and learning the game of chess. It performs the following core functionalities:1. It imports necessary modules and classes.2. It defines a function called save_game_data that saves game data to a CSV file.3. It defines a function called main that executes the main logic of the script.4. It initializes a variable called fen_string with a specific chess position.5. It calls the main function with a specified episode number, which triggers the execution of the OmegaZero.6. The main function performs the following steps: a. It creates a play stage and plays a specified number of games using the MCTS algorithm. b. It saves the game data from the play stage using the save_game_data function. c. It creates a learn stage and trains the OmegaZero player using the game data. d. It creates an evaluate stage and evaluates the performance of the trained player. e. It prints the episode number and metrics obtained from the evaluate stage. f. It saves the metrics to a log file. |
| [run_episodes.sh]({file_path}) | The code is a shell script that runs a specified number of episodes of a program called omegazero.py, passing the episode number as an argument. After running the episodes, it then executes another program called evaluation.py.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [evaluation.py]({file_path})   | The code in `omegazero/evaluation.py` reads data from `game_data.csv` and `loss_data.csv` and then creates two types of plots. The `plot_game_data()` function creates a 2x2 grid of subplots showing the Q-Values over episodes. The top left subplot displays the MCTS and NN values for the white player learning. The top right subplot displays the MCTS and NN values for the black player stockfish. The bottom left subplot displays the MCTS and NN values for the white player stockfish. The bottom right subplot displays the MCTS and NN values for the black player learning. The `plot_loss_data()` function creates a plot showing the loss over epochs. The x-axis represents epochs ordered by episodes and the y-axis represents the loss values.                                                                                                                                                                                                                                                                                                                                                |
| [quellen.txt]({file_path})     | This code calculates a value based on the evaluation of a chess position. It uses a formula that takes into account the centipawn value (a measure of the quality of a chess move) and applies a sigmoid function to it. The result is a value that ranges between 0 and 100, representing the desirability of the move. The code also includes a link to a Reddit post discussing the evaluation of chess positions using Stockfish.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [config.py]({file_path})       | The code defines various parameters and settings for training and evaluating a game-playing AI using the Monte Carlo Tree Search algorithm. It includes parameters for memory size, batch size, number of epochs, learning rate, and more. Additionally, it specifies parameters for MCTS simulations during the training and evaluation stages of the AI. These settings determine the behavior and performance of the AI during gameplay and training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [Makefile]({file_path})        | The code in the Makefile provides a set of commands to clean up various files and directories in the omegazero project. The run command exports the current directory to the PYTHONPATH environment variable and then runs the omegazero.py file. The other commands, such as clean_games, clean_artifacts, etc., are used to remove specific files or directories for cleaning purposes. The clean command combines all the individual cleaning commands to provide a comprehensive cleanup of the project.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |

</details>

<details><summary>Entities</summary>

| File                     | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                      | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [player.py]({file_path}) | The code provided defines three classes: Player, LearningPlayer, and StockfishPlayer.-The Player class serves as a base class with a single method, makeMove(), which is not implemented.-The LearningPlayer class inherits from the Player class and adds additional methods for generating exploratory and exploitative moves. It uses a brain object for decision-making and stores the exploration probability and brain instance as attributes.-The StockfishPlayer class also inherits from the Player class and uses the Stockfish chess engine to make moves. It implements a method to evaluate the current chess position using Stockfish and makes a move based on the evaluation. It also stores a brain object and the Stockfish engine as attributes.These classes provide different strategies for making moves in a chess game. |
| [brain.py]({file_path})  | The code represents a `Brain` class that is responsible for the decision-making process in a game-playing AI. It uses Monte Carlo Tree Search (MCTS) algorithm with a neural network to simulate and evaluate different game states. It can learn and train the neural network, select actions based on the current game state and temperature parameter, and execute specific actions based on a move chosen by an external entity (e.g., Stockfish). The class also provides methods for building and updating the MCTS tree structure.                                                                                                                                                                                                                                                                                                       |
| [memory.py]({file_path}) | The code defines a Memory class with methods for managing short-term memory (stmemory) and long-term memory (ltmemory). The ltmemory_nparray method converts the ltmemory into numpy arrays. The commit_stmemory method appends state, policy, and value to the stmemory deque. The commit_ltmemory method moves all entries from stmemory to ltmemory and then clears stmemory. The clear_stmemory method clears the stmemory deque.                                                                                                                                                                                                                                                                                                                                                                                                           |
| [game.py]({file_path})   | The code defines a class `GameState` that represents the state of a chess game. It contains methods to get and take actions, calculate allowed actions, and convert the state to a tensor representation. The `Game` class uses `GameState` and `Player` objects to simulate a chess game and save it as a PGN file. The code also includes helper functions for converting the action values and finding state-action pairs.                                                                                                                                                                                                                                                                                                                                                                                                                   |

</details>

<details><summary>Stages</summary>

| File                       | Summary                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---                        | ---                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [learn.py]({file_path})    | The code is part of an OmegaZero program and specifically represents the learn stage. It imports a LearningPlayer class from the entities.player module and sets the path for artifacts. The LearnStage class has methods for learning and setting input, where learning triggers the learning process in the brain (a neural network model), and setting input sets the brain for the stage. The getOutput method returns the brain. |
| [play.py]({file_path})     | The code represents the play stage of the OmegaZero project. It initializes a brain with old memories and plays a specified number of games against the Stockfish engine. The exploration probability gradually decreases as the games progress using exponential decay. The code tracks move values and updates the brain's memory after each game. The output is the updated brain with new memories.                               |
| [evaluate.py]({file_path}) | The code is part of an evaluation stage in an OmegaZero chess AI. It plays a specified number of games against a Stockfish player to evaluate the performance of a newly trained brain. The code initializes the metrics, creates a new trained brain, and then plays the games. After each game, it determines the result and updates the metrics accordingly-wins, losses, and draws. The final metrics are returned as the output. |

</details>

<details><summary>Algorithms</summary>

| File                   | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---                    | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [mcts.py]({file_path}) | The code represents a Monte Carlo Tree Search (MCTS) algorithm for making decisions in a game. It consists of three classes: Node, Edge, and MCTS. The Node class represents a state of the game, while the Edge class represents a move from one state to another. The MCTS class uses the Node and Edge classes to perform a tree search and make decisions based on the statistics of the edges. The algorithm iteratively expands the tree by adding nodes and edges, and uses a backfilling process to update the statistics of the edges based on the outcomes of simulations.                                             |
| [nn.py]({file_path})   | The code above implements a chess neural network model using PyTorch. The model is defined in the `ChessNet` class, which consists of several convolutional and fully connected layers. The model is trained using the AlphaLoss function, which calculates the loss for the predicted values and policies compared to the ground truth. The training data is loaded using a `CustomDataset` class and a `DataLoader` is used for batch processing. The model is trained for a specified number of epochs, with the optimizer and scheduler handling the learning rate updates. The model is saved after training for later use. |

</details>

---

## 🚀 Getting Started

### 🔧 Installation

1. Clone the  repository:
```sh
git clone git@github.com:glaand/human-vs-chess-rl.git
```

2. Change to the project directory:
```sh
cd human-vs-chess-rl/
```

3. Install the dependencies in a new environment:
```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 🤖 Running 
For running the omegazero, execute the episode batch file
```sh
bash omegazero/run_episodes.sh
```

For running the antichess, execute the python training file
```sh
python antichess/train_antichess.py
```

For running the gui for both components, please execute the following
```sh
python gui/gui.py
```

[**Return**](#Top)

---

