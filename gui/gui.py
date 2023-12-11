import tkinter as tk
from tkinter import filedialog, ttk
import chess
import chess.pgn
from io import BytesIO
import cairosvg
from PIL import Image, ImageTk
import os

import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import numpy as np
import chess.variant
import chess.pgn
import chess.svg
from io import BytesIO
import cairosvg
from PIL import Image, ImageTk
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + '/../antichess')
from board_function import board_to_input_array

# Load the model
best_player_model = load_model("antichess/model/best_player_antichess.h5")

class AntichessApp:
    """
    A class representing the Antichess application.

    Attributes:
        root (tk.Tk): The root window of the application.
        board (chess.variant.GiveawayBoard): The chess board.
        player_color (str): The color chosen by the player.

    Methods:
        __init__(self, root): Initializes the AntichessApp object.
        choose_color(self): Displays the color selection interface.
        confirm_color(self): Confirms the player's color choice.
        make_move(self): Handles the player's move.
        make_bot_move(self): Makes a move for the bot.
        choose_action(self): Chooses the best action for the bot.
        update_board_display(self): Updates the display of the chess board.
        reset_game(self): Resets the game to the initial state.
    """
    def __init__(self, root):
        self.root = root

        # Initialize session state
        self.board = chess.variant.GiveawayBoard()
        self.player_color = None

        # Chess board display
        self.board_canvas = tk.Canvas(root, width=400, height=400)
        self.board_canvas.pack()

        # Buttons and entry
        self.move_entry = tk.Entry(root)
        self.move_entry.pack()
        self.make_move_button = tk.Button(root, text="Make Move", command=self.make_move)
        self.make_move_button.pack()
        self.reset_button = tk.Button(root, text="Reset Game", command=self.reset_game)
        self.reset_button.pack()

        self.update_board_display()

        # Initialize the GUI
        self.choose_color()

    def choose_color(self):
        color_label = tk.Label(self.root, text="Choose your color:")
        color_label.pack()

        self.color_var = tk.StringVar()
        radio_white = tk.Radiobutton(self.root, text="White", variable=self.color_var, value="white")
        radio_black = tk.Radiobutton(self.root, text="Black", variable=self.color_var, value="black")
        radio_white.pack()
        radio_black.pack()

        confirm_button = tk.Button(self.root, text="Confirm Color", command=self.confirm_color)
        confirm_button.pack()

    def confirm_color(self):
        self.player_color = self.color_var.get()

        if self.player_color == "black":
            self.make_bot_move()

        self.update_board_display()

    def make_move(self):
        move_input = self.move_entry.get().strip()
        print(self.board.legal_moves)
        print(move_input)
        try:
            move = chess.Move.from_uci(move_input)
            if move not in self.board.legal_moves:
                messagebox.showerror("Error", "Illegal move")
                return

            capturing_moves = [m for m in self.board.legal_moves if self.board.is_capture(m)]
            if capturing_moves and move not in capturing_moves:
                messagebox.showerror("Error", "A capture is available, and you must capture.")
                return

            self.board.push(move)
            self.make_bot_move()
            self.update_board_display()

            if self.board.is_game_over():
                messagebox.showinfo("Game Over", f"Game over: {self.board.result()}")
                self.reset_game()

        except ValueError:
            messagebox.showerror("Error", "Invalid move")

    def make_bot_move(self):
        if not self.board.is_game_over() and self.board.turn != (self.player_color == 'white'):
            bot_move = self.choose_action()
            self.board.push(bot_move)

    def choose_action(self):
        legal_moves_list = list(self.board.legal_moves)
        if not legal_moves_list:
            return chess.Move.null()

        capturing_moves = [move for move in legal_moves_list if self.board.is_capture(move)]

        if capturing_moves:
            legal_moves_list = capturing_moves

        q_values = best_player_model.predict(np.array([board_to_input_array(self.board)]))[0]
        best_move_index = np.argmax(q_values)
        best_move_uci = legal_moves_list[min(best_move_index, len(legal_moves_list) - 1)].uci()
        return chess.Move.from_uci(best_move_uci)

    def update_board_display(self):
        flipped = False
        if self.player_color == "black":
            flipped = True
        self.board_canvas.delete("all")
        board_svg = chess.svg.board(board=self.board, size=400, flipped=flipped)
        png_data = cairosvg.svg2png(board_svg.encode("utf-8"))
        board_image = Image.open(BytesIO(png_data))
        board_image = ImageTk.PhotoImage(board_image)
        self.board_canvas.create_image(0, 0, anchor="nw", image=board_image)
        self.board_canvas.image = board_image

    def reset_game(self):
        self.board = chess.variant.GiveawayBoard()
        self.confirm_color()

class ChessBoard:
    """
    Represents a chess board in the GUI.

    Attributes:
    - gui_obj: The GUI object associated with the chess board.
    - img: The image of the chess board.
    - game_moves: A list of moves in the chess game.
    - current_move: The index of the current move.
    - row: The row position of the chess board.
    - col: The column position of the chess board.
    - states: A list of chess board states.
    - photo_img: The image of the chess board as a PhotoImage.

    Methods:
    - load_pgn(pgn_file_path): Loads a PGN file and initializes the chess board.
    - draw_board(): Draws the chess board on the GUI canvas.
    - prev_move(): Moves to the previous move in the game.
    - next_move(): Moves to the next move in the game.
    """
    def __init__(self, gui_obj, row, col):
        self.gui_obj = gui_obj
        self.img = None
        self.game_moves = []  
        self.current_move = 0
        self.row = row
        self.col = col
        self.states = []
        self.photo_img = None

    def load_pgn(self, pgn_file_path):
        """
        Loads a PGN file and initializes the chess board.

        Parameters:
        - pgn_file_path: The file path of the PGN file.

        Raises:
        - Exception: If there is an error reading the PGN file.
        """
        try:
            with open(pgn_file_path, "r") as file:
                game = chess.pgn.read_game(file)
                self.chessboard = game.board()
                self.game_moves = [move for move in game.mainline_moves()]
                self.states = [self.chessboard.copy()]
        except Exception as e:
            print(f"Error reading PGN file: {e}")

    def draw_board(self):
        """
        Draws the chess board on the GUI canvas.
        """
        # Calculate the position based on canvas size and percentage
        min_size = min(self.gui_obj.current_width, self.gui_obj.current_height)
        x = self.col * (min_size // 3)
        y = self.row * (min_size // 3)

        svg_data = chess.svg.board(self.chessboard, size=min_size // 3)
        png_data = cairosvg.svg2png(svg_data.encode("utf-8"))

        # Create a PhotoImage from the PNG data
        img = Image.open(BytesIO(png_data))

        # Apply a grey overlay

        if self.current_move == len(self.game_moves):
            # If the game is over, apply a green overlay
            green_overlay = Image.new('RGBA', img.size, (0, 128, 0, 200))
            img = Image.alpha_composite(img.convert('RGBA'), green_overlay)

        # Convert the modified image back to PhotoImage
        self.photo_img = ImageTk.PhotoImage(img)

        print(f"Drawing board at position {x}, {y}")

        self.gui_obj.canvas.create_image(x, y, anchor=tk.NW, image=self.photo_img)

    def prev_move(self):
        """
        Moves to the previous move in the game.
        """
        # Get the previous move from the game
        if self.current_move > 0:
            self.chessboard = self.states[self.current_move-1].copy()
            self.current_move -= 1
            self.draw_board()

    def next_move(self):
        """
        Moves to the next move in the game.
        """
        # Get the next move from the game
        if self.current_move < len(self.game_moves):
            if self.current_move + 1 not in range(len(self.states)):
                self.chessboard.push(self.game_moves[self.current_move])
                self.states.append(self.chessboard.copy())
            else:
                self.chessboard = self.states[self.current_move + 1].copy() 
            self.current_move += 1
            self.draw_board()

class GUI(tk.Tk):
    """
    The main GUI class for the ChessbotRL application.

    Args:
        num_games (int): The number of games to load.

    Attributes:
        dir_path (str): The path of the selected folder.
        num_games (int): The number of games to load.
        current_width (int): The current width of the GUI window.
        current_height (int): The current height of the GUI window.
        title (str): The title of the GUI window.
        tab_parent (ttk.Notebook): The notebook widget for managing tabs.
        tab_antichess (tk.Frame): The frame for the Antichess tab.
        tab_omegazero (tk.Frame): The frame for the OmegaZero tab.
        antichess_app (AntichessApp): The Antichess application.
        button_frame (tk.Frame): The frame for the buttons.
        next_button (tk.Button): The button for the next move.
        prev_button (tk.Button): The button for the previous move.
        reset_button (tk.Button): The button for resetting the boards.
        select_folder_button (tk.Button): The button for selecting a folder.
        canvas (tk.Canvas): The canvas for drawing the boards.
        boards (list): The list of ChessBoard objects.
    """
    def __init__(self, num_games):
        super().__init__()

        self.dir_path = None
        self.num_games = num_games

        self.current_width = 700
        self.current_height = 800

        self.title("ChessbotRL")
        self.geometry(f"{self.current_width}x{self.current_height}")
        self.resizable(True, True)  # Allow window to be resizable

        self.tab_parent = ttk.Notebook(self)

        self.tab_antichess = tk.Frame(self.tab_parent)
        self.tab_omegazero = tk.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_antichess, text="Antichess")
        self.tab_parent.add(self.tab_omegazero, text="OmegaZero")
        self.tab_parent.pack(expand=1, fill='both')

        self.antichess_app = AntichessApp(self.tab_antichess)
        
        # Add Previous Move and Next Move buttons
        self.button_frame = tk.Frame(self.tab_omegazero)
        self.button_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.next_button = tk.Button(self.button_frame, text="Next Move", command=self.next_move)
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.prev_button = tk.Button(self.button_frame, text="Previous Move", command=self.prev_move)
        self.prev_button.pack(side=tk.RIGHT, padx=10, pady=10)  

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset_boards)
        self.reset_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Add buton to select folder
        self.select_folder_button = tk.Button(self.button_frame, text="Select Folder", command=self.select_folder)
        self.select_folder_button.pack(side=tk.RIGHT, padx=10, pady=10)      

        self.canvas = tk.Canvas(self.tab_omegazero, width=self.current_width, height=self.current_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)  # Expand canvas to fill the window
        self.canvas.bind("<Key>", self.key_pressed)

        self.boards = []

        # Bind the Configure event to handle resizing
        self.canvas.bind("<Configure>", self.handle_resize)
        self.canvas.focus_set()

    def select_folder(self):
        self.dir_path = filedialog.askdirectory()
        if not self.dir_path:
            print("No folder selected")
            exit()
        self.load_games()

    def key_pressed(self, event):
        print(f"Key pressed: {event}")
        if event.keysym == "Left":
            self.prev_move()
        elif event.keysym == "Right":
            self.next_move()
        elif event.keysym == "r":
            self.reset_boards()

    def load_games(self):
        # Get a list of all PGN files in the directory
        pgn_files = [f for f in os.listdir(self.dir_path) if f.endswith('.pgn')]

        # Load a maximum of 9 PGN files
        for i in range(min(self.num_games, len(pgn_files))):
            # Calculate the row and column based on the index
            row = i // 3  # Assuming you want 3 boards in each row
            col = i % 3   # Assuming you want 3 boards in each row

            board = ChessBoard(self, row, col)
            pgn_file_path = os.path.join(self.dir_path, pgn_files[i])
            board.load_pgn(pgn_file_path)
            board.draw_board()
            self.boards.append(board)

    def handle_resize(self, event):
        # Clear the canvas before re-drawing the boards
        self.canvas.delete("all")

        # Re-draw the boards with the updated canvas size
        for i, board in enumerate(self.boards):
            self.current_width = event.width
            self.current_height = event.height
            board.gui_object = self
            board.draw_board()

    def prev_move(self):
        self.canvas.focus_set()
        for board in self.boards:
            board.prev_move()
    
    def next_move(self):
        self.canvas.focus_set()
        for board in self.boards:
            board.next_move()

    def reset_boards(self):
        self.canvas.focus_set()
        for board in self.boards:
            board.current_move = 0
            board.chessboard = board.states[0].copy()
            board.draw_board()

if __name__ == "__main__":
    app = GUI(num_games=9)
    app.mainloop()
