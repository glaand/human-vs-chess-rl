import tkinter as tk
from tkinter import filedialog
import chess
import chess.pgn
from io import BytesIO
import cairosvg
from PIL import Image, ImageTk

class ChessBoard:
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
        try:
            with open(pgn_file_path, "r") as file:
                game = chess.pgn.read_game(file)
                self.chessboard = game.board()
                self.game_moves = [move for move in game.mainline_moves()]
                self.states = [self.chessboard.copy()]
        except Exception as e:
            print(f"Error reading PGN file: {e}")

    def draw_board(self):
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
        # Get the previous move from the game
        if self.current_move > 0:
            self.chessboard = self.states[self.current_move-1].copy()
            self.current_move -= 1
            self.draw_board()

    def next_move(self):
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
    def __init__(self, num_games):
        super().__init__()

        self.dir_path = None
        self.num_games = num_games

        self.current_width = 800
        self.current_height = 600

        self.title("OmegaZero - Evaluate games")
        self.geometry(f"{self.current_width}x{self.current_height}")
        self.resizable(True, True)  # Allow window to be resizable

        self.dir_path = filedialog.askdirectory()
        if not self.dir_path:
            print("No folder selected")
            exit()
        
        # Add Previous Move and Next Move buttons
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.next_button = tk.Button(self.button_frame, text="Next Move", command=self.next_move)
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.prev_button = tk.Button(self.button_frame, text="Previous Move", command=self.prev_move)
        self.prev_button.pack(side=tk.RIGHT, padx=10, pady=10)        

        self.canvas = tk.Canvas(self, width=self.current_width, height=self.current_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)  # Expand canvas to fill the window
        self.canvas.bind("<Key>", self.key_pressed)

        self.boards = []
        self.load_games()


        # Bind the Configure event to handle resizing
        self.canvas.bind("<Configure>", self.handle_resize)
        self.canvas.focus_set()

    def key_pressed(self, event):
        print(f"Key pressed: {event}")
        if event.keysym == "Left":
            self.prev_move()
        elif event.keysym == "Right":
            self.next_move()
        elif event.keysym == "r":
            self.reset_boards()

    def load_games(self):
        for i in range(self.num_games):
            # Calculate the row and column based on the index
            row = i // 3  # Assuming you want 3 boards in each row
            col = i % 3   # Assuming you want 3 boards in each row

            board = ChessBoard(self, row, col)
            board.load_pgn(f"{self.dir_path}/game_{i}.pgn")
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
        for board in self.boards:
            board.prev_move()
    
    def next_move(self):
        for board in self.boards:
            board.next_move()

    def reset_boards(self):
        for board in self.boards:
            board.current_move = 0
            board.chessboard = board.states[0].copy()
            board.draw_board()

if __name__ == "__main__":
    app = GUI(num_games=9)
    app.mainloop()
