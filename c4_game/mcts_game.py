from game import Game

# https://github.com/Zeta36/chess-alpha-zero/blob/master/src/chess_zero/agent/player_chess.py#L187

class MCTSGame(Game):
    def __init__(self):
        super().__init__()

    def action_mask(self):
        pass

    def is_winner(self):
        pass