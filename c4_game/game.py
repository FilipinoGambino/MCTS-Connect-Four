import numpy as np


class Game:
    def __init__(self, config):
        self.config = config
        self.rows = config['rows']
        self.cols = config['columns']
        self.board_dims = (self.rows, self.cols)
        self.in_a_row = config['inarow']
        self.board = np.zeros(shape=self.board_dims, dtype=np.uint8)

    def step(self, action):
        row = self.get_lowest_available_row(action)
        self.board[row, action] = self.current_player_mark

    def get_lowest_available_row(self, column):
        for row in range(self.rows-1, -1, -1):
            if self.board[row,column] == 0:
                return row
        raise StopIteration(f"Column {column} is full.\n{self.board}")

    @property
    def current_player_mark(self):
        return self.p1_mark if self.is_p1_turn else self.p2_mark

    @property
    def turn(self):
        return np.count_nonzero(self.board)

    @property
    def is_p1_turn(self):
        return self.turn % 2 == 0

    @property
    def p1_mark(self):
        return 1

    @property
    def p2_mark(self):
        return 2

    @property
    def max_turns(self):
        if isinstance(self.board, np.ndarray):
            return self.board.size
        elif isinstance(self.board, list):
            return len(self.board)
        else:
            raise NotImplementedError