from collections import deque
import numpy as np


class Game:
    def __init__(self, config):
        self.config = config
        self.rows = config['rows']
        self.cols = config['cols']
        self.board_dims = (self.rows, self.cols)
        self.in_a_row = config['inarow']
        self.board = np.zeros(shape=self.board_dims, dtype=np.uint8)
        self.history = deque(maxlen=8)

        for _ in range(self.history.maxlen):
            self.history.append(self.board)

    def update(self, obs):
        self.board = np.array(obs['board'], dtype=np.uint8).reshape(self.board_dims)
        self.history.append(self.board)

    def step(self, action):
        row = self.get_lowest_available_row(action)
        self.board[row, action] = self.current_player_mark
        self.history.append(self.board)

    def get_lowest_available_row(self, column):
        for row in range(self.rows[0]-1, -1, -1):
            if self.board[row,column] == 0:
                return row
        raise StopIteration(f"Column {column} is full. {self.board}")

    @property
    def current_player_mark(self):
        return self.p1_mark if self.p1_turn else self.p2_mark

    @property
    def turn(self):
        return np.count_nonzero(self.board)

    @property
    def p1_turn(self):
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