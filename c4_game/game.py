import numpy as np

from utility_constants import BOARD_SIZE


class Game:
    def __init__(self):
        self.board = np.zeros(shape=BOARD_SIZE, dtype=np.uint8)
        self.turn = 0

    def update(self, obs):
        self.board = np.array(obs['board'], dtype=np.uint8).reshape(BOARD_SIZE)
        self.turn = obs['step']

    def step(self, action):
        row = self.get_lowest_available_row(action)
        self.board[row, action] = self.current_player
        self.turn += 1

    def get_lowest_available_row(self, column):
        for row in range(BOARD_SIZE[0]-1,-1,-1):
            if self.board[row,column] == 0:
                return row
        raise StopIteration(f"Column {column} is full. {self.board}")

    @property
    def current_player(self):
        '''
        :return: Either 1 or 2 depending on which player
        '''
        return self.turn % 2 + 1

    @property
    def value_modifier(self):
        '''
        :return: Either -1 or 1 depending on which player
        '''
        return (self.current_player - 1) * 2 - 1

    @property
    def max_turns(self):
        if isinstance(self.board, np.ndarray):
            return self.board.size
        elif isinstance(self.board, list):
            return len(self.board)
        else:
            raise NotImplementedError