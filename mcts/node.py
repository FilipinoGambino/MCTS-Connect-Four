from collections import defaultdict
import math
import numpy as np

from c4_game.game import Game
from utility_constants import BOARD_SIZE


class Node:
    def __init__(self, flags, game_state: Game, action: int, parent: object=None):
        self.flags = flags
        self.parent = parent

        self.w = 0 # The total value of the next state
        self.q = 0 # The mean value of the next state
        self.n = 0 # The number of times action a has been taken
        self.p = 0 # The prior probability of selecting action a

        self.n_visits = 0
        self.total_value = 0
        self.game_state = game_state
        self.action = action
        self.is_expanded = False

        self.children = {action: None for action in range(BOARD_SIZE[-1])}
        self.children_values = np.zeros(BOARD_SIZE[-1], dtype=np.float32)
        self.children_probs = np.zeros(BOARD_SIZE[-1], dtype=np.float32)
        self.children_n_visits = np.zeros(BOARD_SIZE[-1], dtype=np.int64)


    def expand(self, child_probs):
        self.is_expanded = True


    def backward(self, value: float):
        current = self
        while current.parent:
            current.n_visits += 1
            current.total_value += current.game_state.value_modifier * value
            current = current.parent

    @property
    def n_b(self):
        return sum([child.n for child in self.children])

    @property
    def u(self):
        '''
        :return: Q[s,a] + c * P[s,a] * sqrt(sum(N[s,b])) / (1 + N[s,a])
        '''
        return self.q + self.flags.c * self.p * math.sqrt(self.n_b) / (1 + self.n)

    def best_child(self):
