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
        self.v = 0

        self.game_state = game_state
        self.action = action
        self.has_children = False

    def build_children(self):
        if not self.has_children:
            for action in range(BOARD_SIZE[-1]):
                setattr(self, f"child_{action}", Node(self.flags, self.game_state, self.action, parent=self))
            self.has_children = True

    @property
    def u(self):
        return math.sqrt(self.n) * abs()

    def update(self, value):
        self.n += 1
        self.w += value
        self.q = self.w / self.n

        self.build_children()

    def backward(self, value: float):
        current = self
        while current.parent:
            current.update(current.game_state.value_modifier * value)
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

    # def best_child(self):

if __name__=="__main__":
    class DummyNode(object):
        def __init__(self):
            self.parent = None
    flags = None

    root = Node(flags, Game(), action=None, parent=DummyNode())
    root.update(0.1)
    print(vars(root))