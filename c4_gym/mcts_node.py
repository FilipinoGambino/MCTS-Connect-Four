import copy
from collections import defaultdict
import math
import numpy as np

from c4_game.game import Game
from utility_constants import BOARD_SIZE

N_ACTIONS = BOARD_SIZE[-1]


class DummyNode:
    def __init__(self):
        self.parent = None


class Node:
    def __init__(
            self,
            flags,
            action: int,
            parent=DummyNode()
    ):
        self.flags = flags
        self.game_state = None
        self.action = action
        self.parent = parent

        self.is_leaf = True

        self.children = {x: None for x in range(N_ACTIONS)}
        self.n = 0 # Number of visits
        self.w = 0. # Total value of the next state
        self.next_action_probs = np.zeros(shape=(N_ACTIONS, 1), dtype=np.float32)

    @property
    def q(self):
        '''
        W / N
        :return: Mean value of the next state
        '''
        assert self.n > 0, "Division by zero; increment n"
        return self.w / self.n

    @property
    def u(self):
        return self.children_p * math.sqrt(self.n) / (1 + self.children_n)

    def children_n(self):
        child_visits = np.zeros(shape=(N_ACTIONS, 1))
        for action, child in self.children.items():
            if child:
                child_visits[action] = child.n
        return child_visits

    def children_p(self):
        child_probs = np.zeros(shape=(N_ACTIONS, 1), dtype=np.float32)
        for action, child in self.children.items():
            if child:
                child_probs[action] = child.p
        return child_probs

    def update_params(self, value):
        self.n += 1
        self.w += value

    def update_probs(self, probs):
        assert probs.shape == self.next_action_probs.shape
        probs.to(dtype=self.next_action_probs.dtype)
        self.next_action_probs = probs

    def backward(self, probs: np.ndarry, value: float):
        self.update_probs(probs)

        current = self
        while current.parent:
            # TODO: Check and see if the value modifier needs to be different for each parent
            #  (like alternating per turn on the way back up)
            current.update_params(current.game_state.value_modifier * value)
            current = current.parent

    @property
    def best_action(self):
        '''
        :return: max action index from Q[s,a] + c * P[s,a] * sqrt(sum(N[s,b])) / (1 + N[s,a])
        '''
        return np.argmax(self.q + self.flags.c * self.u)

    def get_next_action(self):
        action = self.best_action
        if not self.children[action]:
            self.children[action] = Node(self.flags,
                                         action,
                                         parent=self)
            self.is_leaf = False
        return self.children[action]


def collect_tree_probs(root: Node):
    # TODO tree traversal
    pass


if __name__=="__main__":
    flags = None

    root = Node(flags, action=None)