import copy
from collections import defaultdict
import math
import numpy as np

from c4_game.game import Game
from utility_constants import BOARD_SIZE

N_ACTIONS = BOARD_SIZE[-1]


class RootNode:
    def __init__(self):
        self.parent = None

        self.is_leaf = True

        self.children = {}
        self.n = 0 # Number of visits
        self.w = 0. # Total value of the next state
        self.p = 0. # Move probabilities
        self.next_action_probs = np.zeros(N_ACTIONS, dtype=np.float32)
        self.next_action_visits = np.zeros(N_ACTIONS, dtype=np.int64)

    @property
    def q(self):
        '''
        W / N
        :return: Mean value of the next state
        '''
        assert self.n > 0, "Need to increment n visits before getting q"
        return self.w / self.n


class ChildNode(RootNode):
    def __init__(
            self,
            flags,
            game_state: Game,
            action: int,
            parent: RootNode=RootNode()
    ):
        super().__init__()
        self.flags = flags
        self.game_state = game_state
        self.action = action
        self.parent = parent

    def self_visits(self):
        return self.parent.next_action_visits[self.action]

    def increment_self_visits(self):
        self.parent.next_action_visits[self.action] += 1


    def update_probs(self, probs):
        self.next_action_probs = probs

    @property
    def u(self):
        return self.next_action_probs * math.sqrt(self.self_visits) / (1 + self.next_action_visits)

    @property
    def q(self):
        '''
        :return: The mean value of the next state
        '''
        next_action_visits = self.next_action_visits + np.where(self.next_action_visits == 0,
                                                                self.flags.unexplored_action,
                                                                0)
        return self.w / next_action_visits

    def update_params(self, value):
        self.increment_self_visits()
        self.w += value


    def backward(self, probs: np.ndarry, value: float):
        self.update_probs(probs)

        current = self
        while current.parent:
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
        self.is_leaf = False
        if not self.children.get(action, None):
            self.children[action] = ChildNode(self.flags,
                                              copy.deepcopy(self.game_state),
                                              action,
                                              parent=self)
        return self.children[action]

    def get_leaf(self):
        current = self
        while not current.is_leaf:
            current = current.get_next_action()

        return current


if __name__=="__main__":
    flags = None

    root = ChildNode(flags, Game(), action=None)