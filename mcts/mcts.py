import numpy as np

from c4_game.game import Game
from node import Node

def train(flags):
    class DummyNode(object):
        def __init__(self):
            self.parent = None

    root = Node(flags, Game(), action=None, parent=DummyNode())
    for _ in flags.leaf_iterations:
        leaf = root.select_leaf()
        probs, value = net(leaf.game_state)
        if leaf.game_state.check_winner() or leaf.game_state.actions():
            leaf.backup(value)
            continue
        leaf.expand(probs)
        leaf.backup(value)