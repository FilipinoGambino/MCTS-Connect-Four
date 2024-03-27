import numpy as np

from c4_game.game import Game
from nns import create_model
from node import ChildNode

def train(flags):
    net = create_model(flags)
    root = ChildNode(flags, Game(), action=None)
    for _ in flags.leaf_iterations:
        leaf = root.select_leaf()
        probs, value = net(leaf.game_state)
        if leaf.game_state.check_winner() or leaf.game_state.actions():
            leaf.backup(value)
            continue
        leaf.expand(probs)
        leaf.backup(value)