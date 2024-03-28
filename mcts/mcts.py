import scipy

from c4_game.game import Game
from nns import create_model
from node import ChildNode

def train(flags):
    net = create_model(flags)
    root = ChildNode(flags, Game(), action=None)
    for _ in flags.max_steps:
        leaf = root.get_next_action()
        logits, value = net(leaf.game_state)
        probs = scipy.special.softmax(logits, axis=-1)
        if leaf.game_state.check_winner() or leaf.game_state.actions():
            leaf.backward(probs, value)