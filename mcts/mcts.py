import scipy

from c4_gym import create_env
from c4_game.game import Game
from nns import create_model
from node import Node

def run_mcts(flags):
    env = create_env(flags)
    output = env.reset()

    net = create_model(flags)
    leaf = Node(flags, output, action=None)

    env.step(leaf.action)

    for _ in flags.max_steps:
        leaf = leaf.get_next_action()
        env.step(leaf.action)
        probs, value = net(output['obs'])
        leaf.backward(probs, value)