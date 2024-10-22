from logging import getLogger
import multiprocessing as mp
from multiprocessing import Manager
import numpy as np
import os
from pathlib import Path
import yaml
import torch

from c4_gym import create_env
from agent.c4_model import C4Model
from agent.c4_player_submission import C4Player
from agent.utils import flags_to_namespace, Stopwatch

CONFIG_PATH = Path("/kaggle_simulations/agent/config.yaml")
MODEL_PATH = Path("/kaggle_simulations/agent/mcts_phase4.pt")
# CONFIG_PATH = Path("C:/Users/nick.gorichs/PycharmProjects/MCTS-Connect-Four/config.yaml")
# MODEL_PATH = Path("C:/Users/nick.gorichs/PycharmProjects/MCTS-Connect-Four/models/mcts_phase4.pt")

# logger = getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"
AGENT = None
mp.set_start_method("spawn")

class RLAgent:
    def __init__(self, *args):
        with open(CONFIG_PATH, 'r') as file:
            self.flags = flags_to_namespace(yaml.safe_load(file))
        self.model_path = MODEL_PATH

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.env = create_env(
            self.flags,
            device=self.device,
        )
        self.action_placeholder = torch.ones(1)

        self.m = Manager()

        model = C4Model(self.flags, self.device, self.model_path)
        self.player = C4Player(self.flags, model)
        self.stopwatch = Stopwatch()

    def __call__(self, obs, conf):
        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        env_output = self.preprocess(obs)

        self.stopwatch.stop().start("Model inference")
        action = self.player.action(self.env, env_output)

        self.stopwatch.stop().start("Internal env stepping")
        _ = self.env.step(action)

        self.stopwatch.stop()

        value_msg = f"Turn: {obs['step']}"
        timing_msg = f"{str(self.stopwatch)}"
        overage_time_msg = f"Remaining overage time: {obs['remainingOverageTime']:.2f}"

        print(" - ".join([value_msg, timing_msg, overage_time_msg]))

        return action

    def preprocess(self, obs):
        if obs['step'] == 0:
            print(self.player.tree)
            print("Resetting")
            self.player.reset()
            return self.env.reset()
        if obs['step'] == 1:
            print(self.player.tree)
            print("Resetting")
            self.player.reset()
            self.env.reset()
        old_board = self.env.board
        new_board = np.array(obs['board']).reshape(old_board.shape)
        difference = np.subtract(new_board, old_board)
        opponent_action = np.argmax(difference) % self.env.game_state.cols
        return self.env.step(opponent_action)


def agent(obs, conf):
    global AGENT
    if AGENT is None:
        AGENT = RLAgent()
    return AGENT(obs, conf)


# if __name__=="__main__":
#     from kaggle_environments import evaluate, make
#     import time
#
#     env = make('connectx', debug=False)
#
#     env.reset()
#     env.run([agent, agent])
#     print(f"\nagent v agent\n{env.render(mode='ansi')}")
#     env.reset()
#     env.run([agent, 'negamax'])
#     print(f"\nagent v negamax\n{env.render(mode='ansi')}")
#     env.reset()
#     env.run(['negamax', agent])
#     print(f"\nnegamax v agent\n{env.render(mode='ansi')}")
#     env.reset()

    # def print_time(start, end):
    #     duration = int(end - start)
    #     hours = duration // 3600
    #     remaining_duration = duration % 3600
    #     minutes = remaining_duration // 60
    #     remaining_duration = remaining_duration % 60
    #     seconds = int(remaining_duration)
    #     print(f"That took {hours:02d}:{minutes:02d}:{seconds:02d}  |  (seconds duration: {duration})")

    # def mean_reward(rewards, idx):
    #     wins = sum([1 for r in rewards if r[idx] == 1])
    #     losses = sum([1 for r in rewards if r[idx] == -1])
    #     ties = sum([1 for r in rewards if r[idx] == 0])
    #     return f"Wins: {wins:>3} | Losses: {losses:>3} | Ties: {ties:>3} | Win %: {100 * wins / len(rewards):>5.2f} %"

    # # Run multiple episodes to estimate its performance.
    # overall_start = time.time()
    # section_start = time.time()
    # print("RLAgent vs Negamax Agent => ", mean_reward(evaluate("connectx", [RLAgent(), "negamax"], num_episodes=100), idx=0))
    # print_time(section_start, time.time())
    # section_start = time.time()
    # print("Negamax Agent vs RLAgent => ", mean_reward(evaluate("connectx", ["negamax", RLAgent()], num_episodes=100), idx=-1))
    # print_time(section_start, time.time())
    # section_start = time.time()
    # print("RLAgent vs Random Agent => ", mean_reward(evaluate("connectx", [RLAgent(), "random"], num_episodes=100), idx=0))
    # print_time(section_start, time.time())
    # section_start = time.time()
    # print("Random Agent vs RLAgent => ", mean_reward(evaluate("connectx", ["random", RLAgent()], num_episodes=100), idx=-1))
    # print_time(section_start, time.time())
    # print("Overall duration:")
    # print_time(overall_start, time.time())