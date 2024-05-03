from itertools import count
import logging
from multiprocessing import Manager
import numpy as np
import os
from pathlib import Path
import time
import yaml
import torch

from c4_gym import create_env, C4Env
from c4_model import C4Model
from c4_player_submission import C4Player
from worker.utils import flags_to_namespace, Stopwatch

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)

class RLAgent:
    game = count()
    def __init__(self):
        with open(CONFIG_PATH, 'r') as file:
            self.flags = flags_to_namespace(yaml.safe_load(file))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.env = create_env(
            self.flags,
            device=self.device,
        )
        self.action_placeholder = torch.ones(1)

        self.m = Manager()

        self.model = C4Model(self.flags, self.device, self.flags.best_model_weight_fname)
        self.pipe_pool = self.m.list(
            [self.model.get_pipes(self.flags.search_threads) for _ in range(self.flags.max_processes)]
        )
        self.game_idx = 0
        self.stopwatch = Stopwatch()

    def __call__(self, obs, conf):
        self.stopwatch.reset()

        self.stopwatch.start("Observation processing")
        env_output = self.preprocess(obs)

        self.stopwatch.stop().start("Model inference")
        pipes = self.pipe_pool.pop()

        player = C4Player(self.flags, pipes)
        action = player.action(self.env, env_output)

        _ = self.env.step(action)

        self.pipe_pool.append(pipes)

        self.stopwatch.stop()

        # value_msg = f"Game: {self.game_idx:>3} | Turn: {obs['step']:>2} | Column:{action} |"
        # timing_msg = f"{str(self.stopwatch)} | "
        # overage_time_msg = f"Remaining overage time: {obs['remainingOverageTime']:.2f}"

        # logger.debug(" - ".join([value_msg, timing_msg, overage_time_msg]))
        return action

    def preprocess(self, obs):
        if obs['step'] == 0:
            self.game_idx = next(RLAgent.game) + 1
            return self.env.reset()
        if obs['step'] == 1:
            self.game_idx = next(RLAgent.game) + 1
            self.env.reset()
        old_board = self.env.board
        new_board = np.array(obs['board']).reshape(old_board.shape)
        difference = np.subtract(new_board, old_board)
        opponent_action = np.argmax(difference) % self.env.game_state.cols
        return self.env.step(opponent_action)


if __name__=="__main__":
    from kaggle_environments import evaluate, make
    env = make('connectx', debug=True)

    env.reset()
    # print(env.run([RLAgent(), RLAgent()]))
    # print(f"\np1 v p2\n{env.render(mode='ansi')}")
    # env.reset()
    # print(env.run([RLAgent(), 'random']))
    # print(f"\np1 v random\n{env.render(mode='ansi')}")
    # env.reset()
    # print(env.run(['negamax', RLAgent()]))
    # print(f"\nnegamax v p2\n{env.render(mode='ansi')}")
    # env.reset()
    # print(env.run([RLAgent(), 'negamax']))
    # print(f"\np1 v negamax\n{env.render(mode='ansi')}")
    # env.reset()

    def print_time(start, end):
        duration = int(end - start)
        hours = duration // 3600
        remaining_duration = duration % 3600
        minutes = remaining_duration // 60
        remaining_duration = remaining_duration % 60
        seconds = int(remaining_duration)
        print(f"That took {hours:02d}:{minutes:02d}:{seconds:02d}  |  (seconds duration: {duration})")

    def mean_reward(rewards, idx):
        wins = sum([1 for r in rewards if r[idx] == 1])
        losses = sum([1 for r in rewards if r[idx] == -1])
        ties = sum([1 for r in rewards if r[idx] == 0])
        return f"Wins: {wins:>3} | Losses: {losses:>3} | Ties: {ties:>3} | Win %: {100 * wins / len(rewards):>5.2f} %"


    # Run multiple episodes to estimate its performance.
    overall_start = time.time()
    section_start = time.time()
    print("RLAgent vs Negamax Agent => ", mean_reward(evaluate("connectx", [RLAgent(), "negamax"], num_episodes=100), idx=0))
    print_time(section_start, time.time())
    section_start = time.time()
    print("Negamax Agent vs RLAgent => ", mean_reward(evaluate("connectx", ["negamax", RLAgent()], num_episodes=100), idx=-1))
    print_time(section_start, time.time())
    section_start = time.time()
    print("RLAgent vs Random Agent => ", mean_reward(evaluate("connectx", [RLAgent(), "random"], num_episodes=100), idx=0))
    print_time(section_start, time.time())
    section_start = time.time()
    print("Random Agent vs RLAgent => ", mean_reward(evaluate("connectx", ["random", RLAgent()], num_episodes=100), idx=-1))
    print_time(section_start, time.time())
    print("Overall duration:")
    print_time(overall_start, time.time())