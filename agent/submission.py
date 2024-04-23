from itertools import count
import logging
from multiprocessing import Manager
import os
from pathlib import Path
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
            device=self.device
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
        self.preprocess(obs)

        env_output = self.get_env_output()

        self.stopwatch.stop().start("Model inference")
        pipes = self.pipe_pool.pop()

        player = C4Player(self.flags, pipes)
        action = player.action(self.env, env_output)
        self.env.step(action)

        self.pipe_pool.append(pipes)

        self.stopwatch.stop()

        value_msg = f"Game: {self.game_idx:>3} | Turn: {obs['step']:>2} | Column:{action} |"
        timing_msg = f"{str(self.stopwatch)} | "
        overage_time_msg = f"Remaining overage time: {obs['remainingOverageTime']:.2f}"

        logger.debug(" - ".join([value_msg, timing_msg, overage_time_msg]))
        return action

    def get_env_output(self):
        return self.env.step(self.action_placeholder)

    def preprocess(self, obs):
        if obs['step'] == 0 or obs['step'] == 1:
            self.unwrapped_env.reset()
            self.game_idx = next(RLAgent.game) + 1
        else:
            self.unwrapped_env.manual_step(obs)

    @property
    def unwrapped_env(self) -> C4Env:
        return self.env.unwrapped


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

    def mean_reward1(rewards):
        return sum(r[0] for r in rewards) / float(len(rewards))

    def mean_reward2(rewards):
        return sum(r[-1] for r in rewards) / float(len(rewards))


    # Run multiple episodes to estimate its performance.
    print(evaluate("connectx", [RLAgent(), "negamax"], num_episodes=10))
    # print("My Agent vs Random Agent: ", mean_reward1(evaluate("connectx", [RLAgent(), "random"], num_episodes=100)))
    # print("My Agent vs Random Agent: ", mean_reward2(evaluate("connectx", ["random", RLAgent()], num_episodes=100)))
    print("My Agent vs Negamax Agent: ", mean_reward1(evaluate("connectx", [RLAgent(), "negamax"], num_episodes=10)))
    print("My Agent vs Negamax Agent: ", mean_reward2(evaluate("connectx", ["negamax", RLAgent()], num_episodes=10)))