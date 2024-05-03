#!/usr/bin/env python
import contextlib as __stickytape_contextlib

@__stickytape_contextlib.contextmanager
def __stickytape_temporary_dir():
    import tempfile
    import shutil
    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)

with __stickytape_temporary_dir() as __stickytape_working_dir:
    def __stickytape_write_module(path, contents):
        import os, os.path

        def make_package(path):
            parts = path.split("/")
            partial_path = __stickytape_working_dir
            for part in parts:
                partial_path = os.path.join(partial_path, part)
                if not os.path.exists(partial_path):
                    os.mkdir(partial_path)
                    with open(os.path.join(partial_path, "__init__.py"), "wb") as f:
                        f.write(b"\n")

        make_package(os.path.dirname(path))

        full_path = os.path.join(__stickytape_working_dir, path)
        with open(full_path, "wb") as module_file:
            module_file.write(contents)

    import sys as __stickytape_sys
    __stickytape_sys.path.insert(0, __stickytape_working_dir)

    __stickytape_write_module('c4_model.py', b'"""\r\nDefines the actual model for making policy and value predictions given an observation.\r\n"""\r\n\r\nfrom pathlib import Path\r\nimport torch\r\nfrom logging import getLogger\r\nfrom types import SimpleNamespace\r\n\r\nfrom agent.nns import create_model\r\nfrom agent.c4_api import C4API\r\n\r\n\r\nlogger = getLogger(__name__)\r\n\r\n\r\nclass C4Model:\r\n    """\r\n    The model which can be trained to take observations of a game of chess and return value and policy\r\n    predictions.\r\n\r\n    Attributes:\r\n        :ivar Config config: configuration to use\r\n        :ivar Model model: the PyTorch model to use for predictions\r\n        :ivar ChessModelAPI api: the api to use to listen for and then return this models predictions (on a pipe).\r\n    """\r\n\r\n    def __init__(self, flags: SimpleNamespace, device, fname=None):\r\n        self.flags = flags\r\n        self.device = device\r\n        self.api = None\r\n        self.model = self.build_and_load_model(fname)\r\n\r\n    def get_pipes(self, n_pipes=1):\r\n        """\r\n        Creates a list of pipes on which observations of the game state will be listened for. Whenever\r\n        an observation comes in, returns policy and value network predictions on that pipe.\r\n\r\n        :param int n_pipes: number of pipes to create\r\n        :return str(Connection): a list of all connections to the pipes that were created\r\n        """\r\n        if self.api is None:\r\n            self.api = C4API(self)\r\n            pipes = [self.api.create_pipe() for _ in range(n_pipes)]\r\n            self.api.start()\r\n            return pipes\r\n        return [self.api.create_pipe() for _ in range(n_pipes)]\r\n\r\n    def build_and_load_model(self, fname):\r\n        model = create_model(self.flags, device=self.device)\r\n        if self.flags.worker_type == \'optimize\':\r\n            model.train()\r\n        else:\r\n            model.eval()\r\n        model = model.share_memory()\r\n\r\n        if fname:\r\n            checkpoint_state = torch.load(\r\n                Path(__file__).parent.parent / Path(self.flags.model_dir) / Path(fname),\r\n                map_location=torch.device("cpu")\r\n            )["model_state_dict"]\r\n\r\n            model.load_state_dict(checkpoint_state)\r\n\r\n        return model\r\n\r\n    def save_model(self):\r\n        fpath = Path(self.flags.model_dir) / Path(self.flags.nextgen_model_weight_fname)\r\n        logger.info(f"Saving checkpoint to {fpath}")\r\n        torch.save(obj={"model_state_dict": self.model.state_dict()}, f=fpath)\r\n\r\n    def save_as_best_model(self):\r\n        fpath = Path(self.flags.model_dir) / Path(self.flags.name + self.flags.best_model_weight_fname)\r\n        logger.info(f"Saving checkpoint to {fpath}")\r\n        torch.save(obj={"model_state_dict": self.model.state_dict()}, f=fpath)')
    __stickytape_write_module('c4_player_submission.py', b'import threading\r\nfrom collections import defaultdict\r\nfrom concurrent.futures import ThreadPoolExecutor\r\nimport copy\r\nfrom logging import getLogger\r\nimport numpy as np\r\nfrom threading import Lock\r\nimport torch\r\n\r\nfrom c4_gym.c4_env import C4Env\r\nfrom utility_constants import BOARD_SIZE\r\n\r\nlogger = getLogger(__name__)\r\n_, N_ACTIONS = BOARD_SIZE\r\n\r\n\r\n# these are from AGZ nature paper\r\nclass VisitStats:\r\n    """\r\n    Holds information for use by the AGZ MCTS algorithm on all moves from a given game state (this is generally used inside\r\n    of a defaultdict where a game state in FEN format maps to a VisitStats object).\r\n    Attributes:\r\n        :ivar defaultdict(ActionStats) a: known stats for all actions to take from the the state represented by\r\n            this visitstats.\r\n        :ivar int sum_n: sum of the n value for each of the actions in self.a, representing total\r\n            visits over all actions in self.a.\r\n    """\r\n    def __init__(self):\r\n        self.a = defaultdict(ActionStats)\r\n        self.sum_n = 0\r\n\r\n\r\nclass ActionStats:\r\n    """\r\n    Holds the stats needed for the AGZ MCTS algorithm for a specific action taken from a specific state.\r\n\r\n    Attributes:\r\n        :ivar int n: number of visits to this action by the algorithm\r\n        :ivar float w: every time a child of this action is visited by the algorithm,\r\n            this accumulates the value (calculated from the value network) of that child. This is modified\r\n            by a virtual loss which encourages threads to explore different nodes.\r\n        :ivar float q: mean action value (total value from all visits to actions\r\n            AFTER this action, divided by the total number of visits to this action)\r\n            i.e. it\'s just w / n.\r\n        :ivar float p: prior probability of taking this action, given\r\n            by the policy network.\r\n\r\n    """\r\n    def __init__(self):\r\n        self.n = 0\r\n        self.w = 0\r\n        self.q = 0\r\n        self.p = 0\r\n\r\nclass C4Player:\r\n    def __init__(self, flags, pipes=None):\r\n        self.tree = defaultdict(VisitStats)\r\n        self.flags = flags\r\n        self.moves = []\r\n        self.pipe_pool = pipes\r\n        self.node_lock = defaultdict(Lock)\r\n\r\n    def reset(self):\r\n        """\r\n        reset the tree to begin a new exploration of states\r\n        """\r\n        self.tree = defaultdict(VisitStats)\r\n\r\n    def action(self, env, env_output) -> str:\r\n        """\r\n        Figures out the next best move within the specified environment and returns\r\n        a string describing the action to take.\r\n\r\n        :param C4Env env: environment in which to figure out the action\r\n        :param np.array env_output: observation planes\r\n        :return: None if no action should be taken (indicating a resign). Otherwise, returns a string\r\n            indicating the action to take in uci format\r\n        """\r\n        self.reset()\r\n\r\n        self.search_moves(env, env_output)\r\n        policy = self.calc_policy(env)\r\n\r\n        my_action = int(np.argmax(policy))\r\n\r\n        return my_action\r\n\r\n    def search_moves(self, env, obs) -> (float, float):\r\n        """\r\n        Looks at all the possible moves using the AGZ MCTS algorithm\r\n        and finds the highest value possible move. Does so using multiple threads to get multiple\r\n        estimates from the AGZ MCTS algorithm so we can pick the best.\r\n\r\n        :param C4Env env: env to search for moves within\r\n        :param np.ndarry obs: observation planes derived from the game state\r\n        :return (float,float): the maximum value of all values predicted by each thread,\r\n            and the first value that was predicted.\r\n        """\r\n\r\n        with ThreadPoolExecutor(max_workers=self.flags.search_threads) as executor:\r\n            futures = [\r\n                executor.submit(\r\n                    self.search_my_move,\r\n                    env=copy.deepcopy(env),\r\n                    env_output=obs\r\n                )\r\n                for _ in range(self.flags.simulation_num_per_move)\r\n            ]\r\n\r\n        for f in futures:\r\n            f.result()\r\n\r\n    def search_my_move(self, env: C4Env, env_output) -> float:\r\n        """\r\n        Q, V is value for this Player(always white).\r\n        P is value for the player of next_player (black or white)\r\n\r\n        This method searches for possible moves, adds them to a search tree, and eventually returns the\r\n        best move that was found during the search.\r\n\r\n        :param C4Env env: environment in which to search for the move\r\n        :param np.ndarry env_output: observation planes derived from the game state\r\n        :return float: value of the move. This is calculated by getting a prediction from the value network.\r\n        """\r\n        if env.done:\r\n            if torch.max(env_output[\'reward\']) == 0:\r\n                return 0\r\n            return -1\r\n\r\n        state = env.string_board\r\n\r\n        with self.node_lock[state]:\r\n            action_mask = env_output[\'info\'][\'available_actions_mask\'][0].tolist()\r\n            if state not in self.tree:\r\n                leaf_p, leaf_v = self.evaluate(env_output)\r\n                for a,is_invalid in enumerate(action_mask):\r\n                    if is_invalid is False:\r\n                        self.tree[state].a[a].p = leaf_p[a]\r\n                return leaf_v # From the POV of side to move\r\n\r\n        action_t = self.select_action_q_and_u(env, action_mask)\r\n        if action_mask[action_t]:\r\n            logger.info(f"Action {action_t} is an invalid action:\\n{env.game_state.board}\\n\\n{self.tree[state].a}")\r\n        my_visit_stats = self.tree[state]\r\n        action_stats = my_visit_stats.a[action_t]\r\n\r\n        env_output = env.step(action_t) # noqa\r\n\r\n        leaf_v = -1 * self.search_my_move(env, env_output)  # Next move from enemy POV\r\n\r\n        # BACKUP STEP\r\n        # on returning search path\r\n        # update: N, W, Q\r\n        with self.node_lock[state]:\r\n            my_visit_stats.sum_n += 1\r\n            action_stats.n += 1\r\n            action_stats.w += leaf_v\r\n            action_stats.q = action_stats.w / action_stats.n\r\n        return leaf_v\r\n\r\n    def evaluate(self, obs) -> (np.ndarray, float):\r\n        """ expand new leaf, this is called only once per state\r\n        this is called with state locked\r\n        insert P(a|s), return leaf_v\r\n\r\n        This gets a prediction for the policy and value of the state within the given env\r\n        :return (float, float): the policy and value predictions for this state\r\n        """\r\n        leaf_p, leaf_v = self.predict(obs)\r\n\r\n        return leaf_p, leaf_v\r\n\r\n    def predict(self, state_planes):\r\n        """\r\n        Gets a prediction from the policy and value network\r\n        :param state_planes: the observation state represented as planes\r\n        :return (float,float): policy (prior probability of taking the action leading to this state)\r\n            and value network (value of the state) prediction for this state.\r\n        """\r\n        pipe = self.pipe_pool.pop()\r\n        pipe.send(state_planes)\r\n        ret = pipe.recv()\r\n        self.pipe_pool.append(pipe)\r\n        return ret\r\n\r\n    def select_action_q_and_u(self, env, mask) -> int:\r\n        """\r\n        Picks the next action to explore using the AGZ MCTS algorithm.\r\n\r\n        Picks based on the action which maximizes the maximum action value\r\n        (ActionStats.q) + an upper confidence bound on that action.\r\n\r\n        :param Environment env: env to look for the next moves within\r\n        :param np.ndarray mask: a mask of the action space that is True for each full column\r\n        :return int: the move to explore\r\n        """\r\n        # this method is called with state locked\r\n        state = env.string_board\r\n\r\n        my_visitstats = self.tree[state]\r\n\r\n        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)\r\n\r\n        c_puct = self.flags.c_puct\r\n\r\n        best_s = -np.inf\r\n        best_a = None\r\n\r\n        for action, a_s in my_visitstats.a.items():\r\n            if mask[action]:\r\n                logger.debug(f"Invalid action somehow got into tree {mask} {action}")\r\n                continue\r\n            p_ = a_s.p\r\n            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)\r\n            if b > best_s:\r\n                best_s = b\r\n                best_a = action\r\n        return best_a\r\n\r\n    def calc_policy(self, env):\r\n        """calc \xcf\x80(a|s0)\r\n        :return list(float): a list of probabilities of taking each action, calculated based on visit counts.\r\n        """\r\n        state = env.string_board\r\n        my_visitstats = self.tree[state]\r\n        policy = np.zeros(N_ACTIONS)\r\n        for action, a_s in my_visitstats.a.items():\r\n            policy[action] = a_s.n\r\n        policy /= np.sum(policy)\r\n\r\n        return policy')
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