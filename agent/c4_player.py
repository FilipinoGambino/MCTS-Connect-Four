import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
from logging import getLogger
import numpy as np
from threading import Lock
import torch

from c4_gym.c4_env import C4Env
from utility_constants import BOARD_SIZE

logger = getLogger(__name__)
_, N_ACTIONS = BOARD_SIZE


# these are from AGZ nature paper
class VisitStats:
    """
    Holds information for use by the AGZ MCTS algorithm on all moves from a given game state (this is generally used inside
    of a defaultdict where a game state in FEN format maps to a VisitStats object).
    Attributes:
        :ivar defaultdict(ActionStats) a: known stats for all actions to take from the the state represented by
            this visitstats.
        :ivar int sum_n: sum of the n value for each of the actions in self.a, representing total
            visits over all actions in self.a.
    """
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0


class ActionStats:
    """
    Holds the stats needed for the AGZ MCTS algorithm for a specific action taken from a specific state.

    Attributes:
        :ivar int n: number of visits to this action by the algorithm
        :ivar float w: every time a child of this action is visited by the algorithm,
            this accumulates the value (calculated from the value network) of that child. This is modified
            by a virtual loss which encourages threads to explore different nodes.
        :ivar float q: mean action value (total value from all visits to actions
            AFTER this action, divided by the total number of visits to this action)
            i.e. it's just w / n.
        :ivar float p: prior probability of taking this action, given
            by the policy network.

    """
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0

class C4Player:
    def __init__(self, flags, pipes=None):
        self.tree = defaultdict(VisitStats)
        self.flags = flags
        self.moves = []
        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)
        self.temperature_tau = self.flags.temperature_tau

    def reset(self):
        """
        reset the tree to begin a new exploration of states
        """
        self.tree = defaultdict(VisitStats)

    def action(self, env, env_output) -> str:
        """
        Figures out the next best move within the specified environment and returns
        a string describing the action to take.

        :param C4Env env: environment in which to figure out the action
        :param np.array env_output: observation planes
        :return: returns an integer indicating the action
        """
        self.reset()

        _,_ = self.search_moves(env, env_output)
        policy = self.calc_policy(env, env_output)

        my_action = int(np.random.choice(range(N_ACTIONS), p = policy))

        self.moves.append([
            {
                "obs":env_output["obs"],
                 "info":{"available_actions_mask":env_output["info"]["available_actions_mask"]}
            },
            policy
        ])
        return my_action

    def search_moves(self, env, obs) -> (float, float):
        """
        Looks at all the possible moves using the AGZ MCTS algorithm
         and finds the highest value possible move. Does so using multiple threads to get multiple
         estimates from the AGZ MCTS algorithm so we can pick the best.

        :param C4Env env: env to search for moves within
        :param np.ndarry obs: observation planes derived from the game state
        :return (float,float): the maximum value of all values predicted by each thread,
            and the first value that was predicted.
        """

        with ThreadPoolExecutor(max_workers=self.flags.search_threads) as executor:
            futures = [
                executor.submit(
                    self.search_my_move,
                    env=copy.deepcopy(env),
                    env_output=obs,
                    is_root_node=True
                )
                for _ in range(self.flags.simulation_num_per_move)
            ]

        vals = [f.result() for f in futures]

        return np.max(vals), vals[0]

    def search_my_move(self, env: C4Env, env_output, is_root_node=False) -> float:
        """
        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)

        This method searches for possible moves, adds them to a search tree, and eventually returns the
        best move that was found during the search.

        :param C4Env env: environment in which to search for the move
        :param np.ndarry env_output: observation planes derived from the game state
        :param boolean is_root_node: whether this is the root node of the search.
        :return float: value of the move. This is calculated by getting a prediction from the value network.
        """
        if env.done:
            if torch.max(env_output['reward']) == 0:
                return 0
            return -1

        state = env.string_board

        with self.node_lock[state]:
            action_mask = env_output['info']['available_actions_mask'][0].tolist()
            if state not in self.tree:
                leaf_p, leaf_v = self.evaluate(env_output)
                for a,column_full in enumerate(action_mask):
                    if not column_full:
                        self.tree[state].a[a].p = leaf_p[a]
                return leaf_v # From the POV of side to move

        action_t = self.select_action_q_and_u(env, is_root_node)

        my_visit_stats = self.tree[state]
        action_stats = my_visit_stats.a[action_t]

        env_output = env.step(action_t) # noqa

        leaf_v = -1 * self.search_my_move(env, env_output)  # Next move from enemy POV

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        with self.node_lock[state]:
            my_visit_stats.sum_n += 1
            action_stats.n += 1
            action_stats.w += leaf_v
            action_stats.q = action_stats.w / action_stats.n
        return leaf_v

    def evaluate(self, obs) -> (np.ndarray, float):
        """ expand new leaf, this is called only once per state
        this is called with state locked
        insert P(a|s), return leaf_v

        This gets a prediction for the policy and value of the state within the given env
        :return (float, float): the policy and value predictions for this state
        """
        leaf_p, leaf_v = self.predict(obs)

        return leaf_p, leaf_v

    def predict(self, state_planes):
        """
        Gets a prediction from the policy and value network
        :param state_planes: the observation state represented as planes
        :return (float,float): policy (prior probability of taking the action leading to this state)
            and value network (value of the state) prediction for this state.
        """
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    def select_action_q_and_u(self, env, is_root_node) -> int:
        """
        Picks the next action to explore using the AGZ MCTS algorithm.

        Picks based on the action which maximizes the maximum action value
        (ActionStats.q) + an upper confidence bound on that action.

        :param Environment env: env to look for the next moves within
        :param is_root_node: whether this is for the root node of the MCTS search.
        :return int: the move to explore
        """
        # this method is called with state locked
        state = env.string_board

        my_visitstats = self.tree[state]

        xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

        e = self.flags.noise_eps
        c_puct = self.flags.c_puct
        dir_alpha = self.flags.dirichlet_alpha

        best_s = -np.inf
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))

        for idx,(action, a_s) in enumerate(my_visitstats.a.items()):
            p_ = a_s.p
            if is_root_node:
                p_ = (1 - e) * p_ + e * noise[idx]
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    def calc_policy(self, env, env_output):
        """calc π(a|s0)
        :return list(float): a list of probabilities of taking each action, calculated based on visit counts.
        """
        state = env.string_board
        my_visitstats = self.tree[state]
        policy = np.zeros(N_ACTIONS)
        for action, a_s in my_visitstats.a.items():
            policy[action] = a_s.n
        mask = env_output['info']['available_actions_mask'].cpu()
        policy = np.where(mask, 0, policy).squeeze(axis=0)
        policy /= np.sum(policy)

        return self.apply_temperature(policy, env.game_state.turn)

    def apply_temperature(self, policy, turn):
        """
        Applies a random fluctuation to probability of choosing various actions
        :param policy: list of probabilities of taking each action
        :param turn: number of turns that have occurred in the game so far
        :return: policy, randomly perturbed based on the temperature. High temp = more perturbation. Low temp = less.
        """
        tau = np.power(self.temperature_tau, turn + 1)
        if tau < 0.1:
            tau = 0.
        if tau == 0.:
            action = np.argmax(policy)
            ret = np.zeros(N_ACTIONS)
            ret[action] = 1.
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret

    def finish_game(self, z):
        """
        When game is done, updates the value of all past moves based on the result.

        :param self:
        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]