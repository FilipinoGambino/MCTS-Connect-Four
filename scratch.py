from collections import defaultdict
import pickle

# with open("C:\\Users\\nick.gorichs\\PycharmProjects\\MCTS-Connect-Four\\play_data\\play_20240411-101249.509107.pkl", "rb") as file:
#     obs, probs, win = pickle.load(file)[0]
#
# for k,v in obs.items():
#     print(f"{k}")
# print(probs)
# print(win)

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

f = defaultdict(VisitStats)
f['a'].p = 1
f['b'].p = 2
for