from copy import deepcopy
from typing import Any, Dict, Optional
import gym
import numpy as np
from scipy.signal import convolve2d

from .act_spaces import BaseActSpace
from .obs_spaces import BaseObsSpace
from c4_game.game import Game
from utility_constants import BOARD_SIZE, IN_A_ROW, PLAYER_MARKS, VICTORY_KERNELS

ROWS, COLUMNS = BOARD_SIZE

class C4Env(gym.Env):
    metadata = {'render_modes': ['human']}
    spec = None

    def __init__(
            self,
            flags,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            configuration: Optional[Dict[str, Any]] = None,
            autoplay: bool = True
    ):
        super(C4Env, self).__init__()

        self.flags = flags
        self.act_space = act_space
        self.obs_space = obs_space

        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = dict(rows=ROWS, columns=COLUMNS, inarow=IN_A_ROW)

        self.autoplay = autoplay

        self.game_state = Game(self.configuration)

        self.rewards = [0, 0]
        self.done = False

    def reset(self, **kwargs):
        self.game_state = Game(self.configuration)
        self.rewards = [0, 0]
        self.done = False
        return self.get_obs_reward_done_info()

    def step(self, action):
        if self.autoplay:
            self.game_state.step(action)
        return self.get_obs_reward_done_info()

    def manual_step(self, obs):
        self.game_state.update(obs)

    def check_game_over(self):
        for player_mark in PLAYER_MARKS:
            for kernel_name, kernel in VICTORY_KERNELS.items():
                conv = convolve2d(self.board == player_mark, kernel, mode="valid")
                if np.any(conv == IN_A_ROW):
                    self.rewards = [-1, -1]
                    self.rewards[player_mark - 1] = 1
                    self.done = True
        if self.game_state.turn == self.game_state.max_turns:
            self.done = True

    def info(self):
        return dict(
            available_actions_mask=self.available_actions_mask,
            rewards=self.rewards,
            turn=self.game_state.turn
        )

    def get_obs_reward_done_info(self):
        self.check_game_over()
        return self.game_state, self.rewards, self.done, self.info()

    def render(self, **kwargs):
        raise NotImplementedError

    @property
    def string_board(self):
        return ' '.join(str(x) for x in self.board.flatten())

    @property
    def board(self):
        return self.game_state.board

    @property
    def available_actions_mask(self):
        return np.array(self.game_state.board.all(axis=0), dtype=bool).reshape((1, -1))

    @property
    def winner(self):
        if max(self.rewards) == 0:
            return None
        else:
            return np.argmax(self.rewards) + 1