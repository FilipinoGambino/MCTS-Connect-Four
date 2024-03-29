from typing import Any, Dict, List, Optional, Tuple
import gym
import math
import numpy as np

from .act_spaces import BaseActSpace
from .obs_spaces import BaseObsSpace
from .reward_spaces import GameResultReward
from ..connectx_game.game import Game


class ConnectFour(gym.Env):
    metadata = {'render_modes': ['human']}
    spec = None

    def __init__(
            self,
            act_space: BaseActSpace,
            obs_space: BaseObsSpace,
            configuration: Optional[Dict[str, Any]] = None,
            autoplay: bool = True
    ):
        super(ConnectFour, self).__init__()
        self.action_space = act_space
        self.obs_space = obs_space
        self.default_reward_space = GameResultReward()

        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = dict(rows=6, columns=7, inarow=4)

        self.autoplay = autoplay

        self.game_state = Game()

    def reset(self, **kwargs):
        self.game_state = Game()
        return self.get_obs_reward_done_info()

    def step(self, action):
        if self.autoplay:
            self.game_state.step(action)
        return self.get_obs_reward_done_info()

    def manual_step(self, obs):
        self.game_state.update(obs)

    def info(self, rewards):
        return dict(
            available_actions_mask=self.action_space.get_available_actions_mask(self.board),
            rewards=rewards
        )

    def get_obs_reward_done_info(self):
        rewards, done = self.default_reward_space.compute_rewards(game_state=self.game_state)
        return self.game_state, rewards, done, self.info(rewards)

    def render(self, **kwargs):
        raise NotImplementedError

    @property
    def board(self):
        return self.game_state.board

    @property
    def turn(self):
        return self.game_state.turn