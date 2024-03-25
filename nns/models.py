import gym
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union

from .in_blocks import DictInputLayer
from ..connectx_gym.reward_spaces import RewardSpec
from ..utility_constants import BOARD_SIZE
import logging


class DictActor(nn.Module):
    def __init__(
            self,
            in_channels: int,
            action_space: gym.spaces.Discrete,
    ):
        super(DictActor, self).__init__()
        self.n_actions = action_space.n

        self.actor = nn.Conv2d(
            in_channels,
            self.n_actions,
            kernel_size=BOARD_SIZE
        )


    def forward(
            self,
            x: torch.Tensor,
            available_actions_mask: torch.Tensor,
            sample: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expects an input of shape batch_size, n_channels, h, w
        This input will be projected by the actor, and then converted to shape batch_size, n_channels, 1, h, w
        """
        b, _, h, w = x.shape

        logits = self.actor(x)
        logits = logits.view(-1,self.n_actions)

        aam = available_actions_mask

        assert logits.shape == aam.shape, f"logits: {logits.shape} | mask: {aam.shape}"
        logits = logits + torch.where(
            aam,
            torch.full_like(logits, fill_value=float("-inf")),
            torch.zeros_like(logits)
        )
        if logits.isnan().any():
            logging.warning(f"-------------------FIN-------------------")
        actions = DictActor.logits_to_actions(logits, sample)

        return logits, actions

    @staticmethod
    @torch.no_grad()
    def logits_to_actions(logits: torch.Tensor, sample: bool) -> torch.Tensor:
        if sample:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1, replacement=False)
        else:
            return torch.argmax(logits, dim=-1).unsqueeze(dim=-1)

class MultiLinear(nn.Module):
    # TODO: Add support for subtask float weightings instead of integer indices
    def __init__(self, num_layers: int, in_features: int, out_features: int, bias: bool = True):
        super(MultiLinear, self).__init__()
        self.weights = nn.Parameter(torch.empty((num_layers, in_features, out_features)))
        if bias:
            self.biases = nn.Parameter(torch.empty((num_layers, out_features)))
        else:
            self.register_parameter("biases", None)
        self.reset_parameters()

    def reset_parameters(self) -> NoReturn:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.biases is not None:
            # noinspection PyProtectedMember
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, x: torch.Tensor, embedding_idxs: torch.Tensor) -> torch.Tensor:
        weights = self.weights[embedding_idxs]
        if self.biases is None:
            biases = 0.
        else:
            biases = self.biases[embedding_idxs]
        return torch.matmul(x.unsqueeze(1), weights).squeeze(1) + biases


class BaselineLayer(nn.Module):
    def __init__(self, in_channels: int, reward_space: RewardSpec, n_value_heads: int, rescale_input: bool):
        super(BaselineLayer, self).__init__()
        assert n_value_heads >= 1
        self.reward_min = reward_space.reward_min
        self.reward_max = reward_space.reward_max
        self.multi_headed = n_value_heads > 1
        self.rescale_input = rescale_input
        if self.multi_headed:
            self.linear = MultiLinear(n_value_heads, in_channels, 1)
        else:
            self.linear = nn.Linear(in_channels, 1)
        if reward_space.zero_sum:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Sigmoid()
        if not reward_space.only_once:
            # Expand reward space to n_steps for rewards that occur more than once
            reward_space_expanded = np.prod(BOARD_SIZE)
            self.reward_min *= reward_space_expanded
            self.reward_max *= reward_space_expanded

    def forward(self, x: torch.Tensor,
                input_mask: Optional[torch.Tensor]=None,
                value_head_idxs: Optional[torch.Tensor]=None
                ) -> torch.Tensor:
        """
        Expects an input of shape b , n_channels, x, y
        Returns an output of shape b, 1
        """
        # Average feature planes
        if self.rescale_input:
            x = torch.flatten(x, start_dim=-2, end_dim=-1).sum(dim=-1)
            x = x / torch.flatten(input_mask, start_dim=-2, end_dim=-1).sum(dim=-1)
        else:
            x = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        # Project and reshape input
        if self.multi_headed:
            x = self.linear(x, value_head_idxs.squeeze())
        else:
            x = self.linear(x)
        # Rescale to [0, 1], and then to the desired reward space
        x = self.activation(x)

        return x * (self.reward_max - self.reward_min) + self.reward_min

class BasicActorCriticNetwork(nn.Module):
    def __init__(
            self,
            base_model: nn.Module,
            base_out_channels: int,
            action_space: gym.spaces.Dict,
            reward_space: RewardSpec,
            actor_critic_activation: Callable = nn.ReLU,
            n_action_value_layers: int = 2,
            n_value_heads: int = 1,
            rescale_value_input: bool = True
    ):
        super(BasicActorCriticNetwork, self).__init__()
        self.dict_input_layer = DictInputLayer()
        self.base_model = base_model

        if n_action_value_layers < 2:
            raise ValueError("n_action_value_layers must be >= 2 in order to use spectral_norm")

        self.actor_base = self.make_spectral_norm_head_base(
            n_layers=n_action_value_layers,
            n_channels=base_out_channels,
            activation=actor_critic_activation
        )
        self.actor = DictActor(base_out_channels, action_space)

        self.baseline_base = self.make_spectral_norm_head_base(
            n_layers=n_action_value_layers,
            n_channels=base_out_channels,
            activation=actor_critic_activation
        )
        self.baseline = BaselineLayer(
            in_channels=base_out_channels,
            reward_space=reward_space,
            n_value_heads=n_value_heads,
            rescale_input=rescale_value_input
        )

    def forward(
            self,
            x: Dict[str, Union[dict, torch.Tensor]],
            sample: bool = True,
            **actor_kwargs
    ) -> Dict[str, Any]:
        x, available_actions_mask, subtask_embeddings = self.dict_input_layer(x)
        base_out = self.base_model(x)

        if subtask_embeddings is not None:
            subtask_embeddings = torch.repeat_interleave(subtask_embeddings, 2, dim=0)

        policy_logits, actions = self.actor(
            self.actor_base(base_out),
            available_actions_mask=available_actions_mask,
            sample=sample,
            **actor_kwargs
        )

        baseline = self.baseline(self.baseline_base(base_out))

        return dict(
            actions=actions,
            policy_logits=policy_logits,
            baseline=baseline
        )

    def sample_actions(self, *args, **kwargs):
        return self.forward(*args, sample=True, **kwargs)

    def select_best_actions(self, *args, **kwargs):
        return self.forward(*args, sample=False, **kwargs)

    @staticmethod
    def make_spectral_norm_head_base(n_layers: int, n_channels: int, activation: Callable) -> nn.Module:
        """
        https://arxiv.org/abs/1802.05957
        Returns the base of an action or value head, with the final layer of the base/the semifinal layer of the
        head spectral normalized.
        NB: this function actually returns a base with n_layer - 1 layers, leaving the final layer to be filled in
        with the proper action or value output layer.
        """
        assert n_layers >= 2
        layers = []
        for i in range(n_layers - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, (1, 1)))
            layers.append(activation())
        layers.append(
            nn.utils.spectral_norm(nn.Conv2d(n_channels, n_channels, (1, 1)))
        )
        layers.append(activation())

        return nn.Sequential(*layers)