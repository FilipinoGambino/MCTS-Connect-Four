import torch
from torch import nn

from .models import BasicActorCriticNetwork
from .conv_blocks import ResidualBlock
from .in_blocks import ConvEmbeddingInputLayer
from .attn_blocks import MHABlock, ViTBlock
from c4_gym import obs_spaces
from utility_constants import BOARD_SIZE

def create_model(
        flags,
        device: torch.device,
) -> nn.Module:
    obs_space = flags.obs_space(**flags.obs_space_kwargs)

    return _create_model(
        flags,
        device,
        obs_space,
    )

def _create_model(
        flags,
        device: torch.device,
        obs_space: obs_spaces.BaseObsSpace,
):
    act_space = flags.act_space()
    conv_embedding_input_layer = ConvEmbeddingInputLayer(
        obs_space=obs_space.get_obs_spec(),
        embedding_dim=flags.embedding_dim,
        out_dim=flags.hidden_dim,
    )
    if flags.model_arch == "mha_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            MHABlock(
                dim=flags.hidden_dim,
                heads=flags.n_heads
            )
        )
    elif flags.model_arch == "vit_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ViTBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=BOARD_SIZE[0],
                width=BOARD_SIZE[1],
                mha_layer=MHABlock(
                    dim=flags.hidden_dim,
                    heads=flags.n_heads
                ),
                normalize=flags.normalize,
            ) for _ in range(flags.n_blocks)]
        )
    elif flags.model_arch == "conv_model":
        base_model = nn.Sequential(
            conv_embedding_input_layer,
            *[ResidualBlock(
                in_channels=flags.hidden_dim,
                out_channels=flags.hidden_dim,
                height=BOARD_SIZE[0],
                width=BOARD_SIZE[1],
                kernel_size=flags.kernel_size,
                normalize=flags.normalize,
                activation=nn.LeakyReLU,
                rescale_se_input=flags.rescale_se_input,
            ) for _ in range(flags.n_blocks)]
        )
    else:
        raise NotImplementedError(f"Model_arch: {flags.model_arch}")

    model = BasicActorCriticNetwork(
        base_model=base_model,
        base_out_channels=flags.hidden_dim,
        action_space=act_space.get_action_space(),
        reward_space=flags.reward_space.get_reward_spec(),
        n_value_heads=1,
        rescale_value_input=flags.rescale_value_input
    )

    return model.to(device=device)