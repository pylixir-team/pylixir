from typing import Any, Dict, List, Optional, Type

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
 
        # Encoding - From formula
        pos_encoding = th.zeros(max_len, dim_model)
        positions_list = th.arange(0, max_len, dtype=th.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = th.exp(th.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
 
        pos_encoding[:, 0::2] = th.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = th.cos(positions_list * division_term)
 
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding
        self.register_buffer("pos_encoding", pos_encoding)
 
    def forward(self, token_embedding: th.tensor) -> th.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding)


class TransformerDecisionNet(nn.Module):
    def __init__(self, 
            vector_size: int = 128, 
            hidden_dimension: int = 64,
            transformer_layers: int = 3,
            transformer_heads: int = 8,
        ):

        super(TransformerDecisionNet, self).__init__()

        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads

        self.pe = PositionalEncoding(vector_size, 0.0, 10)
        self.mha = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                vector_size,
                self._transformer_heads,
                dim_feedforward=vector_size * 2,
                batch_first=True
            ) for _ in range(self._transformer_layers)]
        ) 

        self.nn = nn.Sequential(
            nn.Linear(vector_size * (1 + 1 + 2), hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.reroll = nn.Sequential(
            nn.Linear(vector_size * 10, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, x):
        x = self.pe(x)

        for attn in self.mha:
            x = attn(x)

        boards = x[:, :5, :] # [B, 5, N]
        councils = x[:, 5:8, :] # [B, 3, N]
        ctxs = x[:, 8:, :] # [B, 2, N]

        ctxs = th.flatten(ctxs, 1)
        ctxs = th.stack([ctxs, ctxs, ctxs], dim=1)
        ctxs = th.stack([ctxs, ctxs, ctxs, ctxs, ctxs], dim=1)

        boards = th.stack(
            [boards, boards, boards], dim=2
        )
        councils = th.stack(
            [councils, councils, councils, councils, councils], dim=1
        )

        action_space_vector = th.cat([boards, councils], dim=-1)
        action_space_vector = th.cat([action_space_vector, ctxs], dim=-1)
        output = self.nn(action_space_vector)

        output = th.flatten(th.squeeze(output, dim=1), start_dim=1)

        # Reroll defining network
        reroll = self.reroll(th.flatten(x, start_dim=1))
        action = th.cat([output, reroll], dim=1)

        return action


class TransformerQNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        normalize_images: bool = True,
        vector_size: int = 128, 
        hidden_dimension: int = 64,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )


        self.vector_size = vector_size
        self.hidden_dimension = hidden_dimension
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads

        self.q_net = TransformerDecisionNet(
            vector_size=vector_size,
            hidden_dimension=hidden_dimension,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                vector_size=self.vector_size,
                hidden_dimension=self.hidden_dimension,
                transformer_layers=self.transformer_layers,
                transformer_heads=self.transformer_heads,
                features_extractor=self.features_extractor,
            )
        )
        return data


class TransformerQPolicy(DQNPolicy):
    def make_q_net(self) -> TransformerQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return TransformerQNetwork(**net_args).to(self.device)