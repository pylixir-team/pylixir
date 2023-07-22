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


class DiscreteAction(nn.Module):
    def __init__(self, vector_size: int = 128, hidden_dimension: int = 64):
        super(DiscreteAction, self).__init__()
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


class IndependentQNetwork(BasePolicy):
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
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.q_net = DiscreteAction()

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs, self.features_extractor))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class IndependentQPolicy(DQNPolicy):
    def make_q_net(self) -> IndependentQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return IndependentQNetwork(**net_args).to(self.device)