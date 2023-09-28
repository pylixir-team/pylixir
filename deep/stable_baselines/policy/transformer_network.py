import math
from typing import Any, Dict, Optional, Type

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = th.zeros(max_len, dim_model)
        positions_list = th.arange(0, max_len, dtype=th.float).view(
            -1, 1
        )  # 0, 1, 2, 3, 4, 5
        division_term = th.exp(
            th.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )  # 1000^(2i/dim_model)

        pos_encoding[:, 0::2] = th.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = th.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: th.tensor) -> th.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding)


# L=6, learning rate가 너무 강한듯?


class SimpleTransformerDecisionNet(nn.Module):
    def __init__(
        self,
        vector_size: int = 128,
        hidden_dimension: int = 64,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
    ):

        super().__init__()

        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads

        self.pe = PositionalEncoding(vector_size, 0.0, 10)
        self.mha = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    vector_size,
                    self._transformer_heads,
                    dim_feedforward=vector_size * 2,
                    batch_first=True,
                )
                for _ in range(self._transformer_layers)
            ]
        )

        self.action_network = nn.Sequential(
            nn.Linear(vector_size * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        x = self.pe(x)

        for attn in self.mha:
            x = attn(x)

        action = self.action_network(th.flatten(x, 1))
        return action


class DecisionNet(nn.Module):
    def __init__(
        self,
        vector_size: int = 128,
        hidden_dimension: int = 64,
        **kwargs,
    ):
        super().__init__()

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
        boards = x[:, :5, :]  # [B, 5, N]
        councils = x[:, 5:8, :]  # [B, 3, N]
        ctxs = x[:, 8:, :]  # [B, 2, N]

        ctxs = th.flatten(ctxs, 1)
        ctxs = th.stack([ctxs, ctxs, ctxs], dim=1)
        ctxs = th.stack([ctxs, ctxs, ctxs, ctxs, ctxs], dim=1)

        boards = th.stack([boards, boards, boards], dim=2)
        councils = th.stack([councils, councils, councils, councils, councils], dim=1)

        action_space_vector = th.cat([boards, councils], dim=-1)
        action_space_vector = th.cat([action_space_vector, ctxs], dim=-1)
        output = self.nn(action_space_vector)

        output = th.flatten(th.squeeze(output, dim=1), start_dim=1)

        # Reroll defining network
        reroll = self.reroll(th.flatten(x, start_dim=1))
        action = th.cat([output, reroll], dim=1)

        return action


class TransformerDecisionNet(nn.Module):
    def __init__(
        self,
        vector_size: int = 128,
        hidden_dimension: int = 64,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
        **kwargs,
    ):

        super().__init__()

        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads

        self.pe = PositionalEncoding(vector_size, 0.0, 10)
        self.mha = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    vector_size,
                    self._transformer_heads,
                    dim_feedforward=vector_size * 4,
                    batch_first=True,
                )
                for _ in range(self._transformer_layers)
            ]
        )

        self.decision_net = DecisionNet(
            vector_size=vector_size,
            hidden_dimension=hidden_dimension,
        )

    def forward(self, x):
        x = self.pe(x)

        for attn in self.mha:
            x = attn(x)

        action = self.decision_net(x)

        return action


class TransformerQNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
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


class TransformerQPolicy(BasePolicy):

    q_net: TransformerQNetwork
    q_net_target: TransformerQNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        vector_size: int = 128,
        hidden_dimension: int = 64,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        self.vector_size = vector_size
        self.hidden_dimension = hidden_dimension
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "vector_size": vector_size,
            "hidden_dimension": hidden_dimension,
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "normalize_images": normalize_images,
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> TransformerQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        net_args.pop("features_dim")
        return TransformerQNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(  # pylint: disable=protected-access
            observation, deterministic=deterministic
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                vector_size=self.net_args["vector_size"],
                hidden_dimension=self.net_args["hidden_dimension"],
                transformer_layers=self.net_args["transformer_layers"],
                transformer_heads=self.net_args["transformer_heads"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.training = mode
