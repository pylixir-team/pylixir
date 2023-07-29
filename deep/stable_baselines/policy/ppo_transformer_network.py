from typing import Any, Dict, List, Optional, Type

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from deep.stable_baselines.policy.transformer_network import TransformerDecisionNet


class PPOTransformerPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        use_sde: bool = False,
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
        self.net_args = {
            "observation_space": observation_space,
            "action_space": action_space,
            "vector_size": vector_size,
            "hidden_dimension": hidden_dimension,
            "transformer_layers": transformer_layers,
            "transformer_heads": transformer_heads,
            "normalize_images": normalize_images,
        }

        self.vector_size = vector_size
        self.hidden_dimension = hidden_dimension
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
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

    def _make_action_net(self) -> TransformerDecisionNet:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args.pop("features_dim")
        return TransformerDecisionNet(**net_args).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=[],
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.action_net = self._make_action_net()
        self.value_net = nn.Sequential(
            nn.Linear(
                self.mlp_extractor.latent_dim_vf, 64
            ),
            nn.Flatten(start_dim=1),
            nn.Linear(
                64, 1
            )
        )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
