import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from deep.stable_baselines.policy.transformer_network import PositionalEncoding


class FloatExpansion(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.lin_layer = nn.Linear(4, size)

    def forward(self, x):
        v = th.stack(
            [
                x,
                th.pow(x, 2),
                th.pow(x, 3),
                th.pow(x, 0.5),
            ],
            -1,
        )
        v = self.lin_layer(v)
        # v = th.flatten(v, 1)
        return v


def get_major_key(full_key: str) -> str:
    main_key = full_key.replace("suggestion_2", "suggestion_0").replace(
        "suggestion_1", "suggestion_0"
    )
    for idx in range(1, 5):
        main_key = main_key.replace(f"board_{idx}", "board_0")
    for idx in range(1, 5):
        main_key = main_key.replace(f"committee_{idx}", "committee_0")

    return main_key


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        prob_hidden_dim: int = 16,
        suggesion_feature_hidden_dim: int = 16,
        embedding_dim: int = 128,
        flatten_output: bool = True,
    ):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        self._prob_hidden_dim = prob_hidden_dim
        self._suggesion_feature_hidden_dim = suggesion_feature_hidden_dim
        self._embedding_dim = embedding_dim
        self._flatten_output = flatten_output

        board_embedding_size = 0
        council_embedding_size = 0

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key in extractors:
                continue
            if isinstance(subspace, spaces.Box):
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    FloatExpansion(prob_hidden_dim),
                )
                board_embedding_size += prob_hidden_dim
            elif isinstance(subspace, spaces.Discrete):
                if key in ("turn_left", "reroll"):
                    extractors[key] = nn.Sequential(
                        nn.Flatten(), nn.Linear(subspace.n, embedding_dim)
                    )
                    continue

                main_key = get_major_key(key)
                if main_key not in extractors:
                    extractors[main_key] = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(subspace.n, suggesion_feature_hidden_dim),
                    )
                    if "suggestion" in main_key or "committee" in main_key:
                        council_embedding_size += suggesion_feature_hidden_dim

                extractors[key] = extractors[main_key]
            else:
                raise ValueError

        board_embedding_size += suggesion_feature_hidden_dim

        self.extractors = nn.ModuleDict(extractors)
        self.council_collector = nn.Linear(council_embedding_size, embedding_dim)
        self.board_collector = nn.Linear(board_embedding_size, embedding_dim)

        # Update the features dim manually
        self._features_dim = embedding_dim * (3 + 5 + 2)

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_map = {}
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_map[key] = extractor(observations[key])

        # Board embedding setup
        for idx in range(5):
            board_tensor_list = []
            board_tensor_list.append(encoded_tensor_map[f"board_{idx}"])
            board_tensor_list.append(encoded_tensor_map["enchant_lucky"][:, idx])
            board_tensor_list.append(encoded_tensor_map["enchant_prob"][:, idx])

            encoded_tensor_list.append(
                self.board_collector(th.cat(board_tensor_list, dim=1))
            )

        # Council embedding setup
        for idx in range(3):
            council_tensor_list = []
            for k in sorted(encoded_tensor_map.keys()):
                if f"suggestion_{idx}" in k:
                    council_tensor_list.append(encoded_tensor_map[k])

            council_tensor_list.append(encoded_tensor_map[f"committee_{idx}"])
            encoded_tensor_list.append(
                self.council_collector(th.cat(council_tensor_list, dim=1))
            )

        encoded_tensor_list.append(encoded_tensor_map["turn_left"])
        encoded_tensor_list.append(encoded_tensor_map["reroll"])

        v = th.stack(encoded_tensor_list, dim=1)

        if self._flatten_output:
            return th.flatten(v, start_dim=1)

        return v


class CombinedTransformerExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        prob_hidden_dim: int = 16,
        suggesion_feature_hidden_dim: int = 16,
        embedding_dim: int = 128,
        flatten_output: bool = True,
        vector_size: int = 128,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
    ):
        super().__init__(observation_space, features_dim=1)

        self.custom_combined_extractor = CustomCombinedExtractor(
            observation_space=observation_space,
            prob_hidden_dim=prob_hidden_dim,
            suggesion_feature_hidden_dim=suggesion_feature_hidden_dim,
            embedding_dim=embedding_dim,
            flatten_output=False,
        )

        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads
        self._flatten_output = flatten_output

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

    def forward(self, observations) -> th.Tensor:
        combined_feture = self.custom_combined_extractor(observations)

        x = self.pe(combined_feture)

        for attn in self.mha:
            x = attn(x)

        if self._flatten_output:
            return th.flatten(x, start_dim=1)

        return x
