import gymnasium as gym
import torch as th
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FloatExpansion(nn.Module):
    def __init__(self, size: int):
        super(FloatExpansion, self).__init__()
        self.lin_layer = nn.Linear(4, size)

    def forward(self, x):
        v = th.stack([
            x,
            th.pow(x, 2),
            th.pow(x, 3),
            th.pow(x, 0.5),
        ], -1)
        v = self.lin_layer(v)
        #v = th.flatten(v, 1)
        return v


def get_major_key(full_key: str) -> str:
    main_key = full_key.replace(
        "suggestion_2", "suggestion_0"
    ).replace(
        "suggestion_1", "suggestion_0"
    )
    for idx in range(1, 5):
        main_key = main_key.replace(f"board_{idx}", "board_0")
    for idx in range(1, 5):
        main_key = main_key.replace(f"committee_{idx}", "committee_0")

    return main_key

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0

        FLOATING_EMBEDDING_SIZE = 16
        SUGGESION_EMBEDDING_SIZE = 16
        
        COLLECTION_EMBEDDING_SIZE = 256

        board_embedding_size = 0
        council_embedding_size = 0

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key in extractors:
                continue
            elif isinstance(subspace, spaces.Box):
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    FloatExpansion(FLOATING_EMBEDDING_SIZE),
                )
                total_concat_size += FLOATING_EMBEDDING_SIZE * subspace.shape[0]
                board_embedding_size += FLOATING_EMBEDDING_SIZE
            elif isinstance(subspace, spaces.Discrete):
                main_key = get_major_key(key)
                if main_key not in extractors:
                    extractors[main_key] = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(subspace.n, SUGGESION_EMBEDDING_SIZE)
                    )
                    if "suggestion" in main_key or "committee" in main_key:
                        council_embedding_size += SUGGESION_EMBEDDING_SIZE

                extractors[key] = extractors[main_key]
                total_concat_size += SUGGESION_EMBEDDING_SIZE
            else:
                raise ValueError

        board_embedding_size += SUGGESION_EMBEDDING_SIZE

        self.extractors = nn.ModuleDict(extractors)
        self.council_collector = nn.Linear(council_embedding_size, COLLECTION_EMBEDDING_SIZE)
        self.board_collector = nn.Linear(board_embedding_size, COLLECTION_EMBEDDING_SIZE)

        # Update the features dim manually
        # self._features_dim = total_concat_size
        self._features_dim = COLLECTION_EMBEDDING_SIZE * (3 + 5) + SUGGESION_EMBEDDING_SIZE * 2

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

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
