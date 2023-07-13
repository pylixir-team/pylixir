import gymnasium as gym
import torch as th
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FloatExpansion(nn.Module):
    def __init__(self):
        super(FloatExpansion, self).__init__()

    def forward(self, x):
        return th.cat([
            x,
            th.pow(x, 2),
            th.pow(x, 3),
            th.pow(x, 0.5),
        ], -1)

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0

        FLOATING_EMBEDDING_SIZE = 64
        SUGGESION_EMBEDDING_SIZE = 64
        EMBEDDING_SIZE = 64

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key in extractors:
                continue
            elif isinstance(subspace, spaces.Box):
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    FloatExpansion(),
                    nn.Linear(4 * subspace.shape[0], FLOATING_EMBEDDING_SIZE)
                )
                total_concat_size += FLOATING_EMBEDDING_SIZE
            elif isinstance(subspace, spaces.Discrete):
                if "suggestion" in key:
                    main_key = key.replace("suggestion_2", "suggestion_0").replace("suggestion_1", "suggestion_0")
                    if main_key not in extractors:
                        extractors[main_key] = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(subspace.n, SUGGESION_EMBEDDING_SIZE)
                        ) 

                    extractors[key] = extractors[main_key]
                    total_concat_size += SUGGESION_EMBEDDING_SIZE
                else:
                    extractors[key] = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(subspace.n, EMBEDDING_SIZE)
                    ) 

                    total_concat_size += EMBEDDING_SIZE
            else:
                raise ValueError

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        flag = False
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
