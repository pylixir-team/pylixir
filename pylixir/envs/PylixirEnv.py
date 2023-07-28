import pickle
import random
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import open_clip
import openai
import torch
from gymnasium import spaces

from pylixir.data.council.target import UserSelector
from pylixir.envs.observation import EmbeddingProvider
from pylixir.interface.cli import ClientBuilder


class ObsOutofBoundsException(Exception):
    ...


class ObservationSchema:
    def __init__(self) -> None:
        self._discrete_space: list[tuple[str, int]] = []

    def discrete(self, kwd: str, size: int) -> None:
        self._discrete_space.append((kwd, size))

    def discrete_series(self, kwd: str, size: int, count: int) -> None:
        for idx in range(count):
            self.discrete(f"{kwd}_{idx}", size)

    def get_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([size for _, size in self._discrete_space])


def get_observation_schema() -> ObservationSchema:
    schema = ObservationSchema()
    schema.discrete_series("committee", 18, 3)
    schema.discrete("turn_left", 15)
    schema.discrete("reroll", 10)
    schema.discrete_series("board", 15, 5)
    schema.discrete_series("enchant_lucky", 101, 5)
    schema.discrete_series("enchant_prob", 101, 5)

    for idx in range(3):
        prefix = f"suggestion_{idx}"
        schema.discrete(f"{prefix}_applyImmediately", 2)
        schema.discrete(f"{prefix}_applyLimit", 4)
        schema.discrete(f"{prefix}_id", 295)
        schema.discrete(f"{prefix}_logic_0_ratio", 5)
        schema.discrete(f"{prefix}_logic_0_remainTurn", 3)
        schema.discrete(f"{prefix}_logic_0_targetCondition", 8)
        schema.discrete(f"{prefix}_logic_0_targetCount", 5)
        schema.discrete(f"{prefix}_logic_0_targetType", 10)
        schema.discrete(f"{prefix}_logic_0_type", 29)
        schema.discrete(f"{prefix}_logic_1_ratio", 5)
        schema.discrete(f"{prefix}_logic_1_remainTurn", 3)
        schema.discrete(f"{prefix}_logic_1_targetCondition", 8)
        schema.discrete(f"{prefix}_logic_1_targetCount", 5)
        schema.discrete(f"{prefix}_logic_1_targetType", 10)
        schema.discrete(f"{prefix}_logic_1_type", 29)
        schema.discrete(f"{prefix}_pickupRatio", 56)
        schema.discrete(f"{prefix}_range0", 9)
        schema.discrete(f"{prefix}_range1", 9)
        schema.discrete(f"{prefix}_slotType", 5)
        schema.discrete(f"{prefix}_type", 8)

    return schema


def get_embedding_openai(text_to_embed):
    # Embed a line of text
    response = openai.Embedding.create(
        model="text-embedding-ada-002", input=[text_to_embed]
    )
    # Extract the AI output embedding as a list of floats
    embedding = response["data"][0]["embedding"]

    return embedding


class OpenClipEmbeddingProvider:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cuda"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def get_embedding(self, text_to_embed):
        text = self.tokenizer([text_to_embed]).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features[0].detach().cpu().numpy()


class PylixirEnv(gym.Env[Any, Any]):
    observation_space: spaces.MultiDiscrete
    action_space: spaces.Discrete
    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self, render_mode: str = "human", completeness_threshold: int = 16
    ) -> None:
        self.render_mode = render_mode
        self._client_builder = ClientBuilder()

        self._completeness_threshold = completeness_threshold
        self._client = self._client_builder.get_client(0)
        self._embedding_provider = EmbeddingProvider(
            self._client.get_council_pool_index_map()
        )

        # self.observation_space = get_observation_schema().get_space()  # fmt: skip
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(512,), dtype=np.float16
        )
        self.action_space = spaces.Discrete(15 + 1)
        self.model = OpenClipEmbeddingProvider()

    def _get_obs(self) -> np.typing.NDArray[np.float16]:
        txt = self._client.view()
        observation = self.model.get_embedding(txt)
        # embeddings = self.model(**self.tokenizer(txt, return_tensors="pt")).logits
        # observation = np.array(get_embedding(txt), dtype=np.float64)
        # ====================================
        # observation = np.array(
        #     self._embedding_provider.create_observation(self._client)
        # )
        # validation = observation >= self.observation_space.nvec
        # if validation.any():
        #     indices = validation.nonzero()[0]
        #     idx = ", ".join(map(str, indices))
        #     value = ", ".join(map(str, observation[indices]))

        #     with open("client.pkl", "wb") as f:
        #         pickle.dump(self, f)
        #     raise ObsOutofBoundsException(
        #         f"Observation encoding out of bounds: index {idx}, got {value}"
        #     )
        return observation

    def _get_info(self) -> Dict[Any, Any]:
        total_reward = self._embedding_provider.current_total_reward(self._client)
        complete = self._embedding_provider.is_complete(
            self._client, self._completeness_threshold
        )
        return {
            "total_reward": total_reward,
            "complete": complete,
            "current_valuation": self._embedding_provider.current_valuation(
                self._client
            ),
        }

    def render(self) -> None:
        txt = self._client.view()
        print(txt)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.typing.NDArray[np.int64], Dict[Any, Any]]:
        if seed is None:
            seed = random.randint(0, 1 << 16)
        super().reset(seed=seed)
        self._client = self._client_builder.get_client(seed)
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.typing.NDArray[np.int64], float, bool, bool, Dict[Any, Any]]:
        previous_total_reward = self._embedding_provider.current_total_reward(
            self._client
        )
        if action == 15:
            ok = self._client.reroll()
        else:
            action_object = self._embedding_provider.action_index_to_action(action)
            ok = self._client.pick(action_object.sage_index, action_object.effect_index)
        state = self._get_obs()
        reward = (
            self._embedding_provider.current_total_reward(self._client)
            - previous_total_reward
        )
        info = self._get_info()

        if not ok:
            reward = -1
            # observation, reward, terminated, truncated, info
            return state, reward, False, False, info

        done = self._client.is_done()

        # observation, reward, terminated, truncated, info
        return state, reward, done, False, info

    def close(self) -> None:
        return None

    def legal_actions(self) -> list[int]:
        actions = []
        for effect_index in range(5):
            for sage_index in range(3):
                if (
                    sage_index
                    not in self._client.get_state().committee.get_valid_slots()
                ):
                    continue
                if (
                    isinstance(
                        self._client.get_current_councils()[sage_index]
                        .logics[0]
                        .target_selector,
                        UserSelector,
                    )
                    and effect_index > 0
                ):
                    continue
                actions += [effect_index * 3 + sage_index]
        return actions + [15]
