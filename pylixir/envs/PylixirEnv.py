import random
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pylixir.application.game import Client
from pylixir.data.council.target import UserSelector
from pylixir.envs.observation import EmbeddingProvider
from pylixir.interface.cli import ClientBuilder


class PylixirEnv(gym.Env[Any, Any]):
    def __init__(self, completeness_threshold: int = 16) -> None:

        self._client_builder = ClientBuilder()

        self._embedding_provider: EmbeddingProvider
        self._completeness_threshold = completeness_threshold
        self._client: Client

        # fmt: off
        self.observation_space = spaces.MultiDiscrete([
                                    294, 294, 294, # suggestion_vector
                                    18, 18, 18, # committe_vector
                                    15, 3, # progress_vector(turn_left, reroll)
                                    11, 11, 11, 11, 11, # board_vector
                                    *[100] * 10])
        # fmt: on
        self.action_space = spaces.Discrete(15)

    def seed(self, seed: int) -> None:
        self._client = self._client_builder.get_client(seed)
        self._embedding_provider = EmbeddingProvider(
            self._client.get_council_pool_index_map()
        )

    def _get_obs(self) -> np.typing.NDArray[np.int64]:
        return np.array(self._embedding_provider.create_observation(self._client))

    def _get_info(self) -> Dict[Any, Any]:
        return {}

    def render(self) -> None:
        txt = self._client.view()
        print(txt)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[np.typing.NDArray[np.int64], Dict[Any, Any]]:
        if seed is None:
            seed = random.randint(0, 1 << 16)
        super().reset(seed=seed)
        self.seed(seed)
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.typing.NDArray[np.int64], float, bool, bool, Dict[Any, Any]]:
        action_object = self._embedding_provider.action_index_to_action(action)
        previous_total_reward = self._embedding_provider.current_total_reward(
            self._client
        )

        ok = self._client.pick(action_object.sage_index, action_object.effect_index)
        state = self._get_obs()
        reward = (
            self._embedding_provider.current_total_reward(self._client)
            - previous_total_reward
        )
        info = self._get_info()

        if not ok:
            reward = -10
            done = True
            complete = False
            # observation, reward, terminated, truncated, info
            return state, reward, done, False, info

        done = self._client.is_done()
        complete = self._embedding_provider.is_complete(
            self._client, self._completeness_threshold
        )

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
        return actions
