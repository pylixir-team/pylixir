import pickle
import random
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pylixir.application.game import Client
from pylixir.data.council.target import UserSelector
from pylixir.envs.observation import EmbeddingProvider
from pylixir.interface.cli import ClientBuilder


class ObsOutofBoundsException(Exception):
    ...


class PylixirEnv(gym.Env[Any, Any]):
    observation_space: spaces.MultiDiscrete
    action_space: spaces.Discrete
    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self, render_mode: str = "human", completeness_threshold: int = 16
    ) -> None:
        self.render_mode = render_mode
        self._client_builder = ClientBuilder()

        self._embedding_provider: EmbeddingProvider
        self._completeness_threshold = completeness_threshold
        self._client: Client

        # fmt: off
        self.observation_space = spaces.MultiDiscrete([
                                    18, 18, 18, # committe_vector
                                    15, 10, # progress_vector(turn_left, reroll)
                                    15, 15, 15, 15, 15, # board_vector
                                    *[101] * 10,
                                    *[2, 4, 295, 5, 3, 8, 5, 10, 29, 5, 3, 8, 5, 10, 29, 56, 9, 9, 5, 8] * 3]) # suggestion_vector
        # fmt: on
        self.action_space = spaces.Discrete(15)

    def _get_obs(self) -> np.typing.NDArray[np.int64]:
        observation = np.array(
            self._embedding_provider.create_observation(self._client)
        )
        validation = observation >= self.observation_space.nvec
        if validation.any():
            indices = validation.nonzero()[0]
            idx = ", ".join(map(str, indices))
            value = ", ".join(map(str, observation[indices]))

            with open("client.pkl", "wb") as f:
                pickle.dump(self, f)
            raise ObsOutofBoundsException(
                f"Observation encoding out of bounds: index {idx}, got {value}"
            )
        return observation

    def _get_info(self) -> Dict[Any, Any]:
        total_reward = self._embedding_provider.current_total_reward(self._client)
        complete = self._embedding_provider.is_complete(
            self._client, self._completeness_threshold
        )
        return {"total_reward": total_reward, "complete": complete}

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
        # EmbeddingProvider can be in __init__, but since in this structure EmbeddingProvider need self._client, it is in here.
        self._embedding_provider = EmbeddingProvider(
            self._client.get_council_pool_index_map()
        )
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.typing.NDArray[np.int64], float, bool, bool, Dict[Any, Any]]:
        previous_total_reward = self._embedding_provider.current_total_reward(
            self._client
        )
        if action >= 15:
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
            reward = -10
            done = True
            # observation, reward, terminated, truncated, info
            return state, reward, done, False, info

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
        return actions
