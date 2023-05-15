import torch
import torch.nn as nn

from deep.model.observation import EmbeddingConfiguration, ActorCriticBuilder, EmbeddingName


class BaselineRenderer(ActorCriticBuilder):
    def __init__(self, configuration: EmbeddingConfiguration) -> None:
        super(BaselineRenderer, self).__init__()

        self._configuration = configuration
        self.turn_embedding = torch.nn.Embedding(20, configuration.turn_embedding_size, dtype = torch.float32)
        self.reroll_embedding = torch.nn.Embedding(20, configuration.reroll_embedding_size, dtype = torch.float32)

        self.sage_power_embedding = torch.nn.Embedding(11, configuration.sage_embedding_size, dtype = torch.float32)
        self.sage_position_embedding = torch.nn.Embedding(3, configuration.sage_embedding_size, dtype = torch.float32)
        self.council_embedding = torch.nn.Embedding(300, configuration.council_embedding_size, dtype = torch.float32)

    def forward(self):
        raise NotImplementedError

    def create_actor(self):
        return nn.Sequential(
            nn.Linear(self._get_state_dim(), 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self._configuration.action_dim),
            nn.Softmax(dim=-1)
        )

    def create_critic(self):
        return nn.Sequential(
            nn.Linear(self._get_state_dim(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def _get_state_dim(self) -> int:
        state_dim = 0

        _, suggestion_size = self._configuration.get_order_offset(EmbeddingName.suggestion)
        state_dim += suggestion_size * self._configuration.council_embedding_size

        _, sage_size = self._configuration.get_order_offset(EmbeddingName.committee)
        state_dim += sage_size * self._configuration.sage_embedding_size

        state_dim += self._configuration.turn_embedding_size
        state_dim += self._configuration.reroll_embedding_size

        _, board_size = self._configuration.get_order_offset(EmbeddingName.board)
        state_dim += board_size * 4 # Extend as 4

        _, enchanter_size = self._configuration.get_order_offset(EmbeddingName.enchanter)
        state_dim += enchanter_size * 4 # Extend as 4

        return state_dim

    def render_state(self, observation: torch.Tensor):
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)

        suggestion_offset, suggestion_size = self._configuration.get_order_offset(EmbeddingName.suggestion)
        suggestion_input = observation[:, suggestion_offset: suggestion_offset + suggestion_size]
        suggestion_embedding = self.council_embedding(suggestion_input)
        suggestion_tensor = torch.flatten(suggestion_embedding, 1)

        sage_offset, sage_size = self._configuration.get_order_offset(EmbeddingName.committee)
        sage_indices = observation[:, sage_offset: sage_offset + sage_size]
        sage_matrix = self.sage_power_embedding(sage_indices)
        sage_pos_embedding = self.sage_position_embedding(torch.tensor([0, 1, 2]))
        sage_matrix += sage_pos_embedding
        committe_tensor = torch.flatten(sage_matrix, 1)

        progress_offset, _ = self._configuration.get_order_offset(EmbeddingName.progress)
        turn, reroll = observation[:, progress_offset:progress_offset+1], observation[:, progress_offset + 1: progress_offset+2]
        progress_tensor = torch.flatten(torch.cat([self.turn_embedding(turn), self.reroll_embedding(reroll)], 1), 1)

        board_offset, board_size = self._configuration.get_order_offset(EmbeddingName.board)
        board_indices = observation[:, board_offset: board_offset + board_size]
        board_tensor = self.extend_exact_float(board_indices / 11.0)

        enchanter_offset, enchanter_size = self._configuration.get_order_offset(EmbeddingName.enchanter)
        enchanter_input = observation[:, enchanter_offset: enchanter_offset + enchanter_size]
        enchanter_tensor = self.extend_exact_float(enchanter_input / 1000.0)

        return torch.concat([
            suggestion_tensor,
            committe_tensor,
            progress_tensor,
            board_tensor,
            enchanter_tensor,
        ], 1).float()

    def extend_exact_float(self, float_vector):
        sqrt_count = torch.pow(float_vector, 0.5)
        square_count = torch.pow(float_vector, 2)
        cubic_count = torch.pow(float_vector, 3)

        return torch.cat([sqrt_count, float_vector, square_count, cubic_count], 1).float()


class IntendedBaselineRenderer(ActorCriticBuilder):
    def __init__(self, configuration: EmbeddingConfiguration) -> None:
        super(IntendedBaselineRenderer, self).__init__()

        self._board_expansion_size = 12
        self._board_embedding_size = 24

        self._configuration = configuration
        self.turn_embedding = torch.nn.Embedding(20, configuration.turn_embedding_size, dtype = torch.float32)
        self.reroll_embedding = torch.nn.Embedding(20, configuration.reroll_embedding_size, dtype = torch.float32)

        self.sage_power_embedding = torch.nn.Embedding(11, configuration.sage_embedding_size, dtype = torch.float32)
        self.sage_position_embedding = torch.nn.Embedding(3, configuration.sage_embedding_size, dtype = torch.float32)
        self.council_embedding = torch.nn.Embedding(300, configuration.sage_embedding_size, dtype = torch.float32)

        self.board_nn = nn.Sequential(
            nn.Linear(self._board_expansion_size, self._board_embedding_size),
            nn.ReLU(),
        )

    def forward(self):
        raise NotImplementedError

    def create_actor(self):
        return nn.Sequential(
            nn.Linear(self._get_state_dim(), 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self._configuration.action_dim),
            nn.Softmax(dim=-1)
        )

    def create_critic(self):
        return nn.Sequential(
            nn.Linear(self._get_state_dim(), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _get_state_dim(self) -> int:
        state_dim = 0

        _, sage_size = self._configuration.get_order_offset(EmbeddingName.committee)
        state_dim += sage_size * self._configuration.sage_embedding_size

        state_dim += self._configuration.turn_embedding_size
        state_dim += self._configuration.reroll_embedding_size
        state_dim += self._board_embedding_size * 5

        return state_dim

    def render_state(self, observation: torch.Tensor):
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)

        suggestion_offset, suggestion_size = self._configuration.get_order_offset(EmbeddingName.suggestion)
        suggestion_input = observation[:, suggestion_offset: suggestion_offset + suggestion_size]
        suggestion_embedding = self.council_embedding(suggestion_input)

        sage_offset, sage_size = self._configuration.get_order_offset(EmbeddingName.committee)
        sage_indices = observation[:, sage_offset: sage_offset + sage_size]
        sage_matrix = self.sage_power_embedding(sage_indices)
        sage_pos_embedding = self.sage_position_embedding(torch.tensor([0, 1, 2]))
        sage_matrix += sage_pos_embedding
        sage_matrix += suggestion_embedding

        committe_tensor = torch.flatten(sage_matrix, 1)

        progress_offset, _ = self._configuration.get_order_offset(EmbeddingName.progress)
        turn, reroll = observation[:, progress_offset:progress_offset+1], observation[:, progress_offset + 1: progress_offset+2]
        progress_tensor = torch.flatten(torch.cat([self.turn_embedding(turn), self.reroll_embedding(reroll)], 1), 1)

        board_offset, board_size = self._configuration.get_order_offset(EmbeddingName.board)
        board_indices = observation[:, board_offset: board_offset + board_size]
        board_rbf = self.extend_exact_float(board_indices / 11.0)

        enchanter_offset, enchanter_size = self._configuration.get_order_offset(EmbeddingName.enchanter)

        prob_rbf = self.extend_exact_float(observation[:, enchanter_offset: enchanter_offset + enchanter_size // 2] / 1000.0)
        lucky_rbf = self.extend_exact_float(observation[:, enchanter_offset + enchanter_size // 2: enchanter_offset + enchanter_size] / 1000.0)

        board_tensor = torch.cat([board_rbf, prob_rbf, lucky_rbf], 2)
        board_tensor = self.board_nn(board_tensor)
        board_tensor = torch.flatten(board_tensor, 1)

        return torch.concat([
            committe_tensor,
            progress_tensor,
            board_tensor,
        ], 1).float()

    def extend_exact_float(self, float_vector):
        sqrt_count = torch.pow(float_vector, 0.5)
        square_count = torch.pow(float_vector, 2)
        cubic_count = torch.pow(float_vector, 3)

        return torch.stack([sqrt_count, float_vector, square_count, cubic_count], 2).float()

    def extend_as_gaussian(self, float_vector):
        extended = torch.Tensor([0.025 * n for n in range(self._board_expansion_size)])
        extended = torch.stack([extended, extended, extended, extended, extended]).unsqueeze(0)
        exponent = torch.pow(extended - float_vector.unsqueeze(2), 2)
        extended = torch.exp(exponent * 2)
        return extended
