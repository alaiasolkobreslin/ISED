from typing import List, Optional

from experiments.scene.anesi_scene import InferenceModelBase
from experiments.scene import _SceneState
import torch
import torch.nn as nn

from inference_models import InferenceResult

class InferenceModelScene(InferenceModelBase[_SceneState]):

    def __init__(self, N: int, output_dims: List[int], layers = 1, hidden_size: int = 200, prune: bool = True):
        super().__init__(prune)
        self.N = N
        self.layers = layers
        self.output_dims = output_dims

        input_queries = [nn.Linear(10 * 47 * N + sum(output_dims[:i]), hidden_size) for i in range(len(output_dims))]
        output_queries = [nn.Linear(hidden_size, dim) for dim in output_dims]

        y_size = sum(output_dims)

        self.input_layer = nn.ModuleList(input_queries +
                                     [nn.Linear(10 * 47 * N + y_size + i * 47, hidden_size) for i in range(10 * N)])
        self.hiddens = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range((10 * N + len(output_dims)) * (layers - 1))])
        self.outputs = nn.ModuleList(output_queries +
                                     [nn.Linear(hidden_size, 9) for _ in range(10 * N)])
    
    def distribution(self, state:_SceneState) -> torch.Tensor:
        p = state.probability_vector()
        layer_index = len(state.oh_state)
        inputs = torch.cat([p] + state.oh_state, -1)

        z = torch.relu(self.input_layer[layer_index](inputs))

        for i in range(self.layers - 1):
            z = torch.relu(self.hiddens[i * (10 * self.N + len(self.output_dims)) + layer_index](z))

        logits = self.outputs[layer_index](z)
        if logits.shape[-1] > 1:
            dist = torch.softmax(logits, -1)
            return dist
        dist = torch.sigmoid(logits)
        return dist

class IndependentIMScene(InferenceModelBase[_SceneState]):

    def __init__(self, N: int, output_dims: List[int], layers = 1, hidden_size: int = 200):
        super().__init__(False)
        self.layers = layers
        self.output_dims = output_dims

        y_len = sum(output_dims)

        self.input_query = nn.Linear(10 * 47 * N, hidden_size)
        self.hiddens = nn.ModuleList(
                        [nn.Linear(hidden_size, hidden_size) for _ in range(layers - 1)])
        self.output_query = nn.Linear(hidden_size, y_len)
    
    def distribution(self, state:_SceneState) -> torch.Tensor:
        assert not state.generate_w
        if len(state.oh_state) == 0:
            p = state.probability_vector()
            z = torch.relu(self.input_query(p))
            for i in range(self.layers - 1):
                z = torch.relu(self.hiddens[i](z))
            self.logits = self.output_query(z)
        i = len(state.oh_state)
        dim = self.output_dims[i]
        seen_so_far = sum(self.output_dims[:i])
        if dim > 1:
            dist = torch.softmax(self.logits[..., seen_so_far:seen_so_far + dim], -1)
            return dist
        dist = torch.sigmoid(self.logits[..., seen_so_far: seen_so_far + dim])
        return dist
    
    def beam(self, initial_state: _SceneState, beam_size = int):
        state = initial_state
        for _ in range(len(self.output_dims)):
            dist = self.distribution(state)
            probs, action = torch.max(dist, -1)
            state = state.next_state(action)
        return InferenceResult(state, None, None, None, None)