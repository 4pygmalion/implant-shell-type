from abc import ABC, abstractmethod

import torch
import numpy as np


def min_max_scaling(array: np.ndarray) -> np.ndarray:

    a_min, a_max = array.min(), array.max()

    return (array - a_min) / (a_max - a_min) if a_max - a_min != 0 else array


class BaseCAM(ABC):
    def __init__(
        self, model: torch.nn.Module, target_layer: torch.nn.Module, device: str
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None

        self._register_hook()

    @abstractmethod
    def _forward(self, x):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    def _hook_forward_activation(self, module, input, output):
        self.activations = output

    def _hook_backward_activation(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _register_hook(self):
        """Activation map을 반환하는 target layer에 hook을 걸음"""
        self.target_layer.register_forward_hook(self._hook_forward_activation)
        self.target_layer.register_backward_hook(self._hook_backward_activation)
