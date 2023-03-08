from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch


class Model(ABC, torch.nn.Module):
    """Abstract model

    Args:
        normalize: whether to normalize after feed-forward
    """

    def __init__(
        self,
        k: int,
        batch_k: int,
        size: int,
        alpha: Union[float, List[float]] = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.batch_k = batch_k
        self.size = size
        self.alpha = alpha
        self.normalize = normalize
        self.selected_k = None

    @abstractmethod
    def forward(self, z: torch.Tensor, selected_k=None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        raise NotImplementedError

    def sample_alpha(self) -> float:
        if isinstance(self.alpha, float) or isinstance(self.alpha, int):
            return self.alpha
        rand_alpha = np.random.uniform(self.alpha[0], self.alpha[1], size=1)[0]
        if -.5 < rand_alpha < 0:  rand_alpha = -.5
        elif 0 <= rand_alpha < .5: rand_alpha = .5

        self.sampled_alphas = rand_alpha
        return self.sampled_alphas

    def post_process(self, dz: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            norm = torch.norm(dz, dim=1)
            dz = dz / torch.reshape(norm, (-1, 1))
        return self.sample_alpha() * dz
