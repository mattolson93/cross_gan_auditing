'''import torch

from colat.models.abstract import Model


class Fixed(Model):
    """K global fixed directions"""

    def __init__(
        self, k: int, size: int, alpha: float = 0.1, normalize: bool = True
    ) -> None:
        super().__init__(k=k, size=size, alpha=alpha, normalize=normalize)

        self.k = k
        self.size = size
        self.params = torch.nn.Parameter(torch.randn(k, size))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        import pdb; pdb.set_trace()
        #  apply all directions to each batch element
        #[bs, size]
        z = torch.reshape(z, [1, -1, self.size])
        #[1,bs,size]
        z = z.repeat(
            (
                self.k,
                1,
                1,
            )
        )
        #[k,bs,size]
        #  add directions
        all_directions = self.post_process(self.params) #[k,size]

        z += torch.reshape(all_directions, (self.k, 1, self.size))

        # reshape
        return torch.reshape(z, [-1, self.size])

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.params)[k : k + 1, :]


'''
import torch

from colat.models.abstract import Model


class Fixed(Model):
    """K global fixed directions"""

    def __init__(
        self, k: int, size: int,  batch_k: int, alpha: float = 0.1, normalize: bool = True
    ) -> None:
    
        super().__init__(k=k, batch_k=batch_k, size=size, alpha=alpha, normalize=normalize)

        self.k = k
        self.batch_k = min(batch_k, k)
        self.size = size
        self.params = torch.nn.Parameter(torch.randn(k, size))
        self.selected_k = None


    def forward(self, z: torch.Tensor, selected_k=None) -> torch.Tensor:
        #  apply all directions to each batch element
        #[bs,size]
        z = torch.reshape(z, [1, -1, self.size])
        #[1,bs,size]
        z = z.repeat(
            (
                self.batch_k,
                1,
                1,
            )
        )
        #[batch_k, batch, size]

        #  add directions
        selected_k = torch.randperm(self.k)[:self.batch_k] if selected_k is None else selected_k
        all_directions = torch.reshape(self.post_process(self.params[selected_k]), (self.batch_k, 1, self.size))
        z += all_directions

        self.selected_k = selected_k
        # reshape
        return torch.reshape(z, [-1, self.size])

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.params)[k : k + 1, :]
