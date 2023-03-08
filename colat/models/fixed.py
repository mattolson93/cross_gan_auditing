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
        self.need_perm = self.k != batch_k
        self.size = size
        self.params = torch.nn.Parameter(torch.randn(k, size))
        self.need_perm = self.k != batch_k
        self.selected_k = None

    def get_params(self):
        return self.params.detach().cpu().numpy()



    def forward(self, z: torch.Tensor, selected_k=None, unselected_k=None, pos_and_neg=False) -> torch.Tensor:
        if pos_and_neg:
            return self._forward_pos_neg(z, selected_k)
        #  apply all directions to each batch element
        #[bs,size]
        z = torch.reshape(z, [1, -1, self.size])
        z1 = z.repeat(
            (
                self.batch_k,
                1,
                1,
            )
        )
        # calculate directions
        if selected_k is None:
            if self.need_perm:
                random_k = torch.randperm(self.k)
                selected_k = torch.sort(random_k[:self.batch_k])[0]
                unselected_k = torch.sort(random_k[self.batch_k:])[0]
            else:
                selected_k = torch.arange(self.k)
        #import pdb; pdb.set_trace()
        all_directions1 = torch.reshape(self.post_process(self.params[selected_k]), (self.batch_k, 1, self.size))
        z1 += all_directions1

        if self.need_perm == False: return torch.reshape(z1, [-1, self.size])
        z2 = z.repeat(
            (
                self.k - self.batch_k,
                1,
                1,
            )
        )

        all_directions2 = torch.reshape(self.post_process(self.params[unselected_k]), (self.k - self.batch_k, 1, self.size))
        z2 += all_directions2

        self.selected_k = selected_k.detach()
        self.unselected_k = unselected_k.detach()
        
        # reshape
        return torch.reshape(z1, [-1, self.size]), torch.reshape(z2, [-1, self.size])



    def _forward_pos_neg(self, z, selected_k):
        z = torch.reshape(z, [1, -1, self.size])
        z1 = z.repeat(
            (
                self.batch_k,
                1,
                1,
            )
        )
        # calculate directions
        if selected_k is None:
            if self.need_perm:
                random_k = torch.randperm(self.k)
                selected_k = torch.sort(random_k[:self.batch_k])[0]
            else:
                selected_k = torch.arange(self.k)
        #import pdb; pdb.set_trace()
        all_directions1 = torch.reshape(self.post_process(self.params[selected_k]), (self.batch_k, 1, self.size))
        z1_neg = torch.clone(z1) - all_directions1
        z1 += all_directions1
        
        self.selected_k = selected_k.detach()
        
        # reshape
        return torch.reshape(z1, [-1, self.size]), torch.reshape(z1_neg, [-1, self.size]), 

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.params)[k : k + 1, :]
