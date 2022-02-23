import torch

from colat.models.abstract import Model
from colat.utils.net_utils import create_mlp


class LinearConditional(Model):
    """K directions linearly conditional on latent code"""

    def __init__(
        self,
        k: int,
        batch_k: int,
        size: int,
        alpha: float = 0.1,
        normalize: bool = True,
        bias: bool = False,
        batchnorm: bool = False,
    ) -> None:
        super().__init__(k=k, batch_k=batch_k, size=size, alpha=alpha, normalize=normalize)
        self.k = k
        self.size = size
        self.batch_k = min(batch_k, k)
        self.selected_k = None
        self.need_perm = self.k != batch_k

        # make mlp net
        self.nets = torch.nn.ModuleList()

        for i in range(k):
            net = create_mlp(
                depth=1,
                in_features=size,
                middle_features=-1,
                out_features=size,
                bias=bias,
                batchnorm=batchnorm,
                final_norm=batchnorm,
            )
            self.nets.append(net)

    def get_params(self):
        #param_list = torch.cat([net[0].weight.detach().cpu() for net in self.nets])
        return None

    def forward(self, z: torch.Tensor, selected_k=None, unselected_k=None) -> torch.Tensor:
        #  apply all directions to each batch element
        z = torch.reshape(z, [1, -1, self.size])
        z1 = z.repeat(
            (
                self.batch_k,
                1,
                1,
            )
        )
        #import pdb; pdb.set_trace()

        

        # calculate directions
        if selected_k is None:
            if self.need_perm:
                random_k = torch.randperm(self.k)
                selected_k = torch.sort(random_k[:self.batch_k])[0]
                unselected_k = torch.sort(random_k[self.batch_k:])[0]
            else:
                selected_k = torch.arange(self.k)

        dz = []
        for i, k in enumerate(selected_k):
            res_dz = self.nets[k](z1[i, ...])
            res_dz = self.post_process(res_dz)
            dz.append(res_dz)

        dz = torch.stack(dz)
        z1 = z1 + dz

        if self.need_perm == False: return torch.reshape(z1, [-1, self.size])

        z2 = z.repeat(
            (
                self.k - self.batch_k,
                1,
                1,
            )
        )

        dz = []
        with torch.no_grad():
            for i, k in enumerate(unselected_k):
                res_dz = self.nets[k](z2[i, ...])
                res_dz = self.post_process(res_dz)
                dz.append(res_dz)

            dz = torch.stack(dz)
            z2 = z2 + dz

        self.selected_k = selected_k
        self.unselected_k = unselected_k

        #  add directions

        return torch.reshape(z1, [-1, self.size]), torch.reshape(z2, [-1, self.size])

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.nets[k](z))


class NonlinearConditional(Model):
    """K directions nonlinearly conditional on latent code"""

    def __init__(
        self,
        k: int,
        batch_k: int,
        size: int,
        depth: int,
        alpha: float = 0.1,
        normalize: bool = True,
        bias: bool = True,
        batchnorm: bool = True,
        final_norm: bool = False,
    ) -> None:
        super().__init__(k=k, batch_k=batch_k, size=size, alpha=alpha, normalize=normalize)
        self.k = k
        self.size = size
        self.batch_k = min(batch_k, k)
        self.selected_k = None

        # make mlp net
        self.nets = torch.nn.ModuleList()

        for i in range(k):
            net = create_mlp(
                depth=depth,
                in_features=size,
                middle_features=size,
                out_features=size,
                bias=bias,
                batchnorm=batchnorm,
                final_norm=final_norm,
            )
            self.nets.append(net)

    def forward(self, z: torch.Tensor, selected_k=None) -> torch.Tensor:
        #  apply all directions to each batch element
        z = torch.reshape(z, [1, -1, self.size])
        z = z.repeat(
            (
                self.batch_k,
                1,
                1,
            )
        )

        #  calculate directions
        selected_k = torch.randperm(self.k)[:self.batch_k] if selected_k is None else selected_k
        self.selected_k = selected_k
        dz = []
        for i, k in enumerate(selected_k):
            res_dz = self.nets[k](z[i, ...])
            res_dz = self.post_process(res_dz)
            dz.append(res_dz)

        dz = torch.stack(dz)

        #  add directions
        z = z + dz

        return torch.reshape(z, [-1, self.size])

    def forward_single(self, z: torch.Tensor, k: int) -> torch.Tensor:
        return z + self.post_process(self.nets[k](z))
