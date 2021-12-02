import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Computes the contrastive loss

    Args:
        - k: the number of transformations per batch
        - temperature: temp to scale before exponential

    Shape:
        - Input: the raw, feature scores.
                tensor of size :math:`(k x minibatch, F)`, with F the number of features
                expects first axis to be ordered by transformations first (i.e., the
                first "minibatch" elements is for first transformations)
        - Output: scalar
    """

    def __init__(self, k: int, temp: float, abs: bool, reduce: str) -> None:
        super().__init__()
        self.k = k
        self.temp = temp
        self.abs = abs
        self.reduce = reduce
        #         self.iter = 0

    def forward(self, feats: torch.Tensor, overlap_inds) -> torch.Tensor:
        n_samples = len(feats)
        #import pdb; pdb.set_trace()
        assert (n_samples % self.k) == 0, "Batch size is not divisible by given k!"

        # similarity matrix
        sim = torch.mm(feats, feats.t().contiguous())

        if self.abs:
            sim = torch.abs(sim)

        #         if (self.iter % 100) == 0:
        #             print(sim)
        #          self.iter += 1

        sim = torch.exp(sim * self.temp)

        k2 = 2*self.k
        feats2_start = int(n_samples/2)

        # mask for pairs
        mask = torch.zeros((n_samples, n_samples), device=sim.device).bool()
        for i in range(k2):
            start, end = i * (n_samples // k2), (i + 1) * (n_samples // k2)
            mask[start:end, start:end] = 1
            if i < overlap_inds:
                mask[start+feats2_start:end+feats2_start, start:end] = 1
                mask[start:end, start+feats2_start:end+feats2_start] = 1


        #print("here")

        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        # pos and neg similarity and remove self similarity for pos
        pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
        neg = sim.masked_select(~mask).view(n_samples, -1)

        if self.reduce == "mean":
            pos = pos.mean(dim=-1)
            neg = neg.mean(dim=-1)
        elif self.reduce == "sum":
            pos = pos.sum(dim=-1)
            neg = neg.sum(dim=-1)
        else:
            raise ValueError("Only mean and sum is supported for reduce method")

        acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).mean()
        return acc, loss
