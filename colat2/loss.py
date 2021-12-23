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
        for i, is_overlap in enumerate(overlap_inds):
            start1, end1 = i * (n_samples // k2), (i + 1) * (n_samples // k2)
            mask[start1:end1, start1:end1] = 1

            start2, end2 = start1 + feats2_start , end1 + feats2_start
            mask[start2:end2, start2:end2] = 1

            if is_overlap:
                mask[start1:end1, start2:end2] = 1
                mask[start2:end2, start1:end1] = 1



        #print("here")
        #import pdb; pdb.set_trace()

        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        # pos and neg similarity and remove self similarity for pos
        #pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
        #neg = sim.masked_select(~mask).view(n_samples, -1)
        pos_mask = mask * diag_mask 
        neg_mask = (~mask)
        pos = sim * pos_mask
        neg = sim * neg_mask 

        if self.reduce == "mean":
            pos = pos.sum(dim=-1) / pos_mask.sum(-1)
            neg = neg.sum(dim=-1) / neg_mask.sum(-1)
        elif self.reduce == "sum":
            pos = pos.sum(dim=-1)
            neg = neg.sum(dim=-1)
        else:
            raise ValueError("Only mean and sum is supported for reduce method")

        acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).sum()
        return acc, loss
