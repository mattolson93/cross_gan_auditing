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

    def __init__(self, k: int, temp: float, abs: bool, reduce: str, otherweight: float) -> None:
        super().__init__()
        self.k = k
        self.temp = temp
        self.abs = abs
        self.reduce = reduce
        self.otherweight = otherweight
        #         self.iter = 0

    def forward(self, feats1: torch.Tensor, feats2, overlap_inds, bs) -> torch.Tensor:
        n_samples = len(feats1) + len(feats2)
        n_feats1 = len(feats1)
        #assert (n_samples % self.k) == 0, "Batch size is not divisible by given k!"

        labels1 = torch.arange(len(overlap_inds)).repeat_interleave(bs)
        labels2 = -torch.ones_like(labels1)
        next_class = labels1[-1] + 1
        for i, is_overlap in enumerate(overlap_inds):
            bi = i *bs
            if is_overlap:
                class_id = i
            else:
                class_id = next_class
                next_class+=1

            labels2[bi:bi+bs] = class_id

        feats = torch.cat([feats1,  feats2])
        labels = torch.cat([labels1, labels2])
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(feats.device)
        # similarity matrix
        sim = torch.mm(feats, feats.t().contiguous())

        cross_weighting = torch.ones_like(sim)
        cross_weighting[len(feats1):, :len(feats1)] = self.otherweight
        cross_weighting[:len(feats1), len(feats1):] = self.otherweight

        if self.abs:
            exit("simclr abs not implemented")
            sim = torch.abs(sim)

        sim = torch.exp(sim * self.temp )

        # mask for pairs
        mask = labels.bool()
       

        #import pdb; pdb.set_trace()
        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        pos_mask = mask * diag_mask
        neg_mask = ~mask
        total_loss = 0
        total_pos = 0
        total_acc = []
        #import pdb; pdb.set_trace()
        for i in range(sim.shape[0]):
            cur_pos = sim[i][pos_mask[i]]
            cur_neg = sim[i][neg_mask[i]]
            cur_loss = (cross_weighting[i][pos_mask[i]] * -torch.log(cur_pos / cur_neg.sum())).sum()

            total_pos  += cur_pos.shape[0]
            total_loss += cur_loss
            if self.otherweight == 0.0:
                if i < n_feats1:
                    total_acc.append(((sim[i][:n_feats1][pos_mask[i][:n_feats1]] > cur_neg.unsqueeze(1)).float().mean(0) == 1.0).float())
                else:
                    total_acc.append(((sim[i][n_feats1:][pos_mask[i][n_feats1:]] > cur_neg.unsqueeze(1)).float().mean(0) == 1.0).float())
            else:
                total_acc.append(((cur_pos > cur_neg.unsqueeze(1)).float().mean(0) == 1.0).float())

        acc = torch.cat(total_acc).mean()
        loss = total_loss / total_pos
        return acc, loss

        # pos and neg similarity and remove self similarity for pos
        '''pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
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
        return acc, loss'''
