import torch
from torch import nn

class LMCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self, p_hat, gt):
        p_hat = p_hat[:, :-1, :]
        gt = gt[:, 1:]
        loss = self.func(p_hat.reshape(-1, p_hat.shape[-1]), gt.reshape(-1))
        return loss


def get_scheduler(optimizer, lr, steps_per_epoch, epochs):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        anneal_strategy="cos",
        max_lr=lr,
        pct_start=0.2
    )