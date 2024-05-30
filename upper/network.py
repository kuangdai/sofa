import torch
import torch.nn as nn


class SofaNet(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        self.fcs_u1 = nn.ModuleList()
        self.fcs_u2 = nn.ModuleList()
        hidden_sizes = [1] + hidden_sizes + [1]
        for i in range(len(hidden_sizes) - 1):
            self.fcs_u1.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fcs_u2.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

    def forward(self, alpha):
        u1, u2 = alpha[:, None], alpha[:, None]
        for fc_u1, fc_u2 in zip(self.fcs_u1[:-1], self.fcs_u2[:-1]):
            u1 = torch.relu(fc_u1(u1))
            u2 = torch.relu(fc_u2(u2))
        u1 = self.fcs_u1[-1](u1).squeeze(1) * alpha
        u2 = self.fcs_u2[-1](u2).squeeze(1) * alpha
        return u1, u2
