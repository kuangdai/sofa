import torch
import torch.nn as nn


class SofaNet(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        self.fcs_alpha = nn.ModuleList()
        self.fcs_xp = nn.ModuleList()
        self.fcs_yp = nn.ModuleList()
        hidden_sizes = [1] + hidden_sizes + [1]
        for i in range(len(hidden_sizes) - 1):
            self.fcs_alpha.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fcs_xp.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fcs_yp.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

    def forward(self, t):
        # forward
        z = torch.tensor(0., requires_grad=True, device=t.device)
        t = t.unsqueeze(1) + z
        alpha, xp, yp = t, t, t
        for fc_alpha, fc_xp, fc_yp in zip(self.fcs_alpha[:-1], self.fcs_xp[:-1], self.fcs_yp[:-1]):
            alpha = torch.relu(fc_alpha(alpha))
            xp = torch.relu(fc_xp(xp))
            yp = torch.relu(fc_yp(yp))
        alpha = (self.fcs_alpha[-1](alpha) * t).squeeze(1)  # `* t` for alpha(0) = 0
        xp = (self.fcs_xp[-1](xp) * t).squeeze(1)
        yp = (self.fcs_yp[-1](yp) * t).squeeze(1)

        # derivatives
        dummy = torch.ones_like(alpha, requires_grad=True, device=t.device)
        o_alpha = (dummy * alpha).sum()
        o_xp = (dummy * xp).sum()
        o_yp = (dummy * yp).sum()
        o_alpha_z = torch.autograd.grad(o_alpha, z, create_graph=True)[0]
        o_xp_z = torch.autograd.grad(o_xp, z, create_graph=True)[0]
        o_yp_z = torch.autograd.grad(o_yp, z, create_graph=True)[0]
        dt_alpha = torch.autograd.grad(o_alpha_z, dummy, create_graph=True)[0]
        dt_xp = torch.autograd.grad(o_xp_z, dummy, create_graph=True)[0]
        dt_yp = torch.autograd.grad(o_yp_z, dummy, create_graph=True)[0]
        return alpha, xp, yp, dt_alpha, dt_xp, dt_yp
