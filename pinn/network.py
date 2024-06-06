import torch
import torch.nn as nn


class SofaNet(nn.Module):
    def __init__(self, hidden_sizes=None, tanh=False, alpha_scaling=1., xp_scaling=1., yp_scaling=1.):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 128, 128]
        self.fcs_alpha = nn.ModuleList()
        self.fcs_xp = nn.ModuleList()
        self.fcs_yp = nn.ModuleList()
        hidden_sizes = [1] + hidden_sizes + [1]
        for i in range(len(hidden_sizes) - 1):
            self.fcs_alpha.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fcs_xp.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fcs_yp.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

            # scaling weight and bias for diverse initialization patterns
            self.fcs_alpha[-1].weight.data *= alpha_scaling
            self.fcs_alpha[-1].bias.data *= alpha_scaling
            self.fcs_xp[-1].weight.data *= xp_scaling
            self.fcs_xp[-1].bias.data *= xp_scaling
            self.fcs_yp[-1].weight.data *= yp_scaling
            self.fcs_yp[-1].bias.data *= yp_scaling
        self.act = torch.tanh if tanh else torch.relu

    def forward(self, t):
        # forward
        z = torch.tensor(0., requires_grad=True, device=t.device)
        t = t.unsqueeze(1) + z
        alpha, xp, yp = t, t, t
        for fc_alpha, fc_xp, fc_yp in zip(self.fcs_alpha[:-1], self.fcs_xp[:-1], self.fcs_yp[:-1]):
            alpha = self.act(fc_alpha(alpha))
            xp = self.act(fc_xp(xp))
            yp = self.act(fc_yp(yp))
        alpha = self.fcs_alpha[-1](alpha).squeeze(1)
        xp = self.fcs_xp[-1](xp).squeeze(1)
        yp = self.fcs_yp[-1](yp).squeeze(1)

        # initial condition at t=0
        alpha = alpha - alpha[0]
        xp = xp - xp[0]
        yp = yp - yp[0]

        # constrain only sign of variables
        alpha = torch.abs(alpha)
        xp = -torch.abs(xp)
        yp = torch.abs(yp)

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
