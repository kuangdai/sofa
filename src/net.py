import torch
import torch.nn as nn


class SofaNetEllipse(nn.Module):
    def __init__(self, ab0, hidden_sizes, xy_correction):
        super().__init__()
        self.xy_correction = xy_correction
        self.fcs_ab = nn.ModuleList()  # layers for a, b
        self.fcs_kxy = nn.ModuleList()  # layers for k applied to xp, yp
        self.fcs_sxy = nn.ModuleList()  # layers for s applied to xp, yp

        # [1] for alpha, [2] for [a, b], [k_xp, k_yp], [s_xp, s_yp]
        hidden_sizes = [1] + hidden_sizes + [2]
        for i in range(len(hidden_sizes) - 1):
            self.fcs_ab.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if xy_correction:
                self.fcs_kxy.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                self.fcs_sxy.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # initialized so that output is [a, b]
        self.fcs_ab[-1].weight.data[:] = 0.
        self.fcs_ab[-1].bias.data[:] = ab0
        if xy_correction:
            # initialized so that output is [1, 1]
            self.fcs_kxy[-1].weight.data[:] = 0.
            self.fcs_kxy[-1].bias.data[:] = 1.
            # initialized so that output is [0, 0]
            self.fcs_sxy[-1].weight.data[:] = 0.
            self.fcs_sxy[-1].bias.data[:] = 0.

    def forward(self, alpha):
        # zcs
        z = torch.tensor(0., requires_grad=True, device=alpha.device)
        alpha = alpha + z

        # layers
        ab = alpha[:, None]
        for fc_ab in self.fcs_ab[:-1]:
            ab = torch.tanh(fc_ab(ab))
        ab = self.fcs_ab[-1](ab)

        # curve p by ellipse
        a, b = ab[:, 0], ab[:, 1]
        xp = a * (torch.cos(alpha) - 1)
        yp = b * torch.sin(alpha)

        # xy correction
        if self.xy_correction:
            kxy = alpha[:, None]
            for fc_kxy in self.fcs_kxy[:-1]:
                kxy = torch.relu(fc_kxy(kxy))
            kxy = self.fcs_kxy[-1](kxy)
            xp = xp * kxy[:, 0]
            yp = yp * kxy[:, 1]

        # gradient by zcs
        dummy = torch.ones_like(xp, requires_grad=True)
        omega_xp = (dummy * xp).sum()
        omega_yp = (dummy * yp).sum()
        omega_xp_z = torch.autograd.grad(omega_xp, z, create_graph=True)[0]
        omega_yp_z = torch.autograd.grad(omega_yp, z, create_graph=True)[0]
        xp_prime = torch.autograd.grad(omega_xp_z, dummy, create_graph=True)[0]
        yp_prime = torch.autograd.grad(omega_yp_z, dummy, create_graph=True)[0]
        return xp, yp, xp_prime, yp_prime
