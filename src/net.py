import torch
import torch.nn as nn


class SofaNetEllipse(nn.Module):
    def __init__(self, ab0, hidden_sizes):
        super().__init__()
        self.fcs = nn.ModuleList()
        # [1] for alpha, [2] for [a, b]
        hidden_sizes = [1] + hidden_sizes + [2]
        for i in range(len(hidden_sizes) - 1):
            self.fcs.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        # initialized so that output is [sqrt(a), sqrt(b)]
        self.fcs[-1].weight.data[:] = 0.
        self.fcs[-1].bias.data[:] = torch.sqrt(ab0)

    def forward(self, alpha, returns_ab=False):
        n = len(alpha)
        assert n % 2 == 1 and torch.isclose(alpha[n // 2], torch.tensor(torch.pi / 2))

        # take half of alpha for mirror symmetry
        x = alpha[:n // 2 + 1, None]

        # zcs
        z = torch.tensor(0., requires_grad=True, device=alpha.device)
        x = x + z

        # layers
        for fc in self.fcs[:-1]:
            x = torch.tanh(fc(x))
        x = self.fcs[-1](x)
        x = x ** 2

        # gradient by zcs
        dummy = torch.ones_like(x, requires_grad=True)
        omega = (dummy * x).sum()
        omega_z = torch.autograd.grad(omega, z, create_graph=True)[0]
        dab = torch.autograd.grad(omega_z, dummy, create_graph=True)[0]

        # curve p
        a, b = x[:, 0], x[:, 1]
        da, db = dab[:, 0], dab[:, 1]
        alpha = alpha[:n // 2 + 1]
        xp = a * (torch.cos(alpha) - 1)
        yp = b * torch.sin(alpha)
        xp_prime = -a * torch.sin(alpha) + da * (torch.cos(alpha) - 1)
        yp_prime = b * torch.cos(alpha) + db * torch.sin(alpha)

        # mirror symmetry
        xp = torch.cat((xp, 2 * xp[-1] - xp[:-1].flip(dims=[0])))
        yp = torch.cat((yp, yp[:-1].flip(dims=[0])))
        xp_prime = torch.cat((xp_prime, xp_prime[:-1].flip(dims=[0])))
        yp_prime = torch.cat((yp_prime, -yp_prime[:-1].flip(dims=[0])))

        if returns_ab:
            return xp, yp, xp_prime, yp_prime, a, b
        return xp, yp, xp_prime, yp_prime
