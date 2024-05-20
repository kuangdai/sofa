import torch
import torch.nn as nn


class SofaNetEllipse(nn.Module):
    def __init__(self, ab_initial, hidden_sizes):
        super().__init__()
        self.n_ab = ab_initial.shape[0]
        weights, biases = [], []
        # [1] for alpha, [2] for a, b
        hidden_sizes = [1] + hidden_sizes + [2]
        for i in range(len(hidden_sizes) - 1):
            weights.append(torch.randn(self.n_ab, hidden_sizes[i + 1], hidden_sizes[i]))
            biases.append(torch.zeros(self.n_ab, hidden_sizes[i + 1]))
        # initialized so that output is [sqrt(a), sqrt(b)]
        weights[-1][:] = 0.
        biases[-1][:] = torch.sqrt(ab_initial)
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, alpha):
        n = len(alpha)
        assert n % 2 == 1 and torch.isclose(alpha[n // 2], torch.tensor(torch.pi / 2))

        # take half of alpha for mirror symmetry
        x = alpha[None, :n // 2 + 1, None].expand(self.n_ab, -1, -1)

        # zcs
        z = torch.tensor(0., requires_grad=True, device=alpha.device)
        x = x + z

        # layers
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = torch.einsum("Nij,NBj->NBi", weight, x) + bias[:, None, :]
            x = torch.tanh(x)
        x = torch.einsum("Nij,NBj->NBi", self.weights[-1], x) + self.biases[-1][:, None, :]
        x = x ** 2

        # gradient by zcs
        dummy = torch.ones_like(x, requires_grad=True)
        omega = (dummy * x).sum()
        omega_z = torch.autograd.grad(omega, z, create_graph=True)[0]
        dab = torch.autograd.grad(omega_z, dummy, create_graph=True)[0]

        # curve p
        a, b = x[:, :, 0], x[:, :, 1]
        da, db = dab[:, :, 0], dab[:, :, 1]
        alpha = alpha[None, :n // 2 + 1].expand(self.n_ab, -1)
        xp = a * (torch.cos(alpha) - 1)
        yp = b * torch.sin(alpha)
        xp_prime = -a * torch.sin(alpha) + da * (torch.cos(alpha) - 1)
        yp_prime = b * torch.cos(alpha) + db * torch.sin(alpha)

        # mirror symmetry
        xp = torch.cat((xp, 2 * xp[:, -1][:, None] - xp[:, :-1].flip(dims=[1])), dim=1)
        yp = torch.cat((yp, yp[:, :-1].flip(dims=[1])), dim=1)
        xp_prime = torch.cat((xp_prime, xp_prime[:, :-1].flip(dims=[1])), dim=1)
        yp_prime = torch.cat((yp_prime, -yp_prime[:, :-1].flip(dims=[1])), dim=1)
        return xp, yp, xp_prime, yp_prime
