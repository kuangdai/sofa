import torch
import torch.nn as nn


class SofaNetEllipse(nn.Module):
    def __init__(self, ab_initial, hidden_sizes):
        super().__init__()
        self.n_ab = ab_initial.shape[0]

        # parameter for a
        # train square root to make sure (a, b) are positive
        self.sqrt_a = nn.Parameter(torch.sqrt(ab_initial[:, 0]))

        # net for b
        weights, biases = [], []
        hidden_sizes = [1] + hidden_sizes + [1]
        for i in range(len(hidden_sizes) - 1):
            weights.append(torch.randn(self.n_ab, hidden_sizes[i + 1], hidden_sizes[i]))
            biases.append(torch.zeros(self.n_ab, hidden_sizes[i + 1]))
        weights[-1][:] = 0.
        biases[-1][:, 0] = torch.sqrt(ab_initial[:, 1])
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, alpha):
        # take half of alpha for mirror symmetry
        assert len(alpha) % 2 == 1 and torch.isclose(alpha[len(alpha) // 2], torch.tensor(torch.pi / 2))
        x = alpha[None, :len(alpha) // 2 + 1, None].expand(self.n_ab, -1, -1)

        # zcs
        z = torch.tensor(0., requires_grad=True, device=alpha.device)
        x = x + z

        # layers
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = torch.einsum("Nij,NBj->NBi", weight, x) + bias[:, None, :]
            x = torch.relu(x)
        x = torch.einsum("Nij,NBj->NBi", self.weights[-1], x) + self.biases[-1][:, None, :]
        b = (x ** 2).squeeze(2)

        # gradient by zcs
        dummy = torch.ones_like(b, requires_grad=True)
        omega = (dummy * b).sum()
        omega_z = torch.autograd.grad(omega, z, create_graph=True)[0]
        db_alpha = torch.autograd.grad(omega_z, dummy, create_graph=True)[0]

        # mirror symmetry
        b = torch.cat((b, b[:, :-1].flip(dims=[1])), dim=1)
        db_alpha = torch.cat((db_alpha, -db_alpha[:, :-1].flip(dims=[1])), dim=1)

        # curve p
        a = (self.sqrt_a ** 2)[:, None].expand(-1, len(alpha))
        alpha = alpha[None, :].expand(self.n_ab, -1)
        xp = a * (torch.cos(alpha) - 1)
        yp = b * torch.sin(alpha)
        xp_prime = -a * torch.sin(alpha)
        yp_prime = b * torch.cos(alpha) + db_alpha * torch.sin(alpha)
        return xp, yp, xp_prime, yp_prime
