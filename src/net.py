import torch
import torch.nn as nn


class SofaNetEllipse(nn.Module):
    def __init__(self, ab_initial, hidden_sizes):
        super().__init__()
        self.n_ab = ab_initial.shape[0]

        # parameter for a
        self.sqrt_a = nn.Parameter(torch.sqrt(ab_initial[:, 0]))

        # net for b
        weights, biases = [], []
        hidden_sizes = [1] + hidden_sizes + [1]
        for i in range(len(hidden_sizes) - 1):
            weights.append(torch.randn(self.n_ab, hidden_sizes[i + 1], hidden_sizes[i]))
            biases.append(torch.zeros(self.n_ab, hidden_sizes[i + 1]))
        weights[-1][:] = 0.
        biases[-1][:] = torch.sqrt(ab_initial[:, 1])
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, alpha, compute_gradients=False):
        #####
        # a #
        #####
        a = (self.sqrt_a ** 2)[:, None].expand(-1, len(alpha))

        #####
        # b #
        #####
        # expand size of alpha
        alpha = alpha[None, :, None].expand(self.n_ab, -1, -1)

        # zcs
        z = torch.tensor(0., requires_grad=True, device=alpha.device)
        if compute_gradients:
            alpha = alpha + z

        # parity
        x = torch.sin(alpha)

        # layers
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = torch.einsum("Nij,NBj->NBi", weight, x) + bias[:, None, :]
            x = torch.relu(x)
        x = torch.einsum("Nij,NBj->NBi", self.weights[-1], x) + self.biases[-1][:, None, :]
        b = (x ** 2).squeeze(2)

        if not compute_gradients:
            return torch.stack((a, b), dim=2)

        # gradient by zcs
        dummy = torch.ones_like(x, requires_grad=True)
        omega = (dummy * b).sum()
        omega_z = torch.autograd.grad(omega, z, create_graph=True)[0]
        db = torch.autograd.grad(omega_z, dummy, create_graph=True)[0]
        da = torch.zeros_like(db)
        return torch.stack((a, b), dim=2), torch.stack((da, db), dim=2)
