import torch
import torch.nn as nn


class SofaNet(nn.Module):
    def __init__(self, ab, hidden_sizes):
        super().__init__()
        self.na, self.nb = ab.shape[0], ab.shape[1]

        # weights
        weights, biases = [], []
        hidden_sizes = [1] + hidden_sizes + [2]  # 1 for alpha, 2 for a, b
        for i in range(len(hidden_sizes) - 1):
            weights.append(torch.randn(self.na, self.nb, hidden_sizes[i + 1], hidden_sizes[i]))
            biases.append(torch.zeros(self.na, self.nb, hidden_sizes[i + 1]))

        # initial condition
        weights[-1][:] = 0.
        biases[-1][:] = ab[:]

        # trainable
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)

    def forward(self, alpha, compute_gradients=False):
        # expand size of alpha
        alpha = alpha[None, None, :, None].expand(self.na, self.nb, -1, -1)

        # zcs
        z = torch.tensor(0., requires_grad=True, device=alpha.device)
        if compute_gradients:
            alpha = alpha + z

        # parity
        x = torch.sin(alpha)

        # layers
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = torch.einsum("MNij,MNBj->MNBi", w, x) + b[:, None, :]
            x = torch.relu(x)
        x = torch.einsum("MNij,MNBj->MNBi", self.weights[-1], x) + self.biases[-1][:, None, :]

        # gradient
        if compute_gradients:
            # autodiff by zcs
            dummy = torch.ones_like(x, requires_grad=True)
            omega = (dummy * x).sum()
            omega_z = torch.autograd.grad(omega, z, create_graph=True)[0]
            dx = torch.autograd.grad(omega_z, dummy, create_graph=True)[0]
            return x, dx
        else:
            return x
