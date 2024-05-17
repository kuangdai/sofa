import torch
import torch.nn as nn


class SofaNet(nn.Module):
    def __init__(self, n_nets, hidden_sizes, ab_lower=0.05, ab_upper=1.05):
        super().__init__()
        weights = []
        biases = []
        hidden_sizes = [1] + hidden_sizes + [2]  # 1 for alpha, 2 for a, b
        for i in range(len(hidden_sizes) - 1):
            weights.append(torch.randn((n_nets, hidden_sizes[i + 1], hidden_sizes[i])))
            biases.append(torch.zeros(n_nets, hidden_sizes[i + 1]))
        self.n_nets = n_nets
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)
        self.ab_lower = ab_lower
        self.ab_upper = ab_upper

    def forward(self, alpha, compute_gradients=False):
        # expand size of alpha
        alpha = alpha[None, :, None].expand(self.n_nets, -1, -1)

        # zcs
        z = torch.tensor(0., requires_grad=True, device=alpha.device)
        if compute_gradients:
            alpha = alpha + z

        # parity
        x = torch.sin(alpha)

        # layers
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = torch.einsum("nij,nbj->nbi", w, x) + b[:, None, :]
            x = torch.relu(x)
        x = torch.einsum("nij,nbj->nbi", self.weights[-1], x) + self.biases[-1][:, None, :]

        # range
        x = torch.sigmoid(x)
        x = self.ab_lower + (self.ab_upper - self.ab_lower) * x

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
