from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


class FCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(FCN, self).__init__()
        layers = []
        hidden_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i < len(hidden_sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    parser = ArgumentParser("Moving sofa",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--alpha-samples", type=int,
                        default=1024, help="number of alpha samples")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[64, 64], help="hidden sizes of NN")
    parser.add_argument("--lr", type=float,
                        default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        default=10000, help="number of epochs")
    parser.add_argument("--device", type=str,
                        default="cpu", help="training device")
    args = parser.parse_args()

    # sample alpha data
    all_alpha = torch.rand((args.epochs, args.alpha_samples)) * torch.pi / 4
    all_alpha[:, 0] = 0.
    all_alpha[:, -1] = torch.pi / 4

    # model
    model = FCN(1, 3, args.hidden_sizes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training
    progress_bar = trange(args.epochs)
    for epoch in progress_bar:
        # alpha
        alpha = all_alpha[epoch].clone().requires_grad_()

        # forward
        xp, yp, As = model.forward(alpha.unsqueeze(-1)).T

        # initial condition
        xp = xp * torch.sin(alpha)
        yp = yp * torch.sin(alpha)

        # ode loss
        xp_d = torch.autograd.grad(xp.sum(), alpha, create_graph=True)[0]
        yp_d = torch.autograd.grad(yp.sum(), alpha, create_graph=True)[0]
        As_d = torch.autograd.grad(As.sum(), alpha, create_graph=True)[0]
        xp_dd = torch.autograd.grad(xp_d.sum(), alpha, create_graph=True)[0]
        yp_dd = torch.autograd.grad(yp_d.sum(), alpha, create_graph=True)[0]
        ode = As_d - (
                (torch.cos(2 * alpha) - 1) / 2 +
                (5 * torch.sin(alpha) - 3 * torch.sin(3 * alpha)) / 4 * xp_d +
                (torch.cos(3 * alpha) - torch.cos(alpha)) / 4 * xp_dd -
                torch.sin(alpha) * yp +
                3 * (torch.cos(3 * alpha) - torch.cos(alpha)) / 4 * yp_d -
                (torch.sin(3 * alpha) - 3 * torch.sin(alpha)) / 4 * yp_dd
        )
        loss_ode = torch.nn.functional.mse_loss(ode, torch.zeros_like(ode))

        # area loss
        loss_area = -As[-1]

        # total loss
        loss = loss_ode + loss_area

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss_ode=f"{loss_ode.item():.4e}",
                                 loss_area=f"{loss_area.item():.4e}")
