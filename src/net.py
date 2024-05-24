import torch
import torch.nn as nn


class SofaNet(nn.Module):
    def __init__(self, n_in, n_t, hidden_sizes, a0=0.5, b0=0.6, velocity_mode=True):
        super().__init__()
        self.n_in, self.n_t = n_in, n_t
        self.velocity_mode = velocity_mode
        n_out = n_t + 2 if velocity_mode else n_t

        # network
        self.fcs_alpha = nn.ModuleList()
        self.fcs_xp = nn.ModuleList()
        self.fcs_yp = nn.ModuleList()
        hidden_sizes = [n_in] + hidden_sizes + [n_out]
        for i in range(len(hidden_sizes) - 1):
            self.fcs_alpha.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fcs_xp.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.fcs_yp.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # initialized to an elliptical path
        d_alpha = 0 if velocity_mode else torch.pi / 2 / (n_t - 1)
        alpha = torch.linspace(-d_alpha, torch.pi / 2 + d_alpha, n_out)
        if velocity_mode:
            initial_alpha = torch.ones(n_t) * torch.pi / 2
            initial_xp = a0 * (-torch.sin(2 * alpha)) * torch.pi
            initial_yp = b0 * torch.cos(2 * alpha) * torch.pi
        else:
            initial_alpha = alpha
            initial_xp = a0 * (torch.cos(2 * alpha) - 1)
            initial_yp = b0 * torch.sin(2 * alpha)
        self.fcs_alpha[-1].weight.data[:] = 0.
        self.fcs_xp[-1].weight.data[:] = 0.
        self.fcs_yp[-1].weight.data[:] = 0.
        self.fcs_alpha[-1].bias.data[:] = initial_alpha
        self.fcs_xp[-1].bias.data[:] = initial_xp
        self.fcs_yp[-1].bias.data[:] = initial_yp

    def forward(self):
        # forward
        # value of x is not important
        device = self.fcs_xp[-1].weight.device
        x = torch.linspace(-1, 1., self.n_in, device=device)
        x_alpha, x_xp, x_yp = x, x, x
        for fc_alpha, fc_xp, fc_yp in zip(self.fcs_alpha[:-1], self.fcs_xp[:-1], self.fcs_yp[:-1]):
            x_alpha = torch.relu(fc_alpha(x_alpha))
            x_xp = torch.relu(fc_xp(x_xp))
            x_yp = torch.relu(fc_yp(x_yp))
        x_alpha = self.fcs_alpha[-1](x_alpha)
        x_xp = self.fcs_xp[-1](x_xp)
        x_yp = self.fcs_yp[-1](x_yp)

        t = torch.linspace(0., 1., self.n_t, device=device)
        if self.velocity_mode:
            # integration
            dt_alpha, dt_xp, dt_yp = x_alpha, x_xp, x_yp
            dt_alpha_mid = (dt_alpha[:-1] + dt_alpha[1:]) / 2 / (self.n_t - 1)
            alpha = torch.cat((torch.zeros([1], device=device), dt_alpha_mid.cumsum(dim=0)))
            dt_xp_mid = (dt_xp[:-1] + dt_xp[1:]) / 2 / (self.n_t - 1)
            xp = torch.cat((torch.zeros([1], device=device), dt_xp_mid.cumsum(dim=0)))
            dt_yp_mid = (dt_yp[:-1] + dt_yp[1:]) / 2 / (self.n_t - 1)
            yp = torch.cat((torch.zeros([1], device=device), dt_yp_mid.cumsum(dim=0)))
        else:
            # differentiation
            dt2 = (t[1] - t[0]) * 2
            alpha, xp, yp = x_alpha[1:-1], x_xp[1:-1], x_yp[1:-1]
            dt_alpha = (x_alpha[2:] - x_alpha[:-2]) / dt2
            dt_xp = (x_xp[2:] - x_xp[:-2]) / dt2
            dt_yp = (x_yp[2:] - x_yp[:-2]) / dt2
        return t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp
