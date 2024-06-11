import torch

from pinn.geometry import interp1d_multi_lines


def compute_area(alpha, beta1, beta2, u1, u2,
                 bound=20., n_areas=2000, return_geometry=False):
    # constants
    extend = bound * 4.
    sqrt2 = torch.sqrt(torch.tensor(2., device=alpha.device))

    # xp, yp in fixed coordinates
    xp = u1 * torch.cos(alpha) - u2 * torch.sin(alpha)
    yp = u1 * torch.sin(alpha) + u2 * torch.cos(alpha)

    # geometry groups
    gg = {}  # noqa

    ##################
    # vertical inner #
    ##################
    gg["x_lvi"] = torch.stack((xp, xp + torch.sin(alpha) * extend), dim=1)
    gg["y_lvi"] = torch.stack((yp, yp - torch.cos(alpha) * extend), dim=1)
    beta = torch.tensor((beta1, beta2), device=alpha.device)
    gg["x_bvi"] = torch.stack((-torch.sin(beta) * extend, torch.sin(beta) * extend), dim=1)
    gg["y_bvi"] = torch.stack((torch.cos(beta) * extend, -torch.cos(beta) * extend), dim=1)

    ##################
    # vertical outer #
    ##################
    xq = xp + sqrt2 * torch.cos(alpha + torch.pi / 4)
    yq = yp + sqrt2 * torch.sin(alpha + torch.pi / 4)
    gg["x_lvo"] = torch.stack((xq, xq + torch.sin(alpha) * extend), dim=1)
    gg["y_lvo"] = torch.stack((yq, yq - torch.cos(alpha) * extend), dim=1)
    gg["x_bvo"] = torch.stack((torch.cos(beta) - torch.sin(beta) * extend,
                               torch.cos(beta) + torch.sin(beta) * extend), dim=1)
    gg["y_bvo"] = torch.stack((torch.sin(beta) + torch.cos(beta) * extend,
                               torch.sin(beta) - torch.cos(beta) * extend), dim=1)

    ####################
    # horizontal inner #
    ####################
    gg["x_lhi"] = torch.stack((xp, xp - torch.cos(alpha) * extend), dim=1)
    gg["y_lhi"] = torch.stack((yp, yp - torch.sin(alpha) * extend), dim=1)

    ####################
    # horizontal outer #
    ####################
    gg["x_lho"] = torch.stack((xq, xq - torch.cos(alpha) * extend), dim=1)
    gg["y_lho"] = torch.stack((yq, yq - torch.sin(alpha) * extend), dim=1)

    # area sample
    x_sample = torch.linspace(0., 1., n_areas, device=xp.device)
    x_min, x_max = -5., 2.
    x_sample = x_min + x_sample * (x_max - x_min)

    # lower edge
    interp_lvi = interp1d_multi_lines(gg[f"x_lvi"], gg[f"y_lvi"], x_sample, min_for_reduce=False)
    interp_lhi = interp1d_multi_lines(gg[f"x_lhi"], gg[f"y_lhi"], x_sample, min_for_reduce=False)
    interp_bvi = interp1d_multi_lines(gg[f"x_bvi"], gg[f"y_bvi"], x_sample, min_for_reduce=True)
    y_sample_lower = torch.maximum(torch.maximum(interp_lvi, interp_lhi), interp_bvi)
    y_sample_lower = y_sample_lower.clamp(0., 1.)

    # upper edge
    interp_lvo = interp1d_multi_lines(gg[f"x_lvo"], gg[f"y_lvo"], x_sample, min_for_reduce=True)
    interp_lho = interp1d_multi_lines(gg[f"x_lho"], gg[f"y_lho"], x_sample, min_for_reduce=True)
    interp_bvo = interp1d_multi_lines(gg[f"x_bvo"], gg[f"y_bvo"], x_sample, min_for_reduce=False)
    y_sample_upper = torch.minimum(torch.minimum(interp_lvo, interp_lho), interp_bvo)
    y_sample_upper = y_sample_upper.clamp(0., 1.)

    # area
    height = torch.clamp(y_sample_upper - y_sample_lower, min=0., max=None)
    area = (height * (x_sample[1] - x_sample[0])).sum()
    if not return_geometry:
        return area

    # update geometry
    gg |= {"alpha": alpha, "beta1": beta1, "beta2": beta2, "u1": u1, "u2": u2,
           "xp": xp, "yp": yp,
           "xq": xq, "yq": yq,
           "x_sample": x_sample,
           "y_sample_lower": y_sample_lower,
           "y_sample_upper": y_sample_upper,
           "area": area}
    return gg
