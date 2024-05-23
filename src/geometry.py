import torch


def split_curve(x, y):
    assert len(x) >= 3

    # location of saddle
    x0 = x[0:-2]
    x1 = x[1:-1]
    x2 = x[2:]
    saddle = torch.where((x2 - x1) * (x1 - x0) < 0)[0] + 1
    saddle = torch.cat((torch.tensor([0], device=x.device),
                        saddle,
                        torch.tensor([len(x)], device=x.device)))

    # split: [..., saddle - 1], [saddle, ...]
    split = list(torch.diff(saddle))
    x_splits = list(torch.split(x, split))
    y_splits = list(torch.split(y, split))

    # append saddle to left: [..., saddle - 1, saddle], [saddle, ...]
    for i in range(len(x_splits) - 1):
        x_splits[i] = torch.cat((x_splits[i], x_splits[i + 1][:1]))
        y_splits[i] = torch.cat((y_splits[i], y_splits[i + 1][:1]))
    return x_splits, y_splits


def denominator(x):
    eps = torch.finfo(x.dtype).eps
    return torch.where(torch.less(x.abs(), eps), eps, x)


def interp1d_sorted(x0, y0, x1, fill_value):
    assert len(x0) >= 2

    # flip if x0 is descending
    if x0[-1] < x0[0]:
        x0 = x0.flip(dims=[0])
        y0 = y0.flip(dims=[0])

    # start-end values
    idx = torch.searchsorted(x0, x1)
    idx -= 1
    idx = idx.clamp(0, len(x0) - 1 - 1)
    xa, ya = x0[idx], y0[idx]
    xb, yb = x0[idx + 1], y0[idx + 1]

    # linear interpolation
    k = (yb - ya) / denominator(xb - xa)
    y1 = ya + k * (x1 - xa)

    # mask those out of range
    eps = torch.finfo(x0.dtype).eps
    y1 = torch.where(torch.isnan(y1), fill_value, y1)
    y1 = torch.where(torch.logical_or(torch.greater(x1, x0[-1] + eps),
                                      torch.less(x1, x0[0] - eps)), fill_value, y1)
    return y1


def interp1d_multi_section(xs, ys, x_target, min_for_reduce):
    # operator for section reduction
    if min_for_reduce:
        fill_value = torch.finfo(x_target.dtype).max
        minmax_op = torch.min
    else:
        fill_value = torch.finfo(x_target.dtype).min
        minmax_op = torch.max

    # split input if it is one curve
    if xs.ndim == 1:
        xs, ys = split_curve(xs, ys)

    # loop over sections
    # TODO: this may be vectorised if we pad sections to same length; still, hard to implement
    out = torch.empty((len(xs), len(x_target)), device=x_target.device)
    for j, (x, y) in enumerate(zip(xs, ys)):
        out[j] = interp1d_sorted(x, y, x_target, fill_value)

    # section reduction
    return minmax_op(out, dim=0)[0]


def compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp,
                 bound=20., n_area_samples=2000, return_outline=False):
    # constants
    eps = torch.finfo(t.dtype).eps
    extend = bound * 3.
    sqrt2 = torch.sqrt(torch.tensor(2., device=t.device))

    # geometry groups
    gg = {}  # noqa

    ##################
    # vertical inner #
    ##################
    gg["x_lvi"] = torch.stack((xp, xp + torch.sin(alpha) * extend), dim=1)
    gg["y_lvi"] = torch.stack((yp, yp - torch.cos(alpha) * extend), dim=1)
    temp = (torch.cos(alpha) * dt_xp + torch.sin(alpha) * dt_yp) / denominator(dt_alpha)
    gg["x_evi"] = xp - torch.sin(alpha) * temp
    gg["y_evi"] = yp + torch.cos(alpha) * temp

    ##################
    # vertical outer #
    ##################
    xq = xp + sqrt2 * torch.cos(alpha + torch.pi / 4)
    yq = yp + sqrt2 * torch.sin(alpha + torch.pi / 4)
    gg["x_lvo"] = torch.stack((xq, xq + torch.sin(alpha) * extend), dim=1)
    gg["y_lvo"] = torch.stack((yq, yq - torch.cos(alpha) * extend), dim=1)
    gg["x_evo"] = gg["x_evi"] + torch.cos(alpha)
    gg["y_evo"] = gg["y_evi"] + torch.sin(alpha)

    ####################
    # horizontal inner #
    ####################
    gg["x_lhi"] = torch.stack((xp, xp - torch.cos(alpha) * extend), dim=1)
    gg["y_lhi"] = torch.stack((yp, yp - torch.sin(alpha) * extend), dim=1)
    temp = (torch.sin(alpha) * dt_xp - torch.cos(alpha) * dt_yp) / denominator(dt_alpha)
    gg["x_ehi"] = xp + torch.cos(alpha) * temp
    gg["y_ehi"] = yp + torch.sin(alpha) * temp

    ####################
    # horizontal outer #
    ####################
    gg["x_lho"] = torch.stack((xq, xq - torch.cos(alpha) * extend), dim=1)
    gg["y_lho"] = torch.stack((yq, yq - torch.sin(alpha) * extend), dim=1)
    gg["x_eho"] = gg["x_ehi"] - torch.sin(alpha)
    gg["y_eho"] = gg["y_ehi"] + torch.cos(alpha)

    # area sample
    x_sample = torch.linspace(0., 1., n_area_samples, device=xp.device)
    x_min, x_max = xp[-1] - 1., 1.
    x_sample = x_min + x_sample * (x_max - x_min)

    # lower edge
    y_sample_lower = interp1d_multi_section(xp, yp, x_sample, min_for_reduce=False)
    for key in ["lvi", "lhi", "evi", "ehi"]:
        interp = interp1d_multi_section(gg[f"x_{key}"],
                                        gg[f"y_{key}"], x_sample, min_for_reduce=False)
        y_sample_lower = torch.maximum(y_sample_lower, interp)

    # upper edge
    y_sample_upper = interp1d_multi_section(xq, yq, x_sample, min_for_reduce=True)
    for key in ["lvo", "lho", "evo", "eho"]:
        interp = interp1d_multi_section(gg[f"x_{key}"],
                                        gg[f"y_{key}"], x_sample, min_for_reduce=True)
        y_sample_upper = torch.minimum(y_sample_upper, interp)

    # area
    height = torch.clamp(y_sample_upper - y_sample_lower, min=0., max=None)
    area = (height * (x_sample[1] - x_sample[0])).sum()
    if not return_outline:
        return area

    # outline
    lu_idx = torch.where(torch.greater_equal(y_sample_upper, y_sample_lower))[0]
    outline = x_sample[lu_idx].detach(), y_sample_lower[lu_idx].detach(), y_sample_upper[lu_idx].detach()
    return area, outline
