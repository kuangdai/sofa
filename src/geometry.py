import torch


def denominator(x):
    eps = torch.finfo(x.dtype).eps
    return torch.where(torch.less(x.abs(), eps), eps, x)


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
    y1 = ya + (yb - ya) / denominator(xb - xa) * (x1 - xa)
    y1 = torch.where(torch.isnan(y1), fill_value, y1)

    # mask those out of range
    y1 = torch.where(torch.logical_or(torch.greater(x1, xb),
                                      torch.less(x1, xa)), fill_value, y1)
    return y1


def reduce_fill_op(x, min_for_reduce):
    if min_for_reduce:
        fill_value = torch.finfo(x.dtype).max
        minmax_op = torch.min
    else:
        fill_value = torch.finfo(x.dtype).min
        minmax_op = torch.max
    return fill_value, minmax_op


def interp1d_multi_section_curve(xs, ys, x_target, min_for_reduce):
    # operator for reduction
    fill_value, minmax_op = reduce_fill_op(x_target, min_for_reduce)

    # split input
    if xs.ndim == 1:
        xs, ys = split_curve(xs, ys)

    # loop over sections
    # TODO: this may be vectorised if we pad sections to same length; still, hard to implement
    out = torch.empty((len(xs), len(x_target)), device=x_target.device)
    for j, (x, y) in enumerate(zip(xs, ys)):
        out[j] = interp1d_sorted(x, y, x_target, fill_value)

    # reduction
    return minmax_op(out, dim=0)[0]


def interp1d_multi_lines(xs, ys, x_target, min_for_reduce):
    # operator for reduction
    fill_value, minmax_op = reduce_fill_op(x_target, min_for_reduce)

    # linear interpolation
    k = (ys[:, 1] - ys[:, 0]) / denominator(xs[:, 1] - xs[:, 0])
    y_target = ys[:, 0, None] + k[:, None] * (x_target[None, :] - xs[:, 0, None])

    # out of range
    out_of_range = torch.logical_or(torch.greater(x_target[None, :], xs[:, 1, None]),
                                    torch.less(x_target[None, :], xs[:, 0, None]))
    y_target = torch.where(out_of_range, fill_value, y_target)

    # reduction
    return minmax_op(y_target, dim=0)[0]


def compute_area(t, alpha, xp, yp, dt_alpha, dt_xp, dt_yp,
                 bound=20., n_areas=2000, envelope=False, return_geometry=False):
    # constants
    extend = bound * 4.
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
    x_sample = torch.linspace(0., 1., n_areas, device=xp.device)
    x_min, x_max = xp[-1] - 1., 1.
    x_sample = x_min + x_sample * (x_max - x_min)

    # lower edge
    y_sample_lower = interp1d_multi_section_curve(xp, yp, x_sample, min_for_reduce=False)
    for key in ["lvi", "lhi"]:
        interp = interp1d_multi_lines(gg[f"x_{key}"], gg[f"y_{key}"], x_sample, min_for_reduce=False)
        y_sample_lower = torch.maximum(y_sample_lower, interp)
    if envelope:
        for key in ["evi", "ehi"]:
            interp = interp1d_multi_section_curve(gg[f"x_{key}"], gg[f"y_{key}"], x_sample, min_for_reduce=False)
            y_sample_lower = torch.maximum(y_sample_lower, interp)
    y_sample_lower = y_sample_lower.clamp(0., 1.)

    # upper edge
    y_sample_upper = interp1d_multi_section_curve(xq, yq, x_sample, min_for_reduce=True)
    for key in ["lvo", "lho"]:
        interp = interp1d_multi_lines(gg[f"x_{key}"], gg[f"y_{key}"], x_sample, min_for_reduce=True)
        y_sample_upper = torch.minimum(y_sample_upper, interp)
    if envelope:
        for key in ["evo", "eho"]:
            interp = interp1d_multi_section_curve(gg[f"x_{key}"], gg[f"y_{key}"], x_sample, min_for_reduce=True)
            y_sample_upper = torch.minimum(y_sample_upper, interp)
    y_sample_upper = y_sample_upper.clamp(0., 1.)

    # area
    height = torch.clamp(y_sample_upper - y_sample_lower, min=0., max=None)
    area = (height * (x_sample[1] - x_sample[0])).sum()
    if not return_geometry:
        return area

    # update geometry
    gg |= {"t": t,
           "alpha": alpha, "xp": xp, "yp": yp,
           "dt_alpha": dt_alpha, "dt_xp": dt_xp, "dt_yp": dt_yp,
           "xq": xq, "yq": yq,
           "x_sample": x_sample,
           "y_sample_lower": y_sample_lower,
           "y_sample_upper": y_sample_upper}
    return area, gg
