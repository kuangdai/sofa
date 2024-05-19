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


def interp1d(x0, y0, x1, outside_value=0.):
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
    eps = torch.finfo(x0.dtype).eps
    k = (yb - ya) / (eps + (xb - xa))
    y1 = ya + k * (x1 - xa)

    # mask those out of range
    y1 = torch.where(torch.logical_and(torch.greater_equal(x1, xa),
                                       torch.less_equal(x1, xb)), y1, outside_value)
    return y1


def interp1d_multi_curve(xs, ys, x_targets, outside_value, min_split_reduce):
    minmax = torch.min if min_split_reduce else torch.max
    # loop over curves
    # TODO: this seems hard to vectorise because number of splits are unequal
    out = torch.empty_like(x_targets)
    for i, (x, y, x_tar) in enumerate(zip(xs, ys, x_targets)):
        # split curve
        x_splits, y_splits = split_curve(x, y)
        # loop over splits
        # TODO: this may be vectorised if we pad splits to same length; still, hard to implement
        out_i = torch.empty((len(x_splits), len(x_tar)), device=x_tar.device)
        for j, (x_split, y_split) in enumerate(zip(x_splits, y_splits)):
            out_i[j] = interp1d(x_split, y_split, x_tar, outside_value=outside_value)
        out[i] = minmax(out_i, dim=0)[0]
    return out


def compute_area(alpha, xp, yp, xp_prime, yp_prime, n_area_samples=2000, return_outline=False):
    # curve u and v
    m, n = xp.shape
    alpha = alpha[None, :].expand(m, -1)
    xu = xp - torch.sin(alpha) * xp_prime + (torch.cos(alpha) - 1) * yp_prime
    yu = yp + (1 + torch.cos(alpha)) * xp_prime + torch.sin(alpha) * yp_prime
    # handle the case where u is a point at zero
    if torch.isclose(xu, torch.zeros_like(xu)).all() and torch.isclose(yu, torch.zeros_like(yu)).all():
        xu[:] = 0.
        yu[:] = 0.
    xv = xu + torch.cos(alpha / 2)
    yv = yu + torch.sin(alpha / 2)

    # area sample
    x_sample = torch.linspace(0, 1., n_area_samples,
                              device=xp.device)[None, :].expand(m, -1)
    center = xp[:, n // 2][:, None]
    x_sample = center + x_sample * (1. - center)

    # lower edge
    y_sample_p = interp1d_multi_curve(xp, yp, x_sample, outside_value=0., min_split_reduce=False)
    y_sample_u = interp1d_multi_curve(xu, yu, x_sample, outside_value=0., min_split_reduce=False)
    y_sample_lower = torch.maximum(y_sample_p, y_sample_u)
    y_sample_lower = torch.clamp(y_sample_lower, min=0., max=1.)

    # upper edge
    y_sample_upper = interp1d_multi_curve(xv, yv, x_sample, outside_value=1., min_split_reduce=True)
    y_sample_upper = torch.clamp(y_sample_upper, min=0., max=1.)

    # area
    height = torch.clamp(y_sample_upper - y_sample_lower, min=0., max=None)
    area = (height * (x_sample[:, 1] - x_sample[:, 0])[:, None]).sum(dim=1) * 2
    if not return_outline:
        return area

    # outline
    outlines = []
    for i in range(m):
        lu_idx = torch.where(torch.greater(y_sample_upper[i], y_sample_lower[i]))[0]
        outlines.append((x_sample[i, lu_idx].detach(),
                         y_sample_lower[i, lu_idx].detach(),
                         y_sample_upper[i, lu_idx].detach()))
    return area, outlines
