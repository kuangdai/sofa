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
    eps = torch.finfo(x0.dtype).eps
    k = (yb - ya) / (eps + (xb - xa))
    y1 = ya + k * (x1 - xa)

    # mask those out of range
    y1 = torch.where(torch.logical_or(torch.greater(x1, x0[-1] + eps),
                                      torch.less(x1, x0[0] - eps)), fill_value, y1)
    return y1


def interp1d_multi_section(x, y, x_target, min_split_reduce):
    if min_split_reduce:
        fill_value = torch.finfo(x.dtype).max
        minmax_op = torch.min
    else:
        fill_value = torch.finfo(x.dtype).min
        minmax_op = torch.max
    # split curve
    x_splits, y_splits = split_curve(x, y)
    # loop over splits
    # TODO: this may be vectorised if we pad splits to same length; still, hard to implement
    out = torch.empty((len(x_splits), len(x_target)), device=x_target.device)
    for j, (x_split, y_split) in enumerate(zip(x_splits, y_splits)):
        out[j] = interp1d_sorted(x_split, y_split, x_target, fill_value)
    # reduce splits
    out = minmax_op(out, dim=0)[0]
    return out


def compute_area(alpha, xp, yp, xp_prime, yp_prime, n_area_samples=2000, return_outline=False):
    # curve u right
    xu_r = xp - torch.sin(alpha) * xp_prime + (torch.cos(alpha) - 1) * yp_prime
    yu_r = yp + (1 + torch.cos(alpha)) * xp_prime + torch.sin(alpha) * yp_prime
    # handle the case where u is a point at zero, which may cause nan
    zero = torch.zeros_like(xu_r)
    if torch.isclose(xu_r, zero).all() and torch.isclose(yu_r, zero).all():
        xu_r = zero
        yu_r = zero

    # curve u left
    xu_l = xp + torch.sin(alpha) * xp_prime + (-torch.cos(alpha) - 1) * yp_prime
    yu_l = yp + (1 - torch.cos(alpha)) * xp_prime - torch.sin(alpha) * yp_prime
    if torch.isclose(xu_l, zero).all() and torch.isclose(yu_l, zero).all():
        xu_l = zero
        yu_l = zero

    # curve v left
    xv_r = xu_r + torch.cos(alpha / 2)
    yv_r = yu_r + torch.sin(alpha / 2)

    # curve v right
    xv_l = xu_l - torch.sin(alpha / 2)
    yv_l = yu_l + torch.cos(alpha / 2)

    # area sample
    x_sample = torch.linspace(0., 1., n_area_samples, device=xp.device)
    x_min, x_max = xv_l.min(), xv_r.max()
    x_sample = x_min + x_sample * (x_max - x_min)

    # lower edge
    y_sample_p = interp1d_multi_section(xp, yp, x_sample, min_split_reduce=False)
    y_sample_u_r = interp1d_multi_section(xu_r, yu_r, x_sample, min_split_reduce=False)
    y_sample_u_l = interp1d_multi_section(xu_l, yu_l, x_sample, min_split_reduce=False)
    y_sample_lower = torch.maximum(y_sample_p, torch.maximum(y_sample_u_r, y_sample_u_l))
    y_sample_lower = torch.clamp(y_sample_lower, min=0., max=1.)

    # upper edge
    y_sample_v_r = interp1d_multi_section(xv_r, yv_r, x_sample, min_split_reduce=True)
    y_sample_v_l = interp1d_multi_section(xv_l, yv_l, x_sample, min_split_reduce=True)
    y_sample_upper = torch.minimum(y_sample_v_r, y_sample_v_l)
    y_sample_upper = torch.clamp(y_sample_upper, min=0., max=1.)

    # area
    height = torch.clamp(y_sample_upper - y_sample_lower, min=0., max=None)
    area = (height * (x_sample[1] - x_sample[0])).sum()
    if not return_outline:
        return area

    # outline
    lu_idx = torch.where(torch.greater_equal(y_sample_upper, y_sample_lower))[0]
    outline = x_sample[lu_idx].detach(), y_sample_lower[lu_idx].detach(), y_sample_upper[lu_idx].detach()
    return area, outline
