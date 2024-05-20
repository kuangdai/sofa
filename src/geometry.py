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


def interp1d_sorted(x0, y0, x1, left_value, right_value):
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
    y1 = torch.where(torch.less(x1, x0[0]), left_value, y1)
    y1 = torch.where(torch.greater(x1, x0[-1]), right_value, y1)
    return y1


def interp1d_multi_section(x, y, x_tar, left_value, right_value, min_split_reduce):
    # split curve
    x_splits, y_splits = split_curve(x, y)
    # loop over splits
    # TODO: this may be vectorised if we pad splits to same length; still, hard to implement
    out = torch.empty((len(x_splits), len(x_tar)), device=x_tar.device)
    for j, (x_split, y_split) in enumerate(zip(x_splits, y_splits)):
        out[j] = interp1d_sorted(x_split, y_split, x_tar, left_value, right_value)
    # reduce splits
    minmax = torch.min if min_split_reduce else torch.max
    out = minmax(out, dim=0)[0]
    return out


def compute_area(alpha, xp, yp, xp_prime, yp_prime, n_area_samples=2000, return_outline=False):
    # curve u
    xu = xp - torch.sin(alpha) * xp_prime + (torch.cos(alpha) - 1) * yp_prime
    yu = yp + (1 + torch.cos(alpha)) * xp_prime + torch.sin(alpha) * yp_prime
    # handle the case where u is a point at zero, which may cause nan
    if torch.isclose(xu, torch.zeros_like(xu)).all() and torch.isclose(yu, torch.zeros_like(yu)).all():
        xu = torch.zeros_like(xu)
        yu = torch.zeros_like(yu)

    # curve q
    sqrt2 = torch.sqrt(torch.tensor(2., device=alpha.device))
    xq = xp + sqrt2 * torch.cos(alpha / 2 + torch.pi / 4)
    yq = yp + sqrt2 * torch.sin(alpha / 2 + torch.pi / 4)

    # curve v
    xv = xu + torch.cos(alpha / 2)
    yv = yu + torch.sin(alpha / 2)

    # area sample
    x_sample = torch.linspace(0., 1., n_area_samples, device=xp.device)
    center = xp[len(alpha) // 2]
    x_sample = center + x_sample * (1. - center)

    # lower edge
    y_sample_p = interp1d_multi_section(xp, yp, x_sample, left_value=0., right_value=0., min_split_reduce=False)
    y_sample_u = interp1d_multi_section(xu, yu, x_sample, left_value=0., right_value=0., min_split_reduce=False)
    y_sample_lower = torch.maximum(y_sample_p, y_sample_u)
    y_sample_lower = torch.clamp(y_sample_lower, min=0., max=1.)

    # upper edge
    y_sample_q = interp1d_multi_section(xq, yq, x_sample, left_value=1., right_value=0., min_split_reduce=True)
    y_sample_v = interp1d_multi_section(xv, yv, x_sample, left_value=1., right_value=0., min_split_reduce=True)
    y_sample_upper = torch.minimum(y_sample_q, y_sample_v)
    y_sample_upper = torch.clamp(y_sample_upper, min=0., max=1.)

    # area
    height = torch.clamp(y_sample_upper - y_sample_lower, min=0., max=None)
    area = (height * (x_sample[1] - x_sample[0])).sum() * 2
    if not return_outline:
        return area

    # outline
    lu_idx = torch.where(torch.greater(y_sample_upper, y_sample_lower))[0]
    outline = x_sample[lu_idx].detach(), y_sample_lower[lu_idx].detach(), y_sample_upper[lu_idx].detach()
    return area, outline
