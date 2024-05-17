import torch


def truncate_curve(x, y, boundary, keep_right):
    m, n = x.shape
    x_mask = x[torch.arange(m), boundary][:, None].expand(-1, n)
    y_mask = y[torch.arange(m), boundary][:, None].expand(-1, n)
    all_idx = torch.arange(n)[None, :].expand(m, -1)
    boundary = boundary[:, None].expand(-1, n)
    if keep_right:
        cond = torch.greater_equal(all_idx, boundary)
    else:
        cond = torch.less_equal(all_idx, boundary)
    x = torch.where(cond, x, x_mask)
    y = torch.where(cond, y, y_mask)
    return x, y


def interp1d(x0, y0, x1, outside_value=0., x0_descending=False):
    if x0_descending:
        x0 = x0.flip(dims=[1])
        y0 = y0.flip(dims=[1])

    # indexing
    idx = torch.searchsorted(x0, x1)
    idx -= 1
    idx = idx.clamp(0, x0.shape[1] - 1 - 1)
    xa = torch.gather(x0, 1, idx)
    ya = torch.gather(y0, 1, idx)
    xb = torch.gather(x0, 1, idx + 1)
    yb = torch.gather(y0, 1, idx + 1)

    # linear
    eps = torch.finfo(x0.dtype).eps
    k = (yb - ya) / (eps + (xb - xa))
    y1 = ya + k * (x1 - xa)

    # mask
    mask = torch.full_like(x1, fill_value=outside_value)
    y1 = torch.where(torch.logical_and(torch.greater_equal(x1, xa),
                                       torch.less_equal(x1, xb)), y1, mask)
    return y1
