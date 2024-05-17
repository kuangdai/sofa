import torch


def truncate_curve(x, y, boundary, keep_right):
    # boundary values
    m, n = x.shape
    m_range = torch.arange(m, device=x.device)
    x_boundary = x[m_range, boundary][:, None].expand(-1, n)
    y_boundary = y[m_range, boundary][:, None].expand(-1, n)

    # masked range
    n_range = torch.arange(n, device=x.device)[None, :].expand(m, -1)
    boundary = boundary[:, None].expand(-1, n)
    if keep_right:
        cond = torch.greater_equal(n_range, boundary)
    else:
        cond = torch.less_equal(n_range, boundary)

    # mask unwanted part with boundary values
    x = torch.where(cond, x, x_boundary)
    y = torch.where(cond, y, y_boundary)
    return x, y


def interp1d(x0, y0, x1, outside_value=0., x0_descending):
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


def compute_area(alpha, ab, dab, n_area_samples=2000):
    # shape: [m, n]
    alpha = alpha[None, :].expand(len(ab), -1)
    m, n = alpha.shape
    a, b = ab[:, 0], ab[:, 1]
    da, db = dab[:, 0], dab[:, 1]

    # curve p
    xp = a * (torch.cos(alpha) - 1)
    yp = b * torch.sin(alpha)

    # curve u
    c = a - b
    dc = da - db
    xu = 2 * torch.sin(alpha / 2) ** 2 * (torch.cos(alpha) * c + torch.sin(alpha) * dc)
    yu = -4 * torch.cos(alpha / 2) ** 2 * torch.sin(alpha / 2) * (torch.cos(alpha / 2) * c + torch.sin(alpha / 2) * dc)

    # curve v
    xv = xu + torch.cos(alpha / 2)
    yv = yu + torch.sin(alpha / 2)

    # outer branch of u
    corner = torch.argmax(yu, dim=1)
    xu, yu = truncate_curve(xu, yu, corner, keep_right=True)

    # two branches of v
    corners = []
    for i in range(m):
        corner_1st = torch.where(xv[i, 1:] - xv[i, :-1] > 0)[0]
        corner_1st = corner_1st[0] if len(corner_1st) > 0 else n - 1
        corner_2nd = torch.argmax(xv[corner_1st:])
        corners.append([corner_1st, corner_1st + corner_2nd])
    corners = torch.tensor(corners, device=alpha.device)
    xv1, yv1 = truncate_curve(xv, yv, corners[:, 0], keep_right=False)
    xv2, yv2 = truncate_curve(xv, yv, corners[:, 1], keep_right=True)

    # bottom
    x_sample = torch.linspace(0, 1., n_area_samples, device=alpha.device)[None, :].expand(m, -1)
    center = a[:, n // 2]
    x_sample = center[:, None] + x_sample * (1. - center[:, None])
    y_sample_p = interp1d(xp, yp, x_sample, x0_descending=True)
    y_sample_u = interp1d(xu, yu, x_sample, x0_descending=False)
    y_sample_lower = torch.maximum(y_sample_p, y_sample_u)

    # top
    y_sample_v1 = interp1d(xv1, yv1, x_sample, outside_value=1., x0_descending=True)
    y_sample_v2 = interp1d(xv2, yv2, x_sample, outside_value=1., x0_descending=True)
    y_sample_upper = torch.minimum(y_sample_v1, y_sample_v2)

    # area
    height = torch.clip(y_sample_upper - y_sample_lower, min=0, max=None)
    area = (height * (x_sample[:, 1] - x_sample[:, 0])[:, None]).sum()
    return area
