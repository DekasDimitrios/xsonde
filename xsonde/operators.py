import torch

SPATIAL = (-2, -1)


def spatial_mean(x):
    return x.mean(dim=SPATIAL, keepdim=True)


def level_view(sigma, x):
    if not torch.is_tensor(sigma):
        return sigma
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    if x.dim() == 5 and sigma.dim() == 1:
        return sigma.view(1, 1, -1, 1, 1)
    return sigma


def scale(x, factor):
    return x * factor


def shift(x, amount):
    return x + amount


def sigma_shift(x, delta, sigma):
    return x + delta * level_view(sigma, x)


def noise(x, delta, sigma, seed):
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    eps = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return x + delta * level_view(sigma, x) * eps


def blend_mean(x, alpha):
    m = spatial_mean(x)
    return m + alpha * (x - m)


def fill_value(x, fill):
    if fill == "mean":
        return spatial_mean(x)
    if fill == "zero":
        return torch.zeros_like(x[..., :1, :1])
    raise ValueError(f"fill must be 'mean' or 'zero', you provided {fill!r}")


def blend_mask(x, mask):
    m = mask.to(device=x.device, dtype=x.dtype)
    return m * x + (1.0 - m) * spatial_mean(x)


def permute_spatial(x, seed, eligible=None):
    h, w = x.shape[-2], x.shape[-1]
    n = h * w
    gen = torch.Generator().manual_seed(int(seed))
    if eligible is None:
        idx = torch.randperm(n, generator=gen)
    else:
        positions = eligible.reshape(-1).cpu().nonzero(as_tuple=True)[0]
        perm = torch.randperm(positions.numel(), generator=gen)
        idx = torch.arange(n)
        idx[positions] = positions[perm]
    flat = x.reshape(*x.shape[:-2], n)
    return flat[..., idx.to(x.device)].reshape(x.shape)


def donor_swap(x, donor):
    if donor.shape != x.shape:
        raise ValueError(f"donor has shape {tuple(donor.shape)}, the source has {tuple(x.shape)}")
    return donor.to(device=x.device, dtype=x.dtype)


def interpolate(x, baseline, alpha):
    return baseline + alpha * (x - baseline)


def rise_masks(n, h, w, mask_h, mask_w, prob, seed, device):
    ch = h // mask_h
    cw = w // mask_w
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(n):
        grid = (torch.rand(1, 1, mask_h, mask_w, generator=gen) < prob).float()
        big = torch.nn.functional.interpolate(grid, size=((mask_h + 1) * ch, (mask_w + 1) * cw),
                                              mode="bilinear", align_corners=False)
        di = int(torch.randint(ch, (1,), generator=gen))
        dj = int(torch.randint(cw, (1,), generator=gen))
        yield big[0, 0, di:di + h, dj:dj + w].to(device)


class Selector:
    def __init__(self, level=None, region=None, land_only=False, time=None):
        self.level = level
        self.region = region
        self.land_only = bool(land_only)
        self.time = time

    @property
    def active(self):
        return self.level is not None or self.region is not None or self.land_only or self.time is not None

    def mask(self, x, land_mask=None):
        if not self.active:
            return None
        m = torch.ones(x.shape, dtype=torch.bool, device=x.device)
        if self.level is not None:
            if x.dim() != 5:
                raise ValueError("a level selector needs a tensor with a level axis")
            sel = torch.zeros_like(m)
            sel[:, :, self.level] = True
            m &= sel
        if self.region is not None:
            h0, h1, w0, w1 = self.region
            sel = torch.zeros_like(m)
            sel[..., h0:h1, w0:w1] = True
            m &= sel
        if self.land_only and land_mask is not None:
            m &= (land_mask.to(x.device) > 0.5)
        if self.time is not None:
            sel = torch.zeros_like(m)
            sel[:, self.time] = True
            m &= sel
        return m

    def eligible(self, h, w, land_mask=None):
        if self.region is None and not self.land_only:
            return None
        e = torch.ones(h, w, dtype=torch.bool)
        if self.region is not None:
            h0, h1, w0, w1 = self.region
            box = torch.zeros(h, w, dtype=torch.bool)
            box[h0:h1, w0:w1] = True
            e &= box
        if self.land_only and land_mask is not None:
            e &= (land_mask.detach().cpu() > 0.5).reshape(h, w)
        return e

    def allowed(self, h, w, land_mask=None, device="cpu"):
        e = self.eligible(h, w, land_mask)
        if e is None:
            return torch.ones(h, w, device=device)
        return e.to(device=device, dtype=torch.float32)


def apply(x, new, selector=None, land_mask=None):
    m = selector.mask(x, land_mask) if selector is not None else None
    if m is None:
        return new
    return torch.where(m, new, x)


def time_index(perm_time):
    if perm_time in (None, "both"):
        return None
    if perm_time == "t0":
        return 0
    if perm_time == "t1":
        return 1
    raise ValueError(f"perm_time should be one of both, t0 or t1, you provided {perm_time}")
