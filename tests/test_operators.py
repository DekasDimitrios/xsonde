import torch

from xsonde import operators
from tests.toy import load_batch


def test_sigma_shift_broadcasts_per_level():
    b = load_batch()
    q = b["vars"]["q"]
    out = operators.sigma_shift(q, 2.0, b["sigma"]["q"])
    diff = (out - q)[0, 0, :, 0, 0]
    assert torch.allclose(diff, 2.0 * b["sigma"]["q"])


def test_permute_spatial_preserves_values_and_columns():
    b = load_batch()
    q = b["vars"]["q"]
    p = operators.permute_spatial(q, seed=1)
    assert not torch.equal(p, q)
    assert torch.equal(p.flatten(-2).sort(dim=-1).values, q.flatten(-2).sort(dim=-1).values)
    idx = (p[0, 0, 0] == q[0, 0, 0, 3, 5]).nonzero()[0]
    assert torch.allclose(p[0, 1, :, idx[0], idx[1]], q[0, 1, :, 3, 5])


def test_selector_region_and_land_only():
    b = load_batch()
    zwd = b["vars"]["zwd"]
    sel = operators.Selector(region=[0, 10, 0, 20], land_only=True)
    new = operators.apply(zwd, zwd + 5.0, sel, b["static"]["lsm"])
    changed = (new != zwd)[0, 0]
    assert changed[10:].sum() == 0 and changed[:, 20:].sum() == 0
    land = b["static"]["lsm"][:10, :20] > 0.5
    assert torch.equal(changed[:10, :20], land)


def test_selector_level_and_time():
    b = load_batch()
    q = b["vars"]["q"]
    sel = operators.Selector(level=2, time=1)
    new = operators.apply(q, q * 2.0, sel)
    assert torch.equal(new[:, 0], q[:, 0])
    assert torch.equal(new[:, 1, 2], 2.0 * q[:, 1, 2])
    assert torch.equal(new[:, 1, 0], q[:, 1, 0])


def test_rise_masks_shape_and_range():
    masks = list(operators.rise_masks(5, 36, 72, 9, 18, 0.5, 0, "cpu"))
    assert len(masks) == 5
    assert all(m.shape == (36, 72) for m in masks)
    assert all(0.0 <= m.min() and m.max() <= 1.0 for m in masks)
