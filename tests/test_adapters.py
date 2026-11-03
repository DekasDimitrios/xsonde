import types

import torch

from xsonde.adapters import AuroraAdapter


def test_aurora_adapter_crops_to_the_patch_grid():
    ad = AuroraAdapter(model=types.SimpleNamespace(patch_size=4), device="cpu")
    md = types.SimpleNamespace(lat=torch.linspace(90, -90, 181), lon=torch.arange(0, 360, 1.0))
    batch = types.SimpleNamespace(metadata=md, static_vars={"lsm": torch.ones(181, 360)},
                                  surf_vars={}, atmos_vars={})
    assert ad.lat(batch).shape == (180,)
    assert ad.lon(batch).shape == (360,)
    assert ad.land_mask(batch).shape == (180, 360)
    assert float(ad.lat(batch)[0]) == 90.0


def test_aurora_adapter_leaves_aligned_grids_alone():
    ad = AuroraAdapter(model=types.SimpleNamespace(patch_size=4), device="cpu")
    md = types.SimpleNamespace(lat=torch.linspace(89.5, -89.5, 180), lon=torch.arange(0, 360, 1.0))
    batch = types.SimpleNamespace(metadata=md, static_vars={}, surf_vars={}, atmos_vars={})
    assert ad.lat(batch).shape == (180,)
    assert ad.land_mask(batch) is None
