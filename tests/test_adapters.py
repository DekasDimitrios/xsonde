import types

import torch

from xsonde.adapters import AuroraAdapter


def test_aurora_adapter_hands_out_the_input_grid():
    ad = AuroraAdapter(model=types.SimpleNamespace(patch_size=4), device="cpu")
    md = types.SimpleNamespace(lat=torch.linspace(90, -90, 181), lon=torch.arange(0, 360, 1.0))
    batch = types.SimpleNamespace(metadata=md, static_vars={"lsm": torch.ones(181, 360)},
                                  surf_vars={}, atmos_vars={})
    assert ad.lat(batch).shape == (181,)
    assert ad.lon(batch).shape == (360,)
    assert ad.land_mask(batch).shape == (181, 360)
    md = types.SimpleNamespace(lat=torch.linspace(89.5, -89.5, 180), lon=torch.arange(0, 360, 1.0))
    batch = types.SimpleNamespace(metadata=md, static_vars={}, surf_vars={}, atmos_vars={})
    assert ad.land_mask(batch) is None
