import torch

from xsonde.reductions import reduce


def test_global_mean_and_land_mean():
    lat = torch.linspace(89.0, -89.0, 36)
    x = torch.ones(1, 1, 36, 72)
    assert torch.isclose(reduce("global_mean", x, lat), torch.tensor(1.0))
    mask = torch.zeros(36, 72)
    mask[:18] = 1.0
    x[:, :, :18] = 3.0
    assert torch.isclose(reduce("land_mean", x, lat, mask=mask), torch.tensor(3.0))
    assert torch.isclose(reduce("ocean_mean", x, lat, mask=mask), torch.tensor(1.0))


def test_responses_vanish_at_baseline():
    lat = torch.linspace(89.0, -89.0, 36)
    x = torch.rand(1, 1, 4, 36, 72)
    assert reduce("rmse_response", x, lat, baseline=x) == 0
    assert reduce("mean_abs_response_level", x, lat, baseline=x).shape == (4,)
    assert reduce("global_mean_level", x, lat).shape == (4,)
