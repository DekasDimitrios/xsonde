from datetime import timedelta

import numpy as np
import torch

from xsonde.adapters import DictAdapter
from xsonde.methods import METHODS
from tests.toy import DATE, L, LEVELS, REDUCTIONS, W, toy_model

H_IN = 37


def cropping_model(v):
    out = toy_model(v)
    return {k: t[..., :-1, :] for k, t in out.items()}


def batch_721_like(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "vars": {"q": torch.rand(1, 2, L, H_IN, W, generator=g) * 0.01,
                 "zwd": torch.rand(1, 2, H_IN, W, generator=g) * 100.0,
                 "tp": torch.rand(1, 2, H_IN, W, generator=g) * 0.001},
        "static": {"lsm": (torch.rand(H_IN, W, generator=g) > 0.6).float()},
        "lat": torch.linspace(90.0, -90.0, H_IN),
        "lon": torch.linspace(0.0, 355.0, W),
        "levels": LEVELS,
        "sigma": {"q": torch.tensor([0.001, 0.002, 0.004, 0.005]), "zwd": 30.0, "tp": 2.0},
    }


def adapter():
    return DictAdapter(model=cropping_model, loader=batch_721_like)


def test_ceteris_paribus_with_land_only_on_a_cropping_model():
    probe = METHODS["ceteris_paribus"]({"source": "zwd", "targets": ["tp"], "mode": "absolute",
                                        "perturbations": [0, 20], "reductions": REDUCTIONS,
                                        "land_only": True, "region": [0, 20, 0, 40]})
    res = probe.run(adapter(), batch_721_like())
    assert res.fields["tp"].shape == (2, H_IN - 1, W)
    assert len(res.lat) == H_IN - 1
    assert res.results["tp"]["landMean"][1] != res.results["tp"]["landMean"][0]


def test_pci_land_only_on_a_cropping_model():
    probe = METHODS["pci"]({"source": "q", "targets": ["tp"], "scheme": "spatial", "repeats": 2,
                            "reductions": REDUCTIONS, "land_only": True})
    res = probe.run(adapter(), batch_721_like())
    assert res.fields["tp"].shape == (2, H_IN - 1, W)
    assert res.baseline_fields["tp"].shape == (H_IN - 1, W)


def test_pci_temporal_accepts_string_donor_dates():
    donor = DATE + timedelta(days=365)
    probe = METHODS["pci"]({"source": "zwd", "targets": ["tp"], "scheme": "temporal",
                            "donor_dates": [donor.isoformat()], "reductions": REDUCTIONS})
    res = probe.run(adapter(), batch_721_like(), donors={donor: batch_721_like(1)})
    assert len(res.results["tp"]["globalMean"]) == 1


def test_occlusion_skips_partial_windows():
    probe = METHODS["occlusion"]({"source": "q", "targets": ["tp"], "block": 4,
                                  "reductions": REDUCTIONS, "land_only": True})
    res = probe.run(adapter(), batch_721_like())
    labels = res.results["tp"]["mapLabels"]
    assert res.fields["tp"].shape == (len(labels), H_IN // 4, W // 4)
    assert all(i + 4 <= H_IN for i, _ in res.results["tp"]["block"])
    assert len(res.lat) == H_IN // 4


def test_rise_on_a_cropping_model():
    probe = METHODS["rise"]({"source": "q", "targets": ["tp"], "masks": 4, "mask_h": 9, "mask_w": 18,
                             "reductions": REDUCTIONS, "land_only": True})
    res = probe.run(adapter(), batch_721_like())
    labels = res.results["tp"]["mapLabels"]
    assert res.fields["tp"].shape == (len(labels), 9, 18)
    assert len(res.lat) == 9 and len(res.lon) == 18
    assert np.isfinite(res.fields["tp"]).all()
