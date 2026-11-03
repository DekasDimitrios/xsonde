from datetime import datetime, timedelta

import numpy as np
import torch

from xsonde.methods import METHODS
from tests.toy import DATE, H, L, REDUCTIONS, W, adapter, exp_tp, load_batch, log_tp


def test_ceteris_paribus_dist_is_monotonic():
    ad = adapter()
    batch = load_batch(DATE)
    probe = METHODS["ceteris_paribus"]({"source": "q", "targets": ["tp", "q"], "mode": "dist",
                                        "perturbations": [-1, 0, 1], "reductions": REDUCTIONS})
    res = probe.run(ad, batch)
    gm = res.results["tp"]["globalMean"]
    assert gm[0] < gm[1] < gm[2]
    assert res.results["tp"]["rmseResp"][1] == 0
    assert res.results["tp"]["rmseResp"][0] > 0
    assert res.fields["tp"].shape == (3, H, W)
    assert res.fields["q"].shape == (3, H, W)
    assert np.shape(res.results["q"]["globalMeanLevel"]) == (3, L)
    assert probe.run_subpath("enso", DATE) == f"dist/variables/enso/2014_01_15T00/q"


def test_ceteris_paribus_sigma_shift_acts_in_model_space():
    ad = adapter()
    batch = load_batch(DATE)
    tp = batch["vars"]["tp"]
    probe = METHODS["ceteris_paribus"]({"source": "tp", "targets": ["tp"], "mode": "dist",
                                        "perturbations": [0, 1], "reductions": REDUCTIONS})
    res = probe.run(ad, batch)
    shifted = exp_tp(log_tp(tp) + 2.0)
    expected = ad.predict(ad.replace(batch, "tp", shifted))["tp"].squeeze().numpy()
    assert np.allclose(res.fields["tp"][1], expected)
    assert np.allclose(res.fields["tp"][0], ad.predict(batch)["tp"].squeeze().numpy())
    assert not np.allclose(res.fields["tp"][1], ad.predict(ad.replace(batch, "tp", tp + 2.0))["tp"].squeeze().numpy())


def test_ceteris_paribus_region_stays_local():
    ad = adapter()
    batch = load_batch(DATE)
    probe = METHODS["ceteris_paribus"]({"source": "zwd", "targets": ["tp"], "mode": "absolute",
                                        "perturbations": [0, 50], "reductions": REDUCTIONS,
                                        "region": [0, 8, 0, 16]})
    res = probe.run(ad, batch)
    diff = np.abs(res.fields["tp"][1] - res.fields["tp"][0])
    assert diff[:8, :16].max() > 0
    assert diff[12:, :].max() == 0 and diff[:, 20:].max() == 0
    assert "zwd_region_0_8_0_16" in probe.run_subpath("x", DATE)


def test_ceteris_paribus_rejects_level_on_surface():
    probe = METHODS["ceteris_paribus"]({"source": "zwd", "targets": ["tp"], "mode": "dist",
                                        "perturbations": [1], "reductions": REDUCTIONS, "level": 850})
    try:
        probe.run(adapter(), load_batch(DATE))
    except ValueError as e:
        assert "no vertical levels" in str(e)
    else:
        raise AssertionError("expected a ValueError")


def test_pci_spatial():
    ad = adapter()
    batch = load_batch(DATE)
    probe = METHODS["pci"]({"source": "q", "targets": ["tp", "zwd"], "scheme": "spatial",
                            "repeats": 3, "reductions": REDUCTIONS})
    res = probe.run(ad, batch)
    assert len(res.results["tp"]["rmseResp"]) == 3
    assert all(v > 0 for v in res.results["tp"]["rmseResp"])
    assert res.baseline_results["tp"]["rmseResp"] == 0
    assert res.baseline_fields["tp"].shape == (H, W)
    assert res.fields["zwd"].shape == (3, H, W)


def test_pci_temporal_uses_donors():
    ad = adapter()
    batch = load_batch(DATE)
    donor_dates = [DATE + timedelta(days=365), DATE + timedelta(days=730)]
    donors = {d: load_batch(d) for d in donor_dates}
    probe = METHODS["pci"]({"source": "zwd", "targets": ["tp"], "scheme": "temporal",
                            "donor_dates": donor_dates, "reductions": REDUCTIONS, "perm_time": "t1"})
    res = probe.run(ad, batch, donors=donors)
    assert len(res.results["tp"]["globalMean"]) == 2
    swapped = ad.replace(batch, "zwd", torch.cat([batch["vars"]["zwd"][:, :1], donors[donor_dates[0]]["vars"]["zwd"][:, 1:]], dim=1))
    expected = ad.predict(swapped)["tp"].squeeze().numpy()
    assert np.allclose(res.fields["tp"][0], expected)
    assert probe.run_subpath("x", DATE).endswith("zwd_t1")


def test_occlusion_finds_the_block_under_the_box():
    ad = adapter()
    batch = load_batch(DATE)
    batch["vars"]["q"][0, -1, -1, 8:12, 20:24] += 0.05
    probe = METHODS["occlusion"]({"source": "q", "targets": ["tp"], "block": 4, "fill": "mean",
                                  "reductions": REDUCTIONS})
    res = probe.run(ad, batch)
    labels = res.results["tp"]["mapLabels"]
    assert labels == [r["label"] for r in REDUCTIONS]
    assert res.fields["tp"].shape == (len(labels), H // 4, W // 4)
    assert len(res.lat) == H // 4 and len(res.lon) == W // 4
    grid = res.fields["tp"][labels.index("box")] - res.baseline_results["tp"]["box"]
    i, j = np.unravel_index(np.nanargmax(np.abs(grid)), grid.shape)
    assert (i, j) == (2, 5)


def test_occlusion_land_only_skips_sea_blocks():
    ad = adapter()
    batch = load_batch(DATE)
    probe = METHODS["occlusion"]({"source": "zwd", "targets": ["tp"], "block": 12,
                                  "reductions": REDUCTIONS, "land_only": True})
    res = probe.run(ad, batch)
    lsm = batch["static"]["lsm"]
    for (i, j) in res.results["tp"]["block"]:
        assert lsm[i:i + 12, j:j + 12].max() > 0.5


def test_rise_shapes():
    ad = adapter()
    batch = load_batch(DATE)
    probe = METHODS["rise"]({"source": "q", "targets": ["tp"], "masks": 20, "mask_h": 9, "mask_w": 18,
                             "reductions": REDUCTIONS, "level": 1000})
    res = probe.run(ad, batch)
    labels = res.results["tp"]["mapLabels"]
    assert res.fields["tp"].shape == (len(labels), 9, 18)
    assert 0.2 < res.results["tp"]["pHat"] < 0.8
    assert np.shape(res.results["tp"]["maskMean"]) == (9, 18)
    assert len(res.results["tp"]["rmseResp"]) == 20
    assert "q_level_1000" in probe.run_subpath("x", DATE)
