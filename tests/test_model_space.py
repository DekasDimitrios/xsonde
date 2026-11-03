import numpy as np
import torch

from xsonde.adapters import DictAdapter
from xsonde.methods import METHODS
from xsonde.reductions import reduce
from tests.toy import DATE, REDUCTIONS, exp_tp, load_batch, log_tp, toy_model


def adapter(lower=None):
    return DictAdapter(model=toy_model, loader=load_batch, transforms={"tp": (log_tp, exp_tp)},
                       lower_bounds={} if lower is None else {"tp": lower})


def test_occlusion_fills_in_model_space():
    ad = adapter()
    batch = load_batch(DATE)
    tp = batch["vars"]["tp"]
    probe = METHODS["occlusion"]({"source": "tp", "targets": ["tp"], "block": 36,
                                  "reductions": REDUCTIONS, "region": [0, 36, 0, 36]})
    res = probe.run(ad, batch)
    assert res.results["tp"]["block"] == [[0, 0]]
    xm = log_tp(tp)
    xm[..., 0:36, 0:36] = xm.mean(dim=(-2, -1), keepdim=True)
    expected = ad.predict(ad.replace(batch, "tp", exp_tp(xm)))["tp"]
    want = float(reduce("global_mean", expected, batch["lat"]))
    assert np.isclose(res.results["tp"]["globalMean"][0], want)
    naive = tp.clone()
    naive[..., 0:36, 0:36] = tp.mean(dim=(-2, -1), keepdim=True)
    other = float(reduce("global_mean", ad.predict(ad.replace(batch, "tp", naive))["tp"], batch["lat"]))
    assert not np.isclose(res.results["tp"]["globalMean"][0], other)


def test_rise_blends_in_model_space():
    ad = adapter()
    batch = load_batch(DATE)
    probe = METHODS["rise"]({"source": "tp", "targets": ["tp"], "masks": 3, "mask_h": 9, "mask_w": 18,
                             "reductions": REDUCTIONS})
    res = probe.run(ad, batch)
    assert np.isfinite(res.fields["tp"]).all()
    assert (res.results["tp"]["rmseResp"][0] > 0)


def test_ceteris_paribus_clamps_to_the_adapter_lower_bound():
    batch = load_batch(DATE)
    probe = METHODS["ceteris_paribus"]({"source": "tp", "targets": ["tp"], "mode": "absolute",
                                        "perturbations": [0, -1.0], "reductions": REDUCTIONS})
    res = probe.run(adapter(lower=0.0), batch)
    assert res.meta["clamp_min"] == 0.0
    zeroed = ad_pred = adapter(lower=0.0).predict(adapter().replace(batch, "tp", torch.zeros_like(batch["vars"]["tp"])))["tp"]
    assert np.allclose(res.fields["tp"][1], zeroed.squeeze().numpy())
    free = probe.run(adapter(), batch)
    assert free.meta["clamp_min"] is None
    assert not np.allclose(free.fields["tp"][1], zeroed.squeeze().numpy())
