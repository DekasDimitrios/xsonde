import os

import numpy as np
import yaml

from xsonde import results as results_io
from xsonde.methods import METHODS
from xsonde.runner import expand_sweep, run_sweep
from tests.toy import DATE, REDUCTIONS, adapter, load_batch


def test_expand_sweep_pairs_and_axes():
    cfg = {"method": {"name": "pci", "source": "q"}, "data": {},
           "sweep": {"pairs": [{"method.source": "q"}, {"method.source": "zwd"}],
                     "data.start_time": ["2014-01-15T00:00:00", "2016-01-15T00:00:00"]}}
    runs = expand_sweep(cfg)
    assert len(runs) == 4
    assert {r["method"]["source"] for r in runs} == {"q", "zwd"}


def test_results_roundtrip(tmp_path):
    probe = METHODS["occlusion"]({"source": "q", "targets": ["tp", "q"], "block": 12,
                                  "reductions": REDUCTIONS})
    res = probe.run(adapter(), load_batch(DATE))
    out = str(tmp_path / "run")
    results_io.save(res, out)
    back = results_io.load(out)
    assert set(back.fields) == {"tp", "q"}
    assert np.allclose(back.fields["tp"], res.fields["tp"], equal_nan=True)
    assert back.results["tp"]["mapLabels"] == res.results["tp"]["mapLabels"]
    assert back.baseline_results["tp"]["box"] == res.baseline_results["tp"]["box"]
    assert np.allclose(back.lat, res.lat)
    assert back.meta["block"] == 12


def test_run_sweep_writes_done_markers_and_skips(tmp_path):
    cfg = {
        "adapter": {"name": "dict", "model": "tests.toy:toy_model", "loader": "tests.toy:load_batch"},
        "data": {"start_time": "2014-01-15T00:00:00"},
        "method": {"name": "ceteris_paribus", "source": "q", "targets": ["tp"], "mode": "dist",
                   "perturbations": [-1, 0, 1], "reductions": REDUCTIONS},
        "sweep": {"method.source": ["q", "zwd"], "data.start_time": ["2014-01-15T00:00:00", "2016-01-15T00:00:00"]},
    }
    path = tmp_path / "enso.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    finished = run_sweep(str(path))
    assert len(finished) == 4
    for out in finished:
        assert os.path.exists(os.path.join(out, "done"))
        assert os.path.exists(os.path.join(out, "resolved_config.json"))
        assert results_io.load(out).fields["tp"].shape[0] == 3
    assert run_sweep(str(path)) == []
    assert len(run_sweep(str(path), force=True, shard=1, nshards=2)) == 2
