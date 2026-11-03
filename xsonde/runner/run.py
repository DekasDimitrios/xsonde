import os
import json
import dataclasses
from itertools import groupby

import torch
import yaml

from xsonde import results as results_io
from xsonde.adapters import build_adapter
from xsonde.config import as_datetime, load_sweep, method_params
from xsonde.methods import METHODS
from xsonde.runner.sweep import expand_sweep


def run_sweep(config_path, force=False, shard=0, nshards=1, adapter=None, tf32=True):
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    raw = load_sweep(config_path)
    exp_base = os.path.dirname(os.path.abspath(config_path))
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    cfgs = expand_sweep(raw)[shard::nshards]
    for cfg in cfgs:
        cfg["data"]["start_time"] = as_datetime(cfg["data"]["start_time"])

    if adapter is None:
        adapter = build_adapter(raw["adapter"])
    adapter.load_model()

    finished = []
    cfgs.sort(key=lambda c: c["data"]["start_time"])
    for start_time, group in groupby(cfgs, key=lambda c: c["data"]["start_time"]):
        batch = None
        donors = {}
        for cfg_dict in group:
            method = cfg_dict["method"]
            probe = METHODS[method["name"]](method_params(method))
            out = os.path.join(exp_base, probe.name, probe.run_subpath(config_name, start_time))
            done = os.path.join(out, "done")
            if not force and os.path.exists(done):
                print(f"{out} already done, skipping", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "config.yaml"), "w") as f:
                yaml.safe_dump(cfg_dict, f)
            if batch is None:
                batch = adapter.load_batch(start_time)
                print(f"batch for {start_time} loaded", flush=True)
            for donor_date in getattr(probe.cfg, "donor_dates", None) or []:
                donor_date = as_datetime(donor_date)
                if donor_date not in donors:
                    donors[donor_date] = adapter.load_batch(donor_date)
                    print(f"donor batch for {donor_date} loaded", flush=True)
            donor_lookup = {d: b for d, b in donors.items()}
            donor_lookup.update({str(d): b for d, b in donors.items()})

            result = probe.run(adapter, batch, donors=donor_lookup)
            results_io.save(result, out)
            with open(os.path.join(out, "resolved_config.json"), "w") as f:
                json.dump({"adapter": raw.get("adapter"), "data": cfg_dict.get("data"),
                           "method": dataclasses.asdict(probe.cfg)}, f, indent=2, default=str)
            open(done, "w").close()
            finished.append(out)
            print(f"results for {out} saved", flush=True)
    return finished
