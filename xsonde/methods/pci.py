import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

from xsonde import operators
from xsonde.core import Probe, Readout, Result, coerce_reductions, scope_label, source_label, validate_region


@dataclass
class PciConfig:
    source: str
    targets: list
    reductions: list
    scheme: str = "spatial"
    repeats: int = 5
    donor_dates: list | None = None
    level: int | None = None
    region: list | None = None
    perm_time: str = "both"
    land_only: bool = False
    seed: int = 42

    def __post_init__(self):
        if self.scheme not in ("spatial", "temporal"):
            raise ValueError(f"scheme should be spatial or temporal, you provided {self.scheme!r}")
        if self.scheme == "temporal":
            if not self.donor_dates:
                raise ValueError("the temporal scheme needs donor_dates, one per repeat")
            self.repeats = len(self.donor_dates)
        elif self.donor_dates:
            raise ValueError("donor_dates only applies to the temporal scheme")
        operators.time_index(self.perm_time)
        if self.repeats < 1:
            raise ValueError(f"repeats should be at least 1, you provided {self.repeats}")
        self.region = validate_region(self.region)
        self.reductions = coerce_reductions(self.reductions)


class Pci(Probe):
    name = "pci"
    config_cls = PciConfig

    def run_subpath(self, config_name, start_time):
        cfg = self.cfg
        return os.path.join(cfg.scheme, scope_label(cfg.level, cfg.region), config_name,
                            f"{start_time:%Y_%m_%dT%H}",
                            source_label(cfg.source, cfg.level, cfg.region, cfg.land_only, cfg.perm_time))

    def run(self, adapter, batch, donors=None):
        cfg = self.cfg
        x = adapter.get(batch, cfg.source)
        if cfg.scheme == "temporal" and x.dim() < 4:
            raise ValueError(f"'{cfg.source}' does not vary by date, a temporal swap is a no-op")
        level_idx = self.level_index(adapter, batch, cfg.source)
        selector = operators.Selector(level=level_idx, region=cfg.region, land_only=cfg.land_only,
                                      time=operators.time_index(cfg.perm_time))
        land = adapter.land_mask(batch)
        readout = Readout(adapter, batch, cfg.targets, cfg.reductions)

        with torch.inference_mode():
            readout.set_baseline(adapter.predict(batch))
            baseline_results = readout.baseline_results()
            baseline_fields = readout.baseline_fields()
            results = readout.empty()
            fields = {t: [] for t in cfg.targets}
            for repeat in range(cfg.repeats):
                new = self._permuted(adapter, batch, x, repeat, selector, land, donors)
                pred = adapter.predict(adapter.replace(batch, cfg.source, new))
                for t in cfg.targets:
                    for label, value in readout.reduce(pred, t).items():
                        results[t][label].append(value)
                    fields[t].append(readout.field(pred, t))

        lat, lon = readout.coords()
        return Result(Readout.stack(results), {t: np.stack(f) for t, f in fields.items()}, lat, lon,
                      baseline_results, baseline_fields,
                      meta={"scheme": cfg.scheme, "repeats": cfg.repeats})

    def _permuted(self, adapter, batch, x, repeat, selector, land, donors):
        cfg = self.cfg
        if cfg.scheme == "temporal":
            donor_date = cfg.donor_dates[repeat]
            donor_batch = (donors or {}).get(donor_date)
            if donor_batch is None:
                raise ValueError(f"no donor batch for {donor_date}; the runner loads donors from donor_dates")
            new = operators.donor_swap(x, adapter.get(donor_batch, cfg.source))
        else:
            eligible = selector.eligible(x.shape[-2], x.shape[-1], land)
            new = operators.permute_spatial(x, cfg.seed + repeat, eligible)
        new = operators.apply(x, new, selector, land)
        if torch.equal(new, x):
            raise ValueError(f"permuting {cfg.source} left the field unchanged; check the scheme")
        return new
