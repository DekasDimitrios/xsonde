import os
from dataclasses import dataclass

import numpy as np
import torch

from xsonde import operators
from xsonde.core import Probe, Readout, Result, coerce_reductions, scope_label, source_label, validate_region

MODES = {"relative": 1.0, "absolute": 0.0, "dist": 0.0, "noise": 0.0, "occlusion": 1.0}


@dataclass
class CeterisParibusConfig:
    source: str
    targets: list
    perturbations: list
    reductions: list
    mode: str = "dist"
    level: int | None = None
    region: list | None = None
    land_only: bool = False
    clamp_min: float | None = None
    seed: int = 42

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode should be one of {sorted(MODES)}, you provided {self.mode!r}")
        self.region = validate_region(self.region)
        self.reductions = coerce_reductions(self.reductions)


class CeterisParibus(Probe):
    name = "ceteris_paribus"
    config_cls = CeterisParibusConfig

    def run_subpath(self, config_name, start_time):
        cfg = self.cfg
        return os.path.join(cfg.mode, scope_label(cfg.level, cfg.region), config_name,
                            f"{start_time:%Y_%m_%dT%H}",
                            source_label(cfg.source, cfg.level, cfg.region, cfg.land_only))

    def run(self, adapter, batch, donors=None):
        cfg = self.cfg
        x = adapter.get(batch, cfg.source)
        selector = operators.Selector(level=self.level_index(adapter, batch, cfg.source),
                                      region=cfg.region, land_only=cfg.land_only)
        land = adapter.land_mask(batch)
        sigma = adapter.sigma(batch, cfg.source) if cfg.mode in ("dist", "noise") else None
        readout = Readout(adapter, batch, cfg.targets, cfg.reductions)
        base_val = MODES[cfg.mode]

        with torch.inference_mode():
            base_pred = adapter.predict(self._perturbed(adapter, batch, x, base_val, selector, land, sigma))
            readout.set_baseline(base_pred)
            results = readout.empty()
            fields = {t: [] for t in cfg.targets}
            for delta in cfg.perturbations:
                if delta == base_val:
                    pred = base_pred
                else:
                    pred = adapter.predict(self._perturbed(adapter, batch, x, delta, selector, land, sigma))
                for t in cfg.targets:
                    for label, value in readout.reduce(pred, t).items():
                        results[t][label].append(value)
                    fields[t].append(readout.field(pred, t))

        lat, lon = readout.coords()
        return Result(Readout.stack(results), {t: np.stack(f) for t, f in fields.items()}, lat, lon,
                      meta={"perturbations": list(cfg.perturbations), "mode": cfg.mode})

    def _perturbed(self, adapter, batch, x, delta, selector, land, sigma):
        cfg = self.cfg
        if cfg.mode == "relative":
            new = operators.scale(x, delta)
        elif cfg.mode == "absolute":
            new = operators.shift(x, delta)
        elif cfg.mode == "dist":
            xm = adapter.to_model_space(batch, cfg.source, x)
            new = adapter.from_model_space(batch, cfg.source, operators.sigma_shift(xm, delta, sigma))
        elif cfg.mode == "noise":
            xm = adapter.to_model_space(batch, cfg.source, x)
            new = adapter.from_model_space(batch, cfg.source, operators.noise(xm, delta, sigma, cfg.seed))
        else:
            new = operators.blend_mean(x, delta)
        if cfg.clamp_min is not None:
            new = new.clamp(min=cfg.clamp_min)
        return adapter.replace(batch, cfg.source, operators.apply(x, new, selector, land))
