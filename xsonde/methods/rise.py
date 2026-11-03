import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from xsonde import operators
from xsonde.core import Probe, Readout, Result, coerce_reductions, scope_label, source_label, validate_region


@dataclass
class RiseConfig:
    source: str
    targets: list
    reductions: list
    masks: int = 1000
    mask_h: int = 9
    mask_w: int = 18
    prob: float = 0.5
    level: int | None = None
    region: list | None = None
    land_only: bool = False
    perm_time: str = "both"
    seed: int = 42

    def __post_init__(self):
        if self.masks < 1:
            raise ValueError(f"masks should be at least 1, you provided {self.masks}")
        if not 0.0 < self.prob < 1.0:
            raise ValueError(f"prob should be in (0, 1), you provided {self.prob}")
        if self.mask_h < 1 or self.mask_w < 1:
            raise ValueError(f"mask_h and mask_w should be at least 1, you provided {self.mask_h}, {self.mask_w}")
        operators.time_index(self.perm_time)
        self.region = validate_region(self.region)
        self.reductions = coerce_reductions(self.reductions)


class Rise(Probe):
    name = "rise"
    config_cls = RiseConfig

    def run_subpath(self, config_name, start_time):
        cfg = self.cfg
        grid = f"n{cfg.masks}_{cfg.mask_h}x{cfg.mask_w}"
        return os.path.join(grid, scope_label(cfg.level, cfg.region), config_name,
                            f"{start_time:%Y_%m_%dT%H}",
                            source_label(cfg.source, cfg.level, cfg.region, cfg.land_only, cfg.perm_time))

    def run(self, adapter, batch, donors=None):
        cfg = self.cfg
        x = adapter.get(batch, cfg.source)
        level_idx = self.level_index(adapter, batch, cfg.source)
        selector = operators.Selector(level=level_idx, time=operators.time_index(cfg.perm_time))
        land = adapter.land_mask(batch)
        h, w = x.shape[-2], x.shape[-1]
        mh, mw = cfg.mask_h, cfg.mask_w
        ch, cw = h // mh, w // mw
        allowed = operators.Selector(region=cfg.region, land_only=cfg.land_only).allowed(h, w, land, x.device)
        readout = Readout(adapter, batch, cfg.targets, cfg.reductions)
        xm = adapter.to_model_space(batch, cfg.source, x)

        with torch.inference_mode():
            readout.set_baseline(adapter.predict(batch))
            baseline_results = readout.baseline_results()
            scalar = {t: {label for label, v in baseline_results[t].items() if np.isscalar(v)} for t in cfg.targets}
            acc = {t: {label: torch.zeros(mh, mw, device=x.device) for label in scalar[t]} for t in cfg.targets}
            traces = readout.empty()
            mask_acc = torch.zeros(mh, mw, device=x.device)

            t0 = time.time()
            for k, raw in enumerate(operators.rise_masks(cfg.masks, h, w, mh, mw, cfg.prob, cfg.seed, x.device)):
                m = 1.0 - (1.0 - raw) * allowed
                blended = adapter.from_model_space(batch, cfg.source, operators.blend_mask(xm, m))
                new = operators.apply(x, blended, selector, land)
                pred = adapter.predict(adapter.replace(batch, cfg.source, new))
                mc = m[:mh * ch, :mw * cw].reshape(mh, ch, mw, cw).mean(dim=(1, 3))
                mask_acc += mc
                for t in cfg.targets:
                    for label, val in readout.reduce(pred, t).items():
                        traces[t][label].append(val)
                        if label in scalar[t]:
                            acc[t][label] += float(val) * mc
                if (k + 1) % 100 == 0:
                    el = time.time() - t0
                    print(f"[rise] {k + 1}/{cfg.masks} masks, {el:.0f}s elapsed, "
                          f"{el / (k + 1) * cfg.masks:.0f}s projected", flush=True)

            fields, results = {}, Readout.stack(traces)
            for t in cfg.targets:
                labels = [r.label for r in cfg.reductions if r.label in scalar[t]]
                fields[t] = np.stack([(acc[t][label] / mask_acc.clamp(min=1e-8)).cpu().numpy()
                                      for label in labels]).astype(np.float32)
                results[t]["mapLabels"] = labels
                results[t]["maskMean"] = (mask_acc / cfg.masks).cpu().tolist()
                results[t]["pHat"] = float(mask_acc.mean() / cfg.masks)

        lat, lon = readout.coords()
        return Result(results, fields, lat[ch // 2::ch], lon[cw // 2::cw], baseline_results, None,
                      meta={"masks": cfg.masks, "mask_h": mh, "mask_w": mw, "prob": cfg.prob})
