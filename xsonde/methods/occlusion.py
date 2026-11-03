import os
from dataclasses import dataclass

import numpy as np
import torch

from xsonde import operators
from xsonde.core import Probe, Readout, Result, coerce_reductions, scope_label, source_label, validate_region


@dataclass
class OcclusionConfig:
    source: str
    targets: list
    reductions: list
    block: int = 80
    fill: str = "mean"
    level: int | None = None
    region: list | None = None
    land_only: bool = False

    def __post_init__(self):
        if self.fill not in ("mean", "zero"):
            raise ValueError(f"fill must be 'mean' or 'zero', you provided {self.fill!r}")
        if self.block < 1:
            raise ValueError(f"block must be positive, you provided {self.block}")
        self.region = validate_region(self.region)
        self.reductions = coerce_reductions(self.reductions)


def blocks(h, w, block, land_mask=None, land_only=False, region=None):
    h0, h1, w0, w1 = region if region else (0, h, 0, w)
    out = []
    for i in range(0, h - h % block, block):
        for j in range(0, w - w % block, block):
            if i >= h1 or i + block <= h0 or j >= w1 or j + block <= w0:
                continue
            if land_only and land_mask is not None:
                if land_mask[..., i:i + block, j:j + block].max() <= 0.5:
                    continue
            out.append((i, j))
    return out


class Occlusion(Probe):
    name = "occlusion"
    config_cls = OcclusionConfig

    def run_subpath(self, config_name, start_time):
        cfg = self.cfg
        return os.path.join(cfg.fill, scope_label(cfg.level, cfg.region), config_name,
                            f"{start_time:%Y_%m_%dT%H}",
                            source_label(cfg.source, cfg.level, cfg.region, cfg.land_only))

    def run(self, adapter, batch, donors=None):
        cfg = self.cfg
        x = adapter.get(batch, cfg.source)
        level_idx = self.level_index(adapter, batch, cfg.source)
        land = adapter.land_mask(batch)
        h, w = x.shape[-2], x.shape[-1]
        b = cfg.block
        windows = blocks(h, w, b, land, cfg.land_only, cfg.region)
        if not windows:
            raise ValueError("no occlusion window matches the region and land settings")
        readout = Readout(adapter, batch, cfg.targets, cfg.reductions)
        xm = adapter.to_model_space(batch, cfg.source, x)
        value = operators.fill_value(xm, cfg.fill)

        with torch.inference_mode():
            readout.set_baseline(adapter.predict(batch))
            baseline_results = readout.baseline_results()
            results = readout.empty()
            for (i, j) in windows:
                t = xm.clone()
                if level_idx is not None:
                    t[..., level_idx, i:i + b, j:j + b] = value[..., level_idx, :, :]
                else:
                    t[..., i:i + b, j:j + b] = value
                t = adapter.from_model_space(batch, cfg.source, t)
                pred = adapter.predict(adapter.replace(batch, cfg.source, t))
                for target in cfg.targets:
                    for label, v in readout.reduce(pred, target).items():
                        results[target][label].append(v)

            ni, nj = h // b, w // b
            fields, map_labels = {}, {}
            for target in cfg.targets:
                labels = [r.label for r in cfg.reductions if results[target][r.label][0].ndim == 0]
                grid = np.full((len(labels), ni, nj), np.nan, dtype=np.float32)
                for k, (i, j) in enumerate(windows):
                    for m, label in enumerate(labels):
                        grid[m, i // b, j // b] = float(results[target][label][k])
                fields[target] = grid
                map_labels[target] = labels
            results = Readout.stack(results)
            for target in results:
                results[target]["block"] = [[int(i), int(j)] for (i, j) in windows]
                results[target]["mapLabels"] = map_labels[target]

        lat, lon = readout.coords()
        return Result(results, fields, lat[b // 2::b], lon[b // 2::b], baseline_results, None,
                      meta={"block": b, "fill": cfg.fill})
