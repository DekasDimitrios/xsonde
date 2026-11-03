import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch

from xsonde.reductions import reduce


@dataclass
class ReductionSpec:
    name: str
    label: str
    params: dict = field(default_factory=dict)


def coerce_reductions(items):
    out = []
    for item in items or []:
        if isinstance(item, ReductionSpec):
            out.append(item)
            continue
        temp = dict(item)
        name = temp.pop("name")
        label = temp.pop("label", name)
        out.append(ReductionSpec(name=name, label=label, params=temp))
    return out


def validate_region(region):
    if region is None:
        return None
    if len(region) != 4:
        raise ValueError(f"region needs [h0, h1, w0, w1], you provided {region}")
    h0, h1, w0, w1 = region
    if h0 >= h1 or w0 >= w1:
        raise ValueError(f"region needs h0 < h1 and w0 < w1, you provided {region}")
    return [int(h0), int(h1), int(w0), int(w1)]


def source_label(source, level=None, region=None, land_only=False, time=None):
    label = source
    if land_only:
        label += "_land"
    if level is not None:
        label += f"_level_{level}"
    if region is not None:
        label += "_region_" + "_".join(map(str, region))
    if time is not None and time != "both":
        label += f"_{time}"
    return label


def scope_label(level=None, region=None):
    return "targeted" if (level is not None or region is not None) else "variables"


def field2d(tensor):
    t = tensor.mean(dim=-3) if tensor.dim() == 5 else tensor
    return t.squeeze().detach().cpu().numpy()


@dataclass
class Result:
    results: dict
    fields: dict
    lat: np.ndarray
    lon: np.ndarray
    baseline_results: dict | None = None
    baseline_fields: dict | None = None
    meta: dict = field(default_factory=dict)

    def to_datasets(self):
        import xarray as xr

        out = {}
        for target, arr in self.fields.items():
            arr = np.asarray(arr)
            labels = (self.results.get(target) or {}).get("mapLabels")
            da = xr.DataArray(arr, dims=("sample", "lat", "lon"),
                              coords={"sample": np.arange(arr.shape[0]), "lat": self.lat, "lon": self.lon},
                              name="fields")
            ds = da.to_dataset()
            if labels is not None and len(labels) == arr.shape[0]:
                ds.attrs["sample_labels"] = json.dumps(list(labels))
            if self.baseline_fields and target in self.baseline_fields:
                ds["baseline"] = xr.DataArray(np.asarray(self.baseline_fields[target]), dims=("lat", "lon"))
            out[target] = ds
        return out


class Readout:
    def __init__(self, adapter, batch, targets, reductions):
        self.adapter = adapter
        self.batch = batch
        self.targets = list(targets)
        self.reductions = list(reductions)
        self.lat = adapter.lat(batch)
        self.lon = adapter.lon(batch)
        self.levels = tuple(adapter.levels(batch))
        self.mask = adapter.land_mask(batch)
        self.baselines = {}

    def set_baseline(self, pred):
        self.baselines = {t: pred[t] for t in self.targets}

    def reduce(self, pred, target):
        tt = pred[target]
        base = self.baselines.get(target, tt)
        return {r.label: reduce(r.name, tt, self.lat, baseline=base, mask=self.mask,
                                levels=self.levels, **r.params) for r in self.reductions}

    def field(self, pred, target):
        return field2d(pred[target])

    def baseline_results(self):
        return {t: {label: v.detach().cpu().tolist() for label, v in self.reduce(self.baselines, t).items()}
                for t in self.targets}

    def baseline_fields(self):
        return {t: self.field(self.baselines, t) for t in self.targets}

    def coords(self):
        return self.lat.detach().cpu().numpy(), self.lon.detach().cpu().numpy()

    def empty(self):
        return {t: {r.label: [] for r in self.reductions} for t in self.targets}

    @staticmethod
    def stack(results):
        return {t: {label: torch.stack(v).detach().cpu().tolist() for label, v in red.items()}
                for t, red in results.items()}


class Probe(ABC):
    name = "probe"
    config_cls = None

    def __init__(self, params):
        self.cfg = self.config_cls(**params)

    def run_subpath(self, config_name, start_time):
        return os.path.join(config_name, f"{start_time:%Y_%m_%dT%H}")

    @abstractmethod
    def run(self, adapter, batch, donors=None):
        ...

    def level_index(self, adapter, batch, source):
        if self.cfg.level is None:
            return None
        if not adapter.is_atmos(batch, source):
            raise ValueError(
                f"level={self.cfg.level} hPa was given, but '{source}' has no vertical levels. "
                f"Remove 'level' from the config, or pick an atmospheric variable."
            )
        levels = list(adapter.levels(batch))
        if self.cfg.level not in levels:
            raise ValueError(f"level {self.cfg.level} is not one of {levels}")
        return levels.index(self.cfg.level)
