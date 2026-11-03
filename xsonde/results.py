import os
import json

import numpy as np

from xsonde.core import Result


def _write_fields(ds, stem):
    try:
        ds.to_netcdf(stem + ".nc")
        return stem + ".nc"
    except (ValueError, ImportError, ModuleNotFoundError):
        arrays = {k: ds[k].values for k in ds.data_vars}
        np.savez(stem + ".npz", lat=ds["lat"].values, lon=ds["lon"].values,
                 attrs=json.dumps(ds.attrs), **arrays)
        return stem + ".npz"


def save(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for target, ds in result.to_datasets().items():
        tdir = os.path.join(out_dir, target)
        os.makedirs(tdir, exist_ok=True)
        _write_fields(ds, os.path.join(tdir, "fields"))
        with open(os.path.join(tdir, "results.json"), "w") as f:
            json.dump(result.results.get(target, {}), f)
        if result.baseline_results and target in result.baseline_results:
            with open(os.path.join(tdir, "baseline.json"), "w") as f:
                json.dump(result.baseline_results[target], f)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(result.meta, f, default=str)


def _read_fields(tdir):
    nc = os.path.join(tdir, "fields.nc")
    npz = os.path.join(tdir, "fields.npz")
    if os.path.isfile(nc):
        import xarray as xr

        with xr.open_dataset(nc) as ds:
            fields = ds["fields"].values
            baseline = ds["baseline"].values if "baseline" in ds else None
            lat, lon = ds["lat"].values, ds["lon"].values
        return fields, baseline, lat, lon
    if os.path.isfile(npz):
        data = np.load(npz)
        baseline = data["baseline"] if "baseline" in data.files else None
        return data["fields"], baseline, data["lat"], data["lon"]
    return None


def load(out_dir):
    results, fields, baseline_results, baseline_fields = {}, {}, {}, {}
    lat = lon = None
    for target in sorted(os.listdir(out_dir)):
        tdir = os.path.join(out_dir, target)
        if not os.path.isdir(tdir):
            continue
        read = _read_fields(tdir)
        if read is None:
            continue
        fields[target], baseline, lat, lon = read
        if baseline is not None:
            baseline_fields[target] = baseline
        rpath = os.path.join(tdir, "results.json")
        results[target] = json.load(open(rpath)) if os.path.isfile(rpath) else {}
        bpath = os.path.join(tdir, "baseline.json")
        if os.path.isfile(bpath):
            baseline_results[target] = json.load(open(bpath))
    mpath = os.path.join(out_dir, "meta.json")
    meta = json.load(open(mpath)) if os.path.isfile(mpath) else {}
    return Result(results, fields, lat, lon, baseline_results or None, baseline_fields or None, meta)
