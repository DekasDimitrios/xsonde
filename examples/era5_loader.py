import os
from datetime import timedelta

import torch
import xarray as xr

from aurora import Batch, Metadata

DATA_DIR = os.environ.get("XSONDE_ERA5_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
SURF = {"2t": "t2m", "10u": "u10", "10v": "v10", "msl": "msl"}
STATIC = {"z": "z", "slt": "slt", "lsm": "lsm"}
ATMOS = ("t", "u", "v", "q", "z")
TIMES = ["00:00", "06:00", "12:00", "18:00"]


def paths(day, data_dir=DATA_DIR):
    stamp = day.strftime("%Y-%m-%d")
    return {
        "static": os.path.join(data_dir, "static.nc"),
        "surface": os.path.join(data_dir, f"{stamp}-surface-level.nc"),
        "atmos": os.path.join(data_dir, f"{stamp}-atmospheric.nc"),
    }


def download(day, data_dir=DATA_DIR):
    import cdsapi

    os.makedirs(data_dir, exist_ok=True)
    p = paths(day, data_dir)
    client = cdsapi.Client()
    common = {"product_type": "reanalysis", "data_format": "netcdf", "download_format": "unarchived"}
    if not os.path.exists(p["static"]):
        client.retrieve("reanalysis-era5-single-levels", {
            **common, "variable": ["geopotential", "land_sea_mask", "soil_type"],
            "year": "2023", "month": "01", "day": "01", "time": "00:00",
        }, p["static"])
    if not os.path.exists(p["surface"]):
        client.retrieve("reanalysis-era5-single-levels", {
            **common, "variable": ["2m_temperature", "10m_u_component_of_wind",
                                   "10m_v_component_of_wind", "mean_sea_level_pressure"],
            "year": day.strftime("%Y"), "month": day.strftime("%m"), "day": day.strftime("%d"), "time": TIMES,
        }, p["surface"])
    if not os.path.exists(p["atmos"]):
        client.retrieve("reanalysis-era5-pressure-levels", {
            **common, "variable": ["temperature", "u_component_of_wind", "v_component_of_wind",
                                   "specific_humidity", "geopotential"],
            "pressure_level": [str(level) for level in LEVELS],
            "year": day.strftime("%Y"), "month": day.strftime("%m"), "day": day.strftime("%d"), "time": TIMES,
        }, p["atmos"])
    return p


def _coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"none of {candidates} found in {list(ds.coords)}")


def load_batch(start_time, data_dir=DATA_DIR):
    p = paths(start_time, data_dir)
    static = xr.open_dataset(p["static"])
    surf = xr.open_dataset(p["surface"])
    atmos = xr.open_dataset(p["atmos"])

    tname = _coord(surf, ("valid_time", "time"))
    times = surf[tname].values.astype("datetime64[s]").tolist()
    second = start_time + timedelta(hours=6)
    if start_time not in times or second not in times:
        raise ValueError(f"{p['surface']} holds {times}, not {start_time} and {second}")
    sel = [times.index(start_time), times.index(second)]

    lname = _coord(atmos, ("pressure_level", "level"))
    atmos = atmos.sel({lname: list(LEVELS)})

    return Batch(
        surf_vars={k: torch.from_numpy(surf[v].values[sel][None]) for k, v in SURF.items()},
        static_vars={k: torch.from_numpy(static[v].values[0]) for k, v in STATIC.items()},
        atmos_vars={k: torch.from_numpy(atmos[k].values[sel][None]) for k in ATMOS},
        metadata=Metadata(
            lat=torch.from_numpy(surf.latitude.values),
            lon=torch.from_numpy(surf.longitude.values),
            time=(times[sel[1]],),
            atmos_levels=tuple(int(level) for level in atmos[lname].values),
        ),
    )
