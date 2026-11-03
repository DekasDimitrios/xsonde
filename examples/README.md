# Examples

| File | What it shows |
|---|---|
| `aurora_small_era5_ceteris_paribus.ipynb` | Public Aurora Small, one day of ERA5 from the Climate Data Store, a Ceteris Paribus probe on humidity, curves, maps, saving |
| `era5_loader.py` | The loader the notebook and the sweep share: downloads ERA5 and builds an Aurora `Batch` for a start time |
| `sweeps/aurora_small_era5.yaml` | The same probe as a sweep, run with `PYTHONPATH=. xsonde run sweeps/aurora_small_era5.yaml` |

Data goes to `examples/data/` (or `$XSONDE_ERA5_DIR`); it is not tracked. Results of the sweep
land next to the yaml.

The loader targets the `microsoft-aurora` package. A fork that changed `Metadata` or that
transforms variables before standardising needs its own loader and, for the latter, the
adapter's `transform` option.
