import torch

from xsonde.adapters import ModelAdapter, register_adapter, resolve


@register_adapter
class DictAdapter(ModelAdapter):
    """In-memory adapter for a callable model over a dict of tensors.

    A batch is a dict with keys `vars` (name to tensor), `lat`, `lon`, `levels`, and
    optionally `static` (name to (H, W) tensor) and `sigma` (name to tensor or float).
    The model takes the `vars` dict and returns a dict of forecasts. `transforms` maps a
    variable to a `(forward, inverse)` pair for variables the model transforms before
    standardising.
    """

    name = "dict"

    def __init__(self, model=None, loader=None, transforms=None):
        self.model = resolve(model)
        self.loader = resolve(loader)
        self.transforms = dict(transforms or {})

    def load_batch(self, start_time):
        if self.loader is None:
            raise ValueError("DictAdapter needs a loader to build batches from a start time")
        return self.loader(start_time)

    def get(self, batch, var):
        if var in batch["vars"]:
            return batch["vars"][var]
        if var in batch.get("static", {}):
            return batch["static"][var]
        raise KeyError(f"{var} is not in the batch")

    def replace(self, batch, var, tensor):
        new = dict(batch)
        if var in batch["vars"]:
            new["vars"] = {**batch["vars"], var: tensor}
        else:
            new["static"] = {**batch.get("static", {}), var: tensor}
        return new

    def predict(self, batch):
        return self.model(batch["vars"])

    def lat(self, batch):
        return batch["lat"]

    def lon(self, batch):
        return batch["lon"]

    def levels(self, batch):
        return tuple(batch.get("levels", ()))

    def is_atmos(self, batch, var):
        return self.get(batch, var).dim() == 5

    def land_mask(self, batch):
        return batch.get("static", {}).get("lsm")

    def to_model_space(self, batch, var, x):
        return self.transforms[var][0](x) if var in self.transforms else x

    def from_model_space(self, batch, var, x):
        return self.transforms[var][1](x) if var in self.transforms else x

    def sigma(self, batch, var):
        sig = batch.get("sigma", {}).get(var)
        if sig is None:
            raise KeyError(f"no sigma for {var} in the batch")
        return torch.as_tensor(sig) if not torch.is_tensor(sig) else sig
