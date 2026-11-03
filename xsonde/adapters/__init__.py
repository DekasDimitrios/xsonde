import importlib
from abc import ABC, abstractmethod

ADAPTERS = {}


def register_adapter(cls):
    ADAPTERS[cls.name] = cls
    return cls


def resolve(dotted):
    if dotted is None or callable(dotted):
        return dotted
    module, _, attr = dotted.partition(":")
    if not attr:
        module, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module), attr)


class ModelAdapter(ABC):
    """Everything a probe needs from a model and its batch format.

    A batch is opaque to the engine. Variables are addressed by name and exposed as
    tensors shaped (B, T, H, W) for surface fields and (B, T, L, H, W) for atmospheric
    fields, in the units the model consumes. Predictions come back in physical units
    shaped (B, 1, H, W) or (B, 1, L, H, W). Coordinates and the land mask describe the
    input grid. A model whose output grid is smaller than its input grid, as Aurora's is
    when it drops the 721st latitude, must return the leading rows and columns of the
    input grid, because probes align coordinates to predictions by slicing from the top
    left.
    """

    name = "adapter"

    @classmethod
    def from_config(cls, cfg):
        params = dict(cfg)
        params.pop("name", None)
        return cls(**params)

    def load_model(self):
        """Instantiate the model once per process, before any batch is loaded."""
        return None

    @abstractmethod
    def load_batch(self, start_time):
        """Return the batch whose first input step is `start_time`."""

    @abstractmethod
    def get(self, batch, var):
        """Tensor of `var` inside `batch`."""

    @abstractmethod
    def replace(self, batch, var, tensor):
        """New batch with `var` swapped for `tensor`; the original is left untouched."""

    @abstractmethod
    def predict(self, batch):
        """Forecast as a dict of variable name to tensor in physical units."""

    @abstractmethod
    def lat(self, batch):
        """Latitudes as a 1D tensor, north to south."""

    @abstractmethod
    def lon(self, batch):
        """Longitudes as a 1D tensor."""

    @abstractmethod
    def levels(self, batch):
        """Pressure levels of the atmospheric variables, in hPa."""

    @abstractmethod
    def is_atmos(self, batch, var):
        """Whether `var` carries a level axis."""

    def land_mask(self, batch):
        """Land fraction as an (H, W) tensor, or None when the batch has no land mask."""
        return None

    def sigma(self, batch, var):
        """Standard deviation of `var` in batch units, per level for atmospheric variables."""
        raise NotImplementedError(f"{self.name} exposes no normalisation statistics")

    def to_model_space(self, batch, var, x):
        """Map `x` from batch units to the space the model normalises in; identity by default.

        Override it for variables the model transforms before standardising, such as a
        log-transformed precipitation, so that sigma-scaled perturbations act in that space.
        """
        return x

    def from_model_space(self, batch, var, x):
        """Inverse of `to_model_space`."""
        return x

    def lower_bound(self, batch, var):
        """Smallest admissible value of `var` in batch units, or None when unbounded.

        Perturbations clamp to it, so a log-transformed precipitation never goes
        negative while humidity is left free, as in the reference study.
        """
        return None

    def named_modules(self):
        """Modules that internal probes may hook, by stable name."""
        return {}


def build_adapter(cfg):
    name = cfg.get("name")
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {name!r}, registered: {sorted(ADAPTERS)}")
    return ADAPTERS[name].from_config(cfg)


from xsonde.adapters.memory import DictAdapter
from xsonde.adapters.aurora import AuroraAdapter

__all__ = ["ADAPTERS", "AuroraAdapter", "DictAdapter", "ModelAdapter", "build_adapter",
           "register_adapter", "resolve"]
