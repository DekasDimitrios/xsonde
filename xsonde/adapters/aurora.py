import dataclasses

import torch

from xsonde.adapters import ModelAdapter, register_adapter, resolve

SIZES = {
    "small": dict(encoder_depths=(2, 6, 2), encoder_num_heads=(4, 8, 16),
                  decoder_depths=(2, 6, 2), decoder_num_heads=(16, 8, 4),
                  embed_dim=256, num_heads=8),
    "large": dict(encoder_depths=(6, 10, 8), encoder_num_heads=(8, 16, 32),
                  decoder_depths=(8, 10, 6), decoder_num_heads=(32, 16, 8),
                  embed_dim=512, num_heads=16),
}


@register_adapter
class AuroraAdapter(ModelAdapter):
    """Adapter for Microsoft Aurora and forks that keep its Batch and Metadata classes.

    `loader` is a callable or dotted path (`package.module:function`) that builds a
    Batch for a start time; data access stays outside the library. `checkpoint_format`
    is `aurora` for checkpoints saved by Aurora itself and `lightning` for
    `state_dict` checkpoints whose keys carry a `net.` prefix.
    """

    name = "aurora"

    def __init__(self, loader=None, checkpoint=None, checkpoint_format="aurora", size="small",
                 device="cuda", model_kwargs=None, random_init_seed=None, model=None,
                 transform=None, hub=None):
        self.loader = resolve(loader)
        self.hub = tuple(hub) if hub else None
        self.transform = resolve(transform)
        self.checkpoint = checkpoint
        self.checkpoint_format = checkpoint_format
        self.size = size
        self.device = device
        self.model_kwargs = dict(model_kwargs or {})
        self.random_init_seed = random_init_seed
        self.model = model

    def load_model(self):
        if self.model is not None:
            self.model.eval()
            return self.model
        from aurora import Aurora

        if self.size not in SIZES:
            raise ValueError(f"size must be one of {sorted(SIZES)}, you provided {self.size!r}")
        if self.random_init_seed is not None:
            torch.manual_seed(self.random_init_seed)
        kwargs = {**SIZES[self.size], "use_lora": False, **self.model_kwargs}
        model = Aurora(**kwargs)
        if self.checkpoint and self.random_init_seed is None:
            if self.checkpoint_format == "lightning":
                ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
                state = ckpt.get("state_dict", ckpt)
                state = {k[4:] if k.startswith("net.") else k: v for k, v in state.items()}
                model.load_state_dict(state)
            else:
                model.load_checkpoint_local(self.checkpoint)
        elif self.hub and self.random_init_seed is None:
            model.load_checkpoint(*self.hub)
        model.to(self.device)
        model.float()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        return model

    def load_batch(self, start_time):
        if self.loader is None:
            raise ValueError("AuroraAdapter needs a loader to build a Batch from a start time")
        return self.loader(start_time)

    @staticmethod
    def _group(batch, var):
        if var in batch.surf_vars:
            return "surf_vars"
        if var in batch.atmos_vars:
            return "atmos_vars"
        if var in batch.static_vars:
            return "static_vars"
        raise KeyError(f"{var} is not a part of the batch")

    def get(self, batch, var):
        return getattr(batch, self._group(batch, var))[var]

    def replace(self, batch, var, tensor):
        group = self._group(batch, var)
        tensors = dict(getattr(batch, group))
        tensors[var] = tensor
        return dataclasses.replace(batch, **{group: tensors})

    def predict(self, batch):
        out = self.model(batch)
        if isinstance(out, tuple):
            out = out[0]
        return {**out.surf_vars, **out.atmos_vars}

    def _crop(self, tensor):
        patch = getattr(self.model, "patch_size", 4)
        h = tensor.shape[-2]
        return tensor[..., :h - h % patch, :]

    def lat(self, batch):
        return self._crop(batch.metadata.lat).to(self.device)

    def lon(self, batch):
        return batch.metadata.lon.to(self.device)

    def levels(self, batch):
        return tuple(batch.metadata.atmos_levels)

    def is_atmos(self, batch, var):
        return var in batch.atmos_vars

    def land_mask(self, batch):
        lsm = batch.static_vars.get("lsm")
        return None if lsm is None else self._crop(lsm).to(self.device)

    def sigma(self, batch, var):
        scales = getattr(batch.metadata, "scales", None)
        if not scales:
            from aurora.normalisation import scales
        if self.is_atmos(batch, var):
            x = batch.atmos_vars[var]
            return torch.tensor([scales[f"{var}_{lvl}"] for lvl in batch.metadata.atmos_levels],
                                device=x.device, dtype=x.dtype)
        return float(scales[var])

    def to_model_space(self, batch, var, x):
        return x if self.transform is None else self.transform(x, var, direct=True)

    def from_model_space(self, batch, var, x):
        return x if self.transform is None else self.transform(x, var, direct=False)

    def named_modules(self):
        if self.model is None:
            return {}
        m = self.model
        out = {}
        for i, layer in enumerate(m.encoder.level_agg.layers):
            out[f"perceiver.encoder.{i}"] = layer[0]
        for i, layer in enumerate(m.decoder.level_decoder.layers):
            out[f"perceiver.decoder.{i}"] = layer[0]
        for i, layer in enumerate(m.backbone.encoder_layers):
            out[f"backbone.enc{i}"] = layer
        for i, layer in enumerate(m.backbone.decoder_layers):
            out[f"backbone.dec{i}"] = layer
        return out
