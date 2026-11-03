import torch
import inspect

def _lat_weight(lat):
    w = torch.cos(torch.deg2rad(lat))
    return w / w.mean()


def single_point(target_tensor: torch.Tensor, lat: torch.Tensor, h: int, w: int):
    return target_tensor[..., h, w].squeeze()


def global_mean(target_tensor: torch.Tensor, lat: torch.Tensor):
    weights = _lat_weight(lat)
    reduced_tensor = (target_tensor * weights[:, None]).mean()
    return reduced_tensor


def global_mean_level(target_tensor: torch.Tensor, lat: torch.Tensor):
    weights = _lat_weight(lat)
    reduced_tensor = (target_tensor * weights[:, None]).mean(dim=(-1, -2)).squeeze()
    return reduced_tensor


def global_sum(target_tensor: torch.Tensor, lat: torch.Tensor):
    weights = _lat_weight(lat)
    reduced_tensor = (target_tensor * weights[:, None]).sum()
    return reduced_tensor


def global_sum_level(target_tensor: torch.Tensor, lat: torch.Tensor):
    weights = _lat_weight(lat)
    reduced_tensor = (target_tensor * weights[:, None]).sum(dim=(-1, -2)).squeeze()
    return reduced_tensor


def regional_mean(target_tensor: torch.Tensor, lat: torch.Tensor, h0: int, h1: int, w0: int, w1: int):
    weights = _lat_weight(lat[h0:h1])
    sub = target_tensor[..., h0:h1, w0:w1]
    reduced_tensor = (sub * weights[:, None]).mean()
    return reduced_tensor


def regional_mean_level(target_tensor: torch.Tensor, lat: torch.Tensor, h0: int, h1: int, w0: int, w1: int):
    weights = _lat_weight(lat[h0:h1])
    sub = target_tensor[..., h0:h1, w0:w1]
    reduced_tensor = (sub * weights[:, None]).mean(dim=(-1, -2)).squeeze()
    return reduced_tensor


def regional_max(target_tensor: torch.Tensor, lat: torch.Tensor, h0: int, h1: int, w0: int, w1: int):
    return target_tensor[..., h0:h1, w0:w1].max()


def regional_max_level(target_tensor: torch.Tensor, lat: torch.Tensor, h0: int, h1: int, w0: int, w1: int):
    return target_tensor[..., h0:h1, w0:w1].amax(dim=(-1, -2)).squeeze()


def regional_percentile(target_tensor: torch.Tensor, lat: torch.Tensor, h0: int, h1: int, w0: int, w1: int, q=0.99):
    return torch.quantile(target_tensor[..., h0:h1, w0:w1], q)


def regional_percentile_level(target_tensor: torch.Tensor, lat: torch.Tensor, h0: int, h1: int, w0: int, w1: int, q=0.99):
    return torch.quantile(target_tensor[..., h0:h1, w0:w1].flatten(start_dim=-2), q, dim=-1).squeeze()


def mean_abs_response(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    weights = _lat_weight(lat)
    return ((target_tensor - baseline).abs() * weights[:, None]).mean()


def rmse_response(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    weights = _lat_weight(lat)
    return (((target_tensor - baseline) ** 2) * weights[:, None]).mean().sqrt()


def max_abs_response(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    return (target_tensor - baseline).abs().max()


def mean_abs_response_level(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    weights = _lat_weight(lat)
    return ((target_tensor - baseline).abs() * weights[:, None]).mean(dim=(-1, -2)).squeeze()


def rmse_response_level(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    weights = _lat_weight(lat)
    return (((target_tensor - baseline) ** 2) * weights[:, None]).mean(dim=(-1, -2)).sqrt().squeeze()


def max_abs_response_level(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    return (target_tensor - baseline).abs().amax(dim=(-1, -2)).squeeze()


def field_max(target_tensor: torch.Tensor, lat: torch.Tensor):
    return target_tensor.max()


def field_max_level(target_tensor: torch.Tensor, lat: torch.Tensor):
    return target_tensor.amax(dim=(-1, -2)).squeeze()


def field_min(target_tensor: torch.Tensor, lat: torch.Tensor):
    return target_tensor.min()


def field_min_level(target_tensor: torch.Tensor, lat: torch.Tensor):
    return target_tensor.amin(dim=(-1, -2)).squeeze()


def spatial_std(target_tensor: torch.Tensor, lat: torch.Tensor):
    weights = _lat_weight(lat)
    mean = (target_tensor * weights[:, None]).mean()
    return (((target_tensor - mean) ** 2) * weights[:, None]).mean().sqrt()


def spatial_std_level(target_tensor: torch.Tensor, lat: torch.Tensor):
    weights = _lat_weight(lat)
    mean = (target_tensor * weights[:, None]).mean(dim=(-1, -2), keepdim=True)
    return (((target_tensor - mean) ** 2) * weights[:, None]).mean(dim=(-1, -2)).sqrt().squeeze()


def median(target_tensor: torch.Tensor, lat: torch.Tensor):
    return target_tensor.median()


def median_level(target_tensor: torch.Tensor, lat: torch.Tensor):
    return target_tensor.flatten(start_dim=-2).median(dim=-1).values.squeeze()


def percentile(target_tensor: torch.Tensor, lat: torch.Tensor, q: float = 0.95):
    return torch.quantile(target_tensor, q)


def percentile_level(target_tensor: torch.Tensor, lat: torch.Tensor, q: float = 0.95):
    return torch.quantile(target_tensor.flatten(start_dim=-2), q, dim=-1).squeeze()


def response_spread(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    weights = _lat_weight(lat)
    response = target_tensor - baseline
    mean = (response * weights[:, None]).mean()
    return (((response - mean) ** 2) * weights[:, None]).mean().sqrt()


def response_spread_level(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    weights = _lat_weight(lat)
    response = target_tensor - baseline
    mean = (response * weights[:, None]).mean(dim=(-1, -2), keepdim=True)
    return (((response - mean) ** 2) * weights[:, None]).mean(dim=(-1, -2)).sqrt().squeeze()


def fraction_increasing(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    weights = _lat_weight(lat)
    return ((target_tensor > baseline).to(target_tensor.dtype) * weights[:, None]).mean()


def fraction_increasing_level(target_tensor: torch.Tensor, lat: torch.Tensor, baseline: torch.Tensor):
    weights = _lat_weight(lat)
    return ((target_tensor > baseline).to(target_tensor.dtype) * weights[:, None]).mean(dim=(-1, -2)).squeeze()


def _masked_mean(target_tensor: torch.Tensor, lat: torch.Tensor, keep: torch.Tensor):
    weights = _lat_weight(lat)[:, None]
    num = (target_tensor * weights * keep).sum(dim=(-1, -2))
    den = (weights * keep).sum(dim=(-1, -2))
    return num / den


def land_mean(target_tensor: torch.Tensor, lat: torch.Tensor, mask: torch.Tensor):
    return _masked_mean(target_tensor, lat, mask > 0.5).mean()


def ocean_mean(target_tensor: torch.Tensor, lat: torch.Tensor, mask: torch.Tensor):
    return _masked_mean(target_tensor, lat, mask <= 0.5).mean()


def land_mean_level(target_tensor: torch.Tensor, lat: torch.Tensor, mask: torch.Tensor):
    return _masked_mean(target_tensor, lat, mask > 0.5).squeeze()


def ocean_mean_level(target_tensor: torch.Tensor, lat: torch.Tensor, mask: torch.Tensor):
    return _masked_mean(target_tensor, lat, mask <= 0.5).squeeze()


def column_integral(target_tensor: torch.Tensor, lat: torch.Tensor, levels):
    if target_tensor.dim() != 5:  # atmos targets only (B, T, C, H, W); surf has no column
        return target_tensor.new_full((), float("nan"))
    pressures = target_tensor.new_tensor([lvl * 100.0 for lvl in levels])  # hPa -> Pa
    column = torch.trapezoid(target_tensor, x=pressures, dim=-3) / 9.80665  # (B, T, H, W), kg/m^2 for q
    weights = _lat_weight(lat)
    return (column * weights[:, None]).mean()


REDUCTIONS = {
    "single_point": single_point,
    "global_mean": global_mean,
    "global_mean_level": global_mean_level,
    "global_sum": global_sum,
    "global_sum_level": global_sum_level,
    "regional_mean": regional_mean,
    "regional_mean_level": regional_mean_level,
    "regional_max": regional_max,
    "regional_max_level": regional_max_level,
    "regional_percentile": regional_percentile,
    "regional_percentile_level": regional_percentile_level,
    "mean_abs_response": mean_abs_response,
    "rmse_response": rmse_response,
    "max_abs_response": max_abs_response,
    "mean_abs_response_level": mean_abs_response_level,
    "rmse_response_level": rmse_response_level,
    "max_abs_response_level": max_abs_response_level,
    "field_max": field_max,
    "field_max_level": field_max_level,
    "field_min": field_min,
    "field_min_level": field_min_level,
    "spatial_std": spatial_std,
    "spatial_std_level": spatial_std_level,
    "median": median,
    "median_level": median_level,
    "percentile": percentile,
    "percentile_level": percentile_level,
    "response_spread": response_spread,
    "response_spread_level": response_spread_level,
    "fraction_increasing": fraction_increasing,
    "fraction_increasing_level": fraction_increasing_level,
    "land_mean": land_mean,
    "ocean_mean": ocean_mean,
    "land_mean_level": land_mean_level,
    "ocean_mean_level": ocean_mean_level,
    "column_integral": column_integral,
}


def reduce(name: str, target_tensor: torch.Tensor, lat: torch.Tensor, baseline=None, mask=None, levels=None, **params):
    if name not in REDUCTIONS:
        raise ValueError(f"The reduction method {name} is not supported.")
    fn = REDUCTIONS[name]
    sig = inspect.signature(fn).parameters
    if "baseline" in sig:
        params["baseline"] = baseline
    if "mask" in sig:
        params["mask"] = mask
    if "levels" in sig:
        params["levels"] = levels
    return fn(target_tensor, lat, **params)
