from datetime import datetime

import torch
import torch.nn.functional as F

from xsonde.adapters import DictAdapter

H, W, L, T = 36, 72, 4, 2
LEVELS = (300, 500, 850, 1000)


def smooth(a):
    return F.avg_pool2d(a.flatten(0, -3), 3, stride=1, padding=1).reshape(a.shape)


def toy_model(v):
    q1 = v["q"][:, -1:]
    zwd1 = v["zwd"][:, -1:]
    tp1 = v["tp"][:, -1:]
    tp = 0.5 * tp1 + 0.02 * smooth(q1[:, :, -1]) + 1e-6 * smooth(zwd1)
    q = 1.1 * q1
    zwd = zwd1 + 1000.0 * q1.mean(dim=2)
    return {"tp": tp, "q": q, "zwd": zwd}


def load_batch(start_time=None):
    seed = 0 if start_time is None else int(start_time.strftime("%Y%m%d%H")) % 2147483647
    g = torch.Generator().manual_seed(seed)
    q = torch.rand(1, T, L, H, W, generator=g) * 0.01
    zwd = torch.rand(1, T, H, W, generator=g) * 100.0
    tp = torch.rand(1, T, H, W, generator=g) * 0.001
    lsm = (torch.rand(H, W, generator=g) > 0.6).float()
    return {
        "vars": {"q": q, "zwd": zwd, "tp": tp},
        "static": {"lsm": lsm},
        "lat": torch.linspace(89.0, -89.0, H),
        "lon": torch.linspace(0.0, 355.0, W),
        "levels": LEVELS,
        "sigma": {"q": torch.tensor([0.001, 0.002, 0.004, 0.005]), "zwd": 30.0, "tp": 2.0},
    }


EPS = 1e-5


def log_tp(x):
    return torch.log(1 + x / EPS)


def exp_tp(x):
    return EPS * (torch.exp(x) - 1)


def adapter():
    return DictAdapter(model=toy_model, loader=load_batch, transforms={"tp": (log_tp, exp_tp)})


REDUCTIONS = [
    {"name": "global_mean", "label": "globalMean"},
    {"name": "regional_mean", "label": "box", "h0": 8, "h1": 12, "w0": 20, "w1": 24},
    {"name": "rmse_response", "label": "rmseResp"},
    {"name": "land_mean", "label": "landMean"},
    {"name": "global_mean_level", "label": "globalMeanLevel"},
]

DATE = datetime(2014, 1, 15, 0)
