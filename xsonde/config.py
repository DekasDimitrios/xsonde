from datetime import datetime

import yaml


def load_sweep(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def method_params(method):
    params = dict(method)
    params.pop("name", None)
    return params


def as_datetime(value):
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
