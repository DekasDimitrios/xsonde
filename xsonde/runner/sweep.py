import copy
import itertools



def set_in(dict, path, value):
    *parents, leaf = path.split(".")
    root = dict
    for key in parents:
        root = root.setdefault(key, {})
    root[leaf] = value

def expand_sweep(cfg):
    base_cfg = copy.deepcopy(cfg)

    sweep_dict = dict(base_cfg.pop("sweep", {}) or {})
    if not sweep_dict:
        return [base_cfg]

    pairs = sweep_dict.pop("pairs", None)
    if pairs is None:
        pair_groups = []
    elif isinstance(pairs, dict):
        pair_groups = list(pairs.values())
    else:
        pair_groups = [pairs]

    axes = list(sweep_dict.keys())
    value_lists = [sweep_dict[ax] for ax in axes]

    cfgs = []
    for combo in itertools.product(*pair_groups):
        for values in itertools.product(*value_lists):
            exp_cfg = copy.deepcopy(base_cfg)
            for pair in combo:
                for path, value in pair.items():
                    set_in(exp_cfg, path, value)
            for path, value in zip(axes, values):
                set_in(exp_cfg, path, value)
            cfgs.append(exp_cfg)

    return cfgs