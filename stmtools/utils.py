from collections.abc import Iterable


def _has_property(ds, keys: str | Iterable):
    if isinstance(keys, str):
        return keys in ds.data_vars.keys()
    elif isinstance(keys, Iterable):
        return set(keys).issubset(ds.data_vars.keys())
    else:
        raise ValueError(f"Invalid type of keys: {type(keys)}.")
