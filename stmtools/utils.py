import xarray as xr
from collections.abc import Iterable


def _has_property(ds, keys: str | Iterable):
    if isinstance(keys, str):
        return keys in ds.data_vars.keys()
    elif isinstance(keys, Iterable):
        return set(keys).issubset(ds.data_vars.keys())
    else:
        raise ValueError(f"Invalid type of keys: {type(keys)}.")


def crop(ds, other, buffer):
    """Crop the other to a given buffer around ds.

    Parameters
    ----------
    ds : xarray.Dataset | xarray.DataArray
        Dataset to crop to.
    other : xarray.Dataset | xarray.DataArray
        Dataset to crop.
    buffer : dict
        A dictionary with the buffer values for each dimension.

    Returns
    -------
    xarray.Dataset
        Cropped dataset.
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    if isinstance(other, xr.DataArray):
        other = other.to_dataset()

    if not isinstance(buffer, dict):
        raise ValueError(f"Invalid type of buffer: {type(buffer)}.")
    for coord in buffer.keys():
        if coord not in ds.coords.keys():
            raise ValueError(f"coordinate '{coord}' not found in ds.")
        if coord not in other.coords.keys():
            raise ValueError(f"coordinate '{coord}' not found in other.")

    other = unstack(other)
    for coord in buffer.keys():
        coord_min = ds[coord].min() - buffer[coord]
        coord_max = ds[coord].max() + buffer[coord]
        other = other.sel({coord: slice(coord_min, coord_max)})
    return other


def unstack(ds):
    for dim in ds.dims:
        if dim not in ds.coords:
            indexer = {dim: [coord for coord in ds.coords if dim in ds[coord].dims]}
            ds = ds.set_index(indexer)
            ds = ds.unstack(dim)
    return ds