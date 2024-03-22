from collections.abc import Iterable

import xarray as xr


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

    original_dims_order = other.dims

    # for dims that are not in coords, unstack the data
    indexer = {}
    for dim in other.dims:
        if dim not in other.coords.keys():
            indexer = {
                dim: [
                    coord for coord in other.coords.keys() if dim in other.coords[coord].dims
                    ]
                }
            other = other.set_index(indexer)
            other = other.unstack(indexer)

    # do the slicing
    for coord in buffer.keys():
        coord_min = ds[coord].min() - buffer[coord]
        coord_max = ds[coord].max() + buffer[coord]
        if coord in other.dims:
            other = other.sel({coord: slice(coord_min, coord_max)})

    # stack back
    for dim, coords in indexer.items():
        for coord in coords:
            coord_value = xr.DataArray(other.coords[coord].values, dims=dim)
            other = other.sel({coord: coord_value})

    # transpose the dimensions back to the original order
    other = other.transpose(*original_dims_order)

    return other
