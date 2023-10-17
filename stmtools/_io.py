import re
import math
from pathlib import Path
import xarray as xr
import dask.dataframe as dd
import dask.array as da


def from_csv(
    file: str | Path,
    blocksize: int = 200e6,
    space_pattern: str = "^pnt_",
    spacetime_pattern: dict
    | None = {"^d_": "deformation", "^a_": "amplitude", "^h2ph_": "h2ph"},
    output_chunksize: dict | None = None,
    coords_cols: list | dict = {"pnt_lat": "lat", "pnt_lon": "lon"},
):
    # requirements:
    # - space cols have uniform patter
    # - each time attr has a uniform pattern
    # output_chunksize example: {"space": 50000, "time": -1}

    # Load csv as Dask DataFrame
    ddf = dd.read_csv(file, blocksize=blocksize)

    # Take the first column and compute the chunk sizes
    da_col0 = ddf[ddf.columns[0]].to_dask_array()
    da_col0.compute_chunk_sizes()
    chunks = da_col0.chunks

    # Count time dimension by one spacetime column
    time_shape = 0
    if spacetime_pattern is not None:
        key = list(spacetime_pattern.keys())[0]
        for column in ddf.columns:
            if re.match(re.compile(key), column):
                time_shape += 1

    # Initiate a template STM
    coords = {
        "space": range(da_col0.shape[0]),
        "time": range(time_shape),
    }
    stmat = xr.Dataset(coords=coords)

    # Read csv by columns
    dict_temp_da = dict()  # Temporary dask array collector
    for k in spacetime_pattern.keys():
        dict_temp_da[k] = []

    # Temporaly save time-series columns to lists in dict_temp_da
    for column in ddf.columns:
        if _is_space(space_pattern, column):
            da_pnt = ddf[column].to_dask_array(lengths=chunks[0])
            stmat = stmat.assign({column: (("space"), da_pnt)})
        else:
            for k in spacetime_pattern.keys():
                if re.match(re.compile(f"{k}"), column):
                    da_list = dict_temp_da[k]
                    da_list.append(ddf[column].to_dask_array(lengths=chunks[0]))
                    dict_temp_da[k] = da_list

    # Stack dask arrays in dict_temp_da, assign to STM
    for k, da_list in dict_temp_da.items():
        da_ts = da.stack(da_list)  # Stack on time dimension
        stmat = stmat.assign({spacetime_pattern[k]: (("time", "space"), da_ts)})

    # Rearrage space and time order
    stmat = stmat.transpose("space", "time")

    # Uniform chunking
    if output_chunksize is None:
        output_chunksize = chunks[0][0]
        if len(chunks[0])>1:
            # If more than one chunk, use the first chunk size, round to next 5000
            output_chunksize = _round_chunksize(output_chunksize)
    stmat = stmat.chunk(output_chunksize)

    # Set coordinates
    if isinstance(coords_cols, dict):
        stmat = stmat.rename(coords_cols)
        stmat = stmat.set_coords(list(coords_cols.values()))
    else:
        stmat = stmat.set_coords(coords_cols)

    return stmat

def _is_space(space_pattern: str, column):
    """Check if column is a space attribute"""
    return re.match(re.compile(space_pattern), column)

def _round_chunksize(size):
    """round size to next 5000
    """
    return math.ceil(size/5000)*5000

