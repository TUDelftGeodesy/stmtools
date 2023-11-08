import dask.array as da
import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def stmat():
    npoints = 10
    ntime = 5
    return xr.Dataset(
        data_vars=dict(
            amplitude=(
                ["space", "time"],
                da.arange(npoints * ntime).reshape((npoints, ntime)),
            ),
            phase=(
                ["space", "time"],
                da.arange(npoints * ntime).reshape((npoints, ntime)),
            ),
        ),
        coords=dict(
            lon=(["space"], da.arange(npoints)),
            lat=(["space"], da.arange(npoints)),
            time=(["time"], np.arange(ntime)),
        ),
    ).unify_chunks()
