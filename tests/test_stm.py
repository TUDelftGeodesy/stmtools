import xarray as xr
import dask.array as da
import numpy as np
from stm import stm
import pytest


@pytest.fixture
def stmat():
    npoints = 10
    ntime = 5
    return xr.Dataset(
        data_vars=dict(
            amplitude=(
                ["points", "time"],
                da.arange(npoints * ntime).reshape((npoints, ntime)),
            ),
            phase=(
                ["points", "time"],
                da.arange(npoints * ntime).reshape((npoints, ntime)),
            ),
        ),
        coords=dict(
            lon=(["points"], da.arange(npoints) * 2),
            lat=(["points"], da.arange(npoints) * 3),
            points=(["points"], np.arange(npoints)),
            time=(["time"], np.arange(ntime)),
        ),
    ).unify_chunks()


class TestSubset:
    def test_method_not_implemented(self, stmat):
        with pytest.raises(NotImplementedError):
            stmat.stm.subset(method="something_else")

    def test_subset_with_threshold(self, stmat):
        # dummy threshold, first three are 2, others 1
        # test selecting first 3
        v_thres = np.ones(
            stmat.points.shape,
        )
        v_thres[0:3] = 2
        stmat = stmat.assign(
            {
                "thres": (
                    ["points"],
                    da.from_array(v_thres),
                )
            }
        )
        stmat_subset = stmat.stm.subset(method="threshold", var="thres", threshold=">1")
        assert stmat_subset.equals(stmat.sel(points=[0, 1, 2]))
