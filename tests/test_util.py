import xarray as xr
import dask.array as da
import numpy as np
import pytest
from stmtools import utils

@pytest.fixture
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

class TestHasProperty():
    def test_has_property_str(self, stmat):
        assert utils._has_property(stmat, 'phase')

    def test_has_property_list(self, stmat):
        assert utils._has_property(stmat, ['phase', 'amplitude'])

    def test_has_property_tuple(self, stmat):
        assert utils._has_property(stmat, ('phase', 'amplitude'))
    
    def test_has_not(self, stmat):
        assert not(utils._has_property(stmat, ['phase', 'no_exists']))

    def test_incorrect_type(self, stmat):
        with pytest.raises(ValueError):
            utils._has_property(stmat, 1)
