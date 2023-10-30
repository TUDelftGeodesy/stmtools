import xarray as xr
import dask.array as da
import numpy as np
import pytest
from stmtools import utils
from tests.test_stm import stmat

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
