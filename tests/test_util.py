import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stmtools import utils


class TestHasProperty:
    def test_has_property_str(self, stmat):
        assert utils._has_property(stmat, "phase")

    def test_has_property_list(self, stmat):
        assert utils._has_property(stmat, ["phase", "amplitude"])

    def test_has_property_tuple(self, stmat):
        assert utils._has_property(stmat, ("phase", "amplitude"))

    def test_has_not(self, stmat):
        assert not (utils._has_property(stmat, ["phase", "no_exists"]))

    def test_incorrect_type(self, stmat):
        with pytest.raises(ValueError):
            utils._has_property(stmat, 1)

@pytest.fixture
def meteo_points():
    n_times = 20
    n_locations = 50
    lon_values = np.arange(n_locations)
    lat_values = np.arange(n_locations)
    time_values = pd.date_range(start='2021-01-01', periods=n_times)
    data = da.arange(n_locations * n_times).reshape((n_locations, n_times))

    return xr.Dataset(
        data_vars=dict(
            temperature=(["space", "time"], data),
        ),
        coords=dict(
            lon=(["space"], lon_values),
            lat=(["space"], lat_values),
            time=(["time"], time_values),
        ),
    ).unify_chunks()


@pytest.fixture
def meteo_raster():
    n_times = 20
    n_locations = 50
    lon_values = np.arange(n_locations)
    lat_values = np.arange(n_locations)
    time_values = pd.date_range(start='2021-01-01', periods=n_times)
    # add x and y values
    x_values = np.arange(n_locations)
    y_values = np.arange(n_locations)
    data = da.arange(n_locations * n_locations * n_times).reshape(
        (n_locations, n_locations, n_times)
        )

    return xr.Dataset(
        data_vars=dict(
            temperature=(["lon", "lat", "time"], data),
        ),
        coords=dict(
            lon=(["lon"], lon_values),
            lat=(["lat"], lat_values),
            x=(["lon"], x_values),
            y=(["lat"], y_values),
            time=(["time"], time_values),
        ),
    ).unify_chunks()

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
            pnt_height=(
                ["space"],
                da.arange(npoints),
            ),
        ),
        coords=dict(
            lon=(["space"], da.arange(npoints)),
            lat=(["space"], da.arange(npoints)),
            time=(["time"], pd.date_range(start='2021-01-02', periods=ntime)),
        ),
    ).unify_chunks()

class TestCrop:
    def test_crop_points(self, stmat, meteo_points):
        buffer = {"lon": 1, "lat": 1, "time": pd.Timedelta("1D")}
        cropped = utils.crop(stmat, meteo_points, buffer)
        # check min and max values of coordinates
        assert cropped.lon.min() == 0
        assert cropped.lon.max() == 10
        assert cropped.lat.min() == 0
        assert cropped.lat.max() == 10
        assert cropped.time.min() == pd.Timestamp("2021-01-01")
        assert cropped.time.max() == pd.Timestamp("2021-01-07")
        assert tuple(cropped.temperature.dims) == ("space", "time")
        assert "lon" in cropped.coords
        assert "lat" in cropped.coords
        assert "time" in cropped.coords

    def test_crop_raster(self, stmat, meteo_raster):
        buffer = {"lon": 1, "lat": 1, "time": 1}
        cropped = utils.crop(stmat, meteo_raster, buffer)
        # check min and max values of coordinates
        assert cropped.lon.min() == 0
        assert cropped.lon.max() == 10
        assert cropped.lat.min() == 0
        assert cropped.lat.max() == 10
        assert cropped.time.min() == pd.Timestamp("2021-01-02")
        assert cropped.time.max() == pd.Timestamp("2021-01-06")
        assert tuple(cropped.dims) == ("lon", "lat", "time")
        assert "lon" in cropped.coords
        assert "lat" in cropped.coords
        assert "time" in cropped.coords
        assert "x" in cropped.coords
        assert "y" in cropped.coords


    def test_all_operations_lazy(self, stmat, meteo_raster):
        buffer = {"lon": 1, "lat": 1, "time": 1}
        cropped = utils.crop(stmat, meteo_raster, buffer)
        assert isinstance(cropped.temperature.data, da.Array)


class TestMonotonicCoords:
    def test_monotonic_coords(self, stmat):
        assert utils.monotonic_coords(stmat, "lon")
        assert utils.monotonic_coords(stmat, "lat")
        assert utils.monotonic_coords(stmat, "time")

    def test_non_monotonic_coords_lon(self, stmat):
        stmat["lon"][0] = 100
        assert not utils.monotonic_coords(stmat, "lon")

    def test_non_monotonic_coords_lat(self, stmat):
        stmat["lat"][0] = 100
        stmat["lat"][1] = 50
        assert not utils.monotonic_coords(stmat, "lat")

    def test_non_monotonic_coords_time(self, stmat):
        stmat["time"].values[0] = '2022-01-02T00:00:00.000000000'
        stmat["time"].values[1] = '2022-01-01T00:00:00.000000000'
        assert not utils.monotonic_coords(stmat, "time")
