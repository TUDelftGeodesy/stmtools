from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stmtools

path_example_csv = Path(__file__).parent / "./data/example.csv"


class TestFromCSV:
    @pytest.fixture
    def example_data(self):
        return stmtools.from_csv(path_example_csv)

    def test_readcsv_dims(self, example_data):
        assert example_data.dims == {"space": 2500, "time": 11}

    def test_readcsv_vars(self, example_data):
        expected_columns = [
            "pnt_height",
            "pnt_ampconsist",
            "pnt_enscoh",
            "pnt_line",
            "pnt_id",
            "pnt_flags",
            "pnt_pixel",
            "pnt_linear",
            "pnt_demheight",
            "pnt_demheight_highres",
            "amplitude",
            "h2ph",
            "deformation",
        ]
        data_vars = [k for k in example_data.data_vars.keys()]
        assert set(data_vars) == set(expected_columns)

    def test_readcsv_output_chunksize(self):
        data = stmtools.from_csv(path_example_csv, output_chunksize={"space": 1000, "time": -1})
        assert data.chunks["space"][0] == 1000
        assert data.chunks["time"][0] == 11

    def test_readcsv_custom_var_name(self):
        data = stmtools.from_csv(
            path_example_csv,
            spacetime_pattern={"^d_": "defo", "^a_": "amp", "^h2ph_": "h2ph"},
        )
        assert set(["defo", "amp"]).issubset([k for k in data.data_vars.keys()])
        assert not (set(["deformation", "amplitude"]).issubset([k for k in data.data_vars.keys()]))

    def test_readcsv_list_coords(self):
        data = stmtools.from_csv(path_example_csv, coords_cols=["pnt_lat", "pnt_lon"])
        assert set(["pnt_lat", "pnt_lon"]).issubset([k for k in data.coords.keys()])
        assert not (set(["lat", "lon"]).issubset([k for k in data.coords.keys()]))

    def test_readcsv_custom_coords(self):
        data = stmtools.from_csv(
            path_example_csv, coords_cols={"pnt_lat": "lat_cus", "pnt_lon": "lon_cus"}
        )
        assert set(["lat_cus", "lat_cus"]).issubset([k for k in data.coords.keys()])
        assert not (set(["pnt_lat", "pnt_lat"]).issubset([k for k in data.coords.keys()]))
        assert not (set(["lat", "lon"]).issubset([k for k in data.coords.keys()]))

    def test_readcsv_wrong_pattern(self):
        with pytest.raises(ValueError):
            _ = stmtools.from_csv(
                path_example_csv, spacetime_pattern={"nonexist": "nonexist_data"}
            )

    def test_readcsv_timevalues(self, tmp_path):
        data = stmtools.from_csv(path_example_csv)
        assert data.time.values[0] == pd.Timestamp("2016-03-27T00:00:00.0")
        assert data.time.values[-1] == pd.Timestamp("2016-07-15T00:00:00.0")
        # test if time is in the right order
        assert (data.time.values[1:] - data.time.values[:-1]).min() > pd.Timedelta("0s")
        assert len(data.time.values) == 11
        assert data.time.dtype == "datetime64[ns]"

        # test if time are not in correct format
        df = pd.read_csv(path_example_csv)
        # rename a column to make it not in the right format
        df.rename(
            columns={
                "d_20160429": "d_2016/4/29",
                "a_20160429": "a_2016/4/29",
                "h2ph_20160429": "h2ph_2016/4/29",
            },
            inplace=True,
        )
        df.to_csv(tmp_path / "example.csv", index=False)
        data = stmtools.from_csv(tmp_path / "example.csv")
        assert data.time.values[0] == 0

    def test_readcsv_dtypes(self):
        data = stmtools.from_csv(path_example_csv)
        data_pd = pd.read_csv(path_example_csv)
        for key, dtype in dict(data.data_vars.dtypes).items():
            if key == "pnt_id":
                assert dtype.type is np.str_
            elif "pnt" in key:
                assert dtype.type is data_pd.dtypes[key].type
