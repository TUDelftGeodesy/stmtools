import stmtools
import pandas as pd
import pytest
from pathlib import Path

path_example_csv = Path(__file__).parent / "../examples/data/example.csv"

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
        data = stmtools.from_csv(
            path_example_csv, output_chunksize={"space": 1000, "time": -1}
        )
        assert data.chunks["space"][0] == 1000
        assert data.chunks["time"][0] == 11

    def test_readcsv_custom_var_name(self):
        data = stmtools.from_csv(
            path_example_csv,
            spacetime_pattern={"^d_": "defo", "^a_": "amp", "^h2ph_": "h2ph"},
        )
        assert set(["defo", "amp"]).issubset([k for k in data.data_vars.keys()])
        assert not (
            set(["deformation", "amplitude"]).issubset(
                [k for k in data.data_vars.keys()]
            )
        )

    def test_readcsv_list_coords(self):
        data = stmtools.from_csv(path_example_csv, coords_cols=["pnt_lat", "pnt_lon"])
        assert set(["pnt_lat", "pnt_lon"]).issubset([k for k in data.coords.keys()])
        assert not (set(["lat", "lon"]).issubset([k for k in data.coords.keys()]))

    def test_readcsv_custom_coords(self):
        data = stmtools.from_csv(
            path_example_csv, coords_cols={"pnt_lat": "lat_cus", "pnt_lon": "lon_cus"}
        )
        assert set(["lat_cus", "lat_cus"]).issubset([k for k in data.coords.keys()])
        assert not (
            set(["pnt_lat", "pnt_lat"]).issubset([k for k in data.coords.keys()])
        )
        assert not (set(["lat", "lon"]).issubset([k for k in data.coords.keys()]))
