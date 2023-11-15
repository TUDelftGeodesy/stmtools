from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely import geometry

from stmtools.stm import _validate_coords

path_multi_polygon = Path(__file__).parent / "./data/multi_polygon.gpkg"


@pytest.fixture
def stmat_rd():
    # A STM with rd coordinates
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
            rdx=(["space"], da.arange(npoints)),
            rdy=(["space"], da.arange(npoints)),
            time=(["time"], np.arange(ntime)),
        ),
    ).unify_chunks()


@pytest.fixture
def stmat_only_point():
    npoints = 10
    return xr.Dataset(
        data_vars=dict(
            amplitude=(["space"], da.arange(npoints)),
            phase=(["space"], da.arange(npoints)),
            pnt_height=(["space"], da.arange(npoints)),
        ),
        coords=dict(lon=(["space"], da.arange(npoints)), lat=(["space"], da.arange(npoints))),
    ).unify_chunks()


@pytest.fixture
def stmat_wrong_space_label():
    npoints = 10
    return xr.Dataset(
        data_vars=dict(
            amplitude=(["space2"], da.arange(npoints)),
            phase=(["space2"], da.arange(npoints)),
            pnt_height=(["space2"], da.arange(npoints)),
        ),
        coords=dict(lon=(["space2"], da.arange(npoints)), lat=(["space2"], da.arange(npoints))),
    ).unify_chunks()


@pytest.fixture
def polygon():
    p1 = geometry.Point(1, 1)
    p2 = geometry.Point(1, 2)
    p3 = geometry.Point(3, 3)
    p4 = geometry.Point(2, 1)
    point_list = [p1, p2, p3, p4, p1]

    data = {
        "ID": ["001"],
        "temperature": [15.24],
        "geometry": [geometry.Polygon(point_list)],
    }

    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def multi_polygon():
    # Two ploygons

    # Polygon 1
    p1 = geometry.Point(1, 1)
    p2 = geometry.Point(1, 2)
    p3 = geometry.Point(3, 3)
    p4 = geometry.Point(2, 1)
    point_list1 = [p1, p2, p3, p4, p1]

    # Polygon 2
    p1 = geometry.Point(5, 5)
    p2 = geometry.Point(5, 6)
    p3 = geometry.Point(7, 7)
    p4 = geometry.Point(6, 5)
    point_list2 = [p1, p2, p3, p4, p1]

    data = {
        "ID": ["001", "002"],
        "temperature": [15.24, 14.12],
        "geometry": [geometry.Polygon(point_list1), geometry.Polygon(point_list2)],
    }

    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def stmat_xy():
    return xr.Dataset(
        coords=dict(
            azimuth=(
                ["space"],
                da.from_array(np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])),
            ),
            range=(
                ["space"],
                da.from_array(np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])),
            ),
        ),
    ).unify_chunks()


@pytest.fixture
def stmat_morton():
    return xr.Dataset(
        coords=dict(
            azimuth=(
                ["space"],
                da.from_array(np.array([0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3])),
            ),
            range=(
                ["space"],
                da.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3])),
            ),
        ),
    ).unify_chunks()


@pytest.fixture
def stmat_lonlat():
    return xr.Dataset(
        coords=dict(
            azimuth=(
                ["space"],
                da.from_array(
                    np.array(
                        [
                            0.00,
                            0.07,
                            0.14,
                            0.21,
                            0.00,
                            0.07,
                            0.14,
                            0.21,
                            0.00,
                            0.07,
                            0.14,
                            0.21,
                            0.00,
                            0.07,
                            0.14,
                            0.21,
                        ]
                    )
                ),
            ),
            range=(
                ["space"],
                da.from_array(
                    np.array(
                        [
                            0.00,
                            0.00,
                            0.00,
                            0.00,
                            0.06,
                            0.06,
                            0.06,
                            0.06,
                            0.12,
                            0.12,
                            0.12,
                            0.12,
                            0.18,
                            0.18,
                            0.18,
                            0.18,
                        ]
                    )
                ),
            ),
        ),
    ).unify_chunks()


@pytest.fixture
def stmat_lonlat_morton():
    return xr.Dataset(
        coords=dict(
            azimuth=(
                ["space"],
                da.from_array(
                    np.array(
                        [
                            0.00,
                            0.07,
                            0.00,
                            0.07,
                            0.14,
                            0.21,
                            0.14,
                            0.21,
                            0.00,
                            0.07,
                            0.00,
                            0.07,
                            0.14,
                            0.21,
                            0.14,
                            0.21,
                        ]
                    )
                ),
            ),
            range=(
                ["space"],
                da.from_array(
                    np.array(
                        [
                            0.00,
                            0.00,
                            0.06,
                            0.06,
                            0.00,
                            0.00,
                            0.06,
                            0.06,
                            0.12,
                            0.12,
                            0.18,
                            0.18,
                            0.12,
                            0.12,
                            0.18,
                            0.18,
                        ]
                    )
                ),
            ),
        ),
    ).unify_chunks()


class TestRegulateDims:
    def test_time_dim_exists(self, stmat_only_point):
        stm_reg = stmat_only_point.stm.regulate_dims()
        assert "time" in stm_reg.dims.keys()

    def test_time_dim_size_one(self, stmat_only_point):
        stm_reg = stmat_only_point.stm.regulate_dims()
        assert stm_reg.dims["time"] == 1

    def test_time_dim_customed_label(self, stmat_wrong_space_label):
        stm_reg = stmat_wrong_space_label.stm.regulate_dims(space_label="space2")
        assert stm_reg.dims["time"] == 1
        assert stm_reg.dims["space"] == 10

    def test_pnt_time_dim_nonexists(self, stmat_only_point):
        """
        For data variable with name pattern "pnt_*", there should be no time dimension.
        """
        stm_reg = stmat_only_point.stm.regulate_dims()
        assert "time" not in stm_reg["pnt_height"].dims

    def test_subset_works_after_regulate_dims(self, stmat_only_point):
        stm_reg = stmat_only_point.stm.regulate_dims()
        stm_reg_subset = stm_reg.stm.subset(method="threshold", var="pnt_height", threshold=">5")
        assert stm_reg_subset.dims["space"] == 4

    def test_validate_coords(self):
        stmat_coords = xr.Dataset(
            data_vars=dict(
                data=(
                    ["space", "time"],
                    da.arange(5 * 10).reshape((10, 5)),
                ),
                x_coor=(["space"], np.arange(10)),
                y_coor=(["space"], np.arange(10)),
            ),
            coords=dict(
                x=(["space"], np.arange(10)),
                y=(["space"], np.arange(10)),
                time=(["time"], np.arange(5)),
            ),
        )

        assert _validate_coords(stmat_coords, "x", "y") == 1
        assert _validate_coords(stmat_coords, "x_coor", "y_coor") == 2
        assert _validate_coords(stmat_coords, "x", "y_coor") == 2

        with pytest.raises(ValueError):
            _validate_coords(stmat_coords, "x_non", "y_non")


class TestAttributes:
    def test_numpoints(self, stmat):
        assert stmat.stm.num_points == 10

    def test_numepochss(self, stmat):
        assert stmat.stm.num_epochs == 5

    def test_register_datatype(self, stmat):
        stmat_with_dtype = stmat.stm.register_datatype("pnt_height", "pntAttrib")
        assert "pnt_height" in stmat_with_dtype.attrs["pntAttrib"]

    def test_register_datatype_nonexists(self, stmat):
        with pytest.raises(ValueError):
            stmat.stm.register_datatype("non_exist", "pntAttrib")


class TestSubset:
    def test_check_missing_dimension(self, stmat_only_point):
        with pytest.raises(KeyError):
            stmat_only_point.stm.subset(method="threshold", var="pnt_height", threshold=">5")

    def test_check_missing_value(self, stmat):
        with pytest.raises(ValueError):
            stmat.stm.subset(method="threshold", var="pnt_height", threshold=">")
            stmat.stm.subset(method="threshold", var="pnt_height", threshold="<")

    def test_method_not_implemented(self, stmat):
        with pytest.raises(NotImplementedError):
            stmat.stm.subset(method="something_else")

    def test_subset_with_threshold(self, stmat):
        # dummy threshold, first three are 2, others 1
        # test selecting first 3
        v_thres = np.ones(
            stmat.space.shape,
        )
        v_thres[0:3] = 3
        stmat = stmat.assign(
            {
                "thres": (
                    ["space"],
                    da.from_array(v_thres),
                )
            }
        )
        stmat_subset_larger = stmat.stm.subset(method="threshold", var="thres", threshold=">2")
        stmat_subset_lower = stmat.stm.subset(method="threshold", var="thres", threshold="<2")
        assert stmat_subset_larger.equals(stmat.sel(space=[0, 1, 2]))
        assert stmat_subset_lower.equals(stmat.sel(space=range(3, 10, 1)))

    def test_subset_with_polygons(self, stmat, polygon):
        stmat_subset = stmat.stm.subset(method="polygon", polygon=polygon)
        assert stmat_subset.equals(stmat.sel(space=[2]))

    def test_subset_with_polygons_rd(self, stmat_rd, polygon):
        stmat_subset = stmat_rd.stm.subset(
            method="polygon", polygon=polygon, xlabel="rdx", ylabel="rdy"
        )
        assert stmat_subset.equals(stmat_rd.sel(space=[2]))

    def test_subset_with_multi_polygons(self, stmat, multi_polygon):
        stmat_subset = stmat.stm.subset(method="polygon", polygon=multi_polygon)
        assert stmat_subset.equals(stmat.sel(space=[2, 6]))

    def test_subset_with_multi_polygons_file(self, stmat):
        stmat_subset = stmat.stm.subset(method="polygon", polygon=path_multi_polygon)
        assert stmat_subset.equals(stmat.sel(space=[2, 6]))


class TestEnrichment:
    def test_enrich_one_field_one_polygon(self, stmat, polygon):
        field = polygon.columns[0]
        stmat = stmat.stm.enrich_from_polygon(polygon, field)
        assert field in stmat.data_vars

        results = stmat[field].data.compute()
        results = [res for res in results if res is not None]
        assert np.all(results == np.array(polygon[field]))

    def test_enrich_multi_fields_one_polygon(self, stmat, polygon):
        fields = ["ID", "temperature"]
        stmat = stmat.stm.enrich_from_polygon(polygon, fields)
        for field in fields:
            assert field in stmat.data_vars

            results = stmat[field].data.compute()
            results = [res for res in results if res is not None]
            assert np.all(results == np.array(polygon[field]))

    def test_enrich_one_field_multi_polygon(self, stmat, multi_polygon):
        field = multi_polygon.columns[0]
        stmat = stmat.stm.enrich_from_polygon(multi_polygon, field)
        assert field in stmat.data_vars

        results = stmat[field].data.compute()
        results = [res for res in results if res is not None]
        assert np.all(results == np.array(multi_polygon[field]))

    def test_enrich_multi_fields_multi_polygon(self, stmat, multi_polygon):
        fields = multi_polygon.columns[0:2]
        stmat = stmat.stm.enrich_from_polygon(multi_polygon, fields)
        for field in fields:
            assert field in stmat.data_vars

            results = stmat[field].data.compute()
            results = [res for res in results if res is not None]
            assert np.all(results == np.array(multi_polygon[field]))

    def test_enrich_multi_fields_multi_polygon_from_file(self, stmat):
        multi_polygon = gpd.read_file(path_multi_polygon)
        fields = multi_polygon.columns[0:2]
        stmat = stmat.stm.enrich_from_polygon(path_multi_polygon, fields)
        for field in fields:
            assert field in stmat.data_vars

            results = stmat[field].data.compute()
            results = [res for res in results if res is not None]
            assert np.all(results == np.array(multi_polygon[field]))

    def test_enrich_exetions(self, stmat, multi_polygon):
        with pytest.raises(NotImplementedError):
            # int not implemented for polygons
            stmat = stmat.stm.enrich_from_polygon(999, "field")

        with pytest.raises(ValueError):
            stmat = stmat.stm.enrich_from_polygon(multi_polygon, "non_exist_field")


class TestOrderPoints:
    def test_order_attr_exists(self, stmat_xy):
        stmat = stmat_xy.stm.get_order(xlabel="azimuth", ylabel="range")
        assert "order" in stmat.keys()
        assert type(stmat.order) == xr.DataArray

    def test_order(self, stmat_xy, stmat_morton):
        stmat = stmat_xy.stm.get_order(xlabel="azimuth", ylabel="range")
        stmat = stmat.sortby(stmat.order)

        assert stmat.azimuth.equals(stmat_morton.azimuth)
        assert stmat.range.equals(stmat_morton.range)

    def test_reorder(self, stmat_xy, stmat_morton):
        stmat = stmat_xy.stm.reorder(xlabel="azimuth", ylabel="range")

        assert stmat.azimuth.equals(stmat_morton.azimuth)
        assert stmat.range.equals(stmat_morton.range)

    def test_reorder_lonlat(self, stmat_lonlat, stmat_lonlat_morton):
        stmat_naive = stmat_lonlat.stm.reorder(xlabel="azimuth", ylabel="range")
        stmat = stmat_lonlat.stm.reorder(xlabel="azimuth", ylabel="range", xscale=15, yscale=17)

        assert not stmat_naive.azimuth.equals(stmat_lonlat_morton.azimuth)
        assert not stmat_naive.range.equals(stmat_lonlat_morton.range)
        assert stmat.azimuth.equals(stmat_lonlat_morton.azimuth)
        assert stmat.range.equals(stmat_lonlat_morton.range)
