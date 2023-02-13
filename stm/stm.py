import xarray as xr
from rasterio import features
import math
import numpy as np
import geopandas as gpd
import affine
from pathlib import Path
from shapely.strtree import STRtree
from shapely.geometry import Point
from collections.abc import Iterable
import dask.array as da

import logging

logger = logging.getLogger(__name__)

# Inspiration:
#   - https://docs.xarray.dev/en/stable/internals/extending-xarray.html
#   - https://corteva.github.io/rioxarray/html/_modules/rioxarray/raster_dataset.html


@xr.register_dataset_accessor("stm")
class SpaceTimeMatrix:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def add_metadata(self, metadata):
        # Example function to demonstrate the composition of xr.Dataset
        # Can be removed later
        self._obj = self._obj.assign_attrs(metadata)
        return self._obj.copy()

    def subset(self, method, **kwargs):
        # To be implemented
        # Spatial query, by polygon, bounding box (priority)
        # Threshold Query
        # Density Query

        match method:  # Match statements available only from python 3.10 onwards
            case "threshold":
                _check_threshold_kwargs(
                    **kwargs
                )  # Check is all required kwargs are available
                if kwargs["threshold"][0] == "<":
                    str_parts = kwargs["threshold"].partition("<")
                    check_mult_relops(
                        str_parts[2]
                    )  # Check to ensure multiple relational operators are not present
                    data_xr_subset = self._obj.where(
                        self._obj[kwargs["var"]] < float(str_parts[2]), drop=True
                    )
                elif kwargs["threshold"][0] == ">":
                    str_parts = kwargs["threshold"].partition(">")
                    check_mult_relops(str_parts[2])
                    data_xr_subset = self._obj.where(
                        self._obj[kwargs["var"]] > float(str_parts[2]), drop=True
                    )
                else:
                    raise Exception(
                        "Suitable relational operator not found! Please check input"
                    )
            case "density":
                check_density_kwargs(**kwargs)  # Check for all require kwargs
                gdf = gpd.GeoDataFrame(
                    self._obj["points"],
                    geometry=gpd.points_from_xy(
                        self._obj[kwargs["x"]], self._obj[kwargs["y"]]
                    ),
                )
                grid_cell = ((shapes) for shapes in zip(gdf.geometry, gdf.index))
                out_x = math.ceil(
                    (max(self._obj[kwargs["x"]]) - min(self._obj[kwargs["x"]]))
                    / kwargs["dx"]
                )
                out_y = math.ceil(
                    (max(self._obj[kwargs["y"]]) - min(self._obj[kwargs["y"]]))
                    / kwargs["dy"]
                )
                raster = features.rasterize(
                    shapes=grid_cell,
                    out_shape=[out_x, out_y],
                    fill=np.NAN,
                    all_touched=True,
                    default_value=1,
                    transform=affine.Affine.from_gdal(
                        min(self._obj[kwargs["x"]]),
                        kwargs["dx"],
                        0.0,
                        max(self._obj[kwargs["y"]]),
                        0.0,
                        -1 * kwargs["dy"],
                    ),
                )
                subset = [
                    item for item in np.unique(raster) if not (math.isnan(item)) == True
                ]
                data_xr_subset = self._obj.sel(points=subset)
            case "polygon":
                _check_polygon_kwargs(**kwargs)
                mask = self._obj.stm._in_polygon(
                    kwargs["polygon"], xlabel="lon", ylabel="lat"
                )
                idx = self._obj.points.data[mask.data]
                data_xr_subset = self._obj.sel(points=idx)
            case other:
                raise NotImplementedError(
                    "Method: {} is not implemented.".format(method)
                )
        chunks = {
            "points": min(
                self._obj.chunksizes["points"][0], data_xr_subset.points.shape[0]
            ),
            "time": min(self._obj.chunksizes["time"][0], data_xr_subset.time.shape[0]),
        }

        data_xr_subset = data_xr_subset.chunk(chunks)

        return data_xr_subset

    def geom_enrich(self, geom, fields, xlabel="lon", ylabel="lat"):
        # Check if coords exists
        _ = _validate_coords(self._obj, xlabel, ylabel)

        # Check if fields is a Iterable or a str
        if isinstance(fields, str):
            fields = [fields]
        elif not isinstance(fields, Iterable):
            raise ValueError("fields need to be a Iterable or a string")

        # Get geom type and the first row
        if isinstance(geom, gpd.GeoDataFrame):
            type_geom = "GeoDataFrame"
            geom_one_row = geom.iloc[0:1]
        elif isinstance(geom, Path) or isinstance(geom, str):
            type_geom = "File"
            geom_one_row = gpd.read_file(geom, rows=1)
        else:
            raise NotImplementedError("Cannot recognize the input geometry.")

        # Check if fields exists in geom
        for field in fields:
            if field not in geom_one_row.columns:
                raise ValueError(
                    'Field "{}" not found in the the input geometry'.format(field)
                )

        # Enrich all fields
        ds = self._obj
        chunks = (ds.chunksizes["points"][0],)  # Assign an empty fields to ds
        for field in fields:
            ds = ds.assign(
                {
                    field: (
                        ["points"],
                        da.from_array(np.full(ds.points.shape, None), chunks=chunks),
                    )
                }
            )
        ds = xr.map_blocks(
            _geom_enrich_block,
            ds,
            args=(geom, fields, xlabel, ylabel, type_geom),
            template=ds,
        )

        return ds

    def _in_polygon(self, geom, xlabel="lon", ylabel="lat"):
        # Check if coords exists
        _ = _validate_coords(self._obj, xlabel, ylabel)

        # Get geom type and the first row
        if isinstance(geom, gpd.GeoDataFrame):
            type_geom = "GeoDataFrame"
        elif isinstance(geom, Path) or isinstance(geom, str):
            type_geom = "File"
        else:
            raise NotImplementedError("Cannot recognize the input geometry.")

        # Enrich all fields
        ds = self._obj
        chunks = (ds.chunksizes["points"][0],)  # Assign an empty fields to ds
        ds = ds.assign(
            {
                "mask": (
                    ["points"],
                    da.from_array(np.full(ds.points.shape, False), chunks=chunks),
                )
            }
        )
        mask = ds["mask"]
        mask = xr.map_blocks(
            _in_polygon_block,
            mask,
            args=(geom, xlabel, ylabel, type_geom),
            template=mask,
        )

        return mask


def _in_polygon_block(mask, geom, xlabel, ylabel, type_geom):
    match_list, _ = _ml_str_query(mask, geom, xlabel, ylabel, type_geom)
    intmid = np.unique(match_list[:, 1])  # incase overlapping polygons
    mask.data[intmid] = True

    return mask


def _geom_enrich_block(ds, geom, fields, xlabel, ylabel, type_geom):
    # Get the match list
    match_list, geom = _ml_str_query(ds, geom, xlabel, ylabel, type_geom)

    if match_list.ndim == 2:  # geometry is an array_like
        intuids = np.unique(match_list[:, 0])
        for intuid in intuids:
            intm = np.where(match_list[:, 0] == intuid)[0]
            intmid = match_list[intm, 1]
            for field in fields:
                ds[field].data[intmid] = geom.iloc[intuid][field]
    elif match_list.ndim == 1:  # geometry is a scalar
        ds[field].data[intmid] = geom[field]

    return ds


def _ml_str_query(dsda, geom, xlabel, ylabel, type_geom):
    # Get the match list from Sort-Tile-Recursive (STR) query
    # this returns an array of two element arrays, the first entry is the positional index
    # into the list of geometries being used to query the tree. the second is the positional index
    # into the list of points for which the tree was constructed

    # Crop the geom to the bounding box of the block
    xmin, ymin, xmax, ymax = [
        dsda[xlabel].data.min(),
        dsda[ylabel].data.min(),
        dsda[xlabel].data.max(),
        dsda[ylabel].data.max(),
    ]
    match type_geom:
        case "GeoDataFrame":
            geom = geom.clip_by_rect(xmin, ymin, xmax, ymax)
        case "File":
            geom = gpd.read_file(geom, bbox=(xmin, ymin, xmax, ymax))

    # Build STR tree for points
    pnttree = STRtree(
        gpd.GeoSeries(map(Point, zip(dsda[xlabel].data, dsda[ylabel].data)))
    )

    match_list = pnttree.query(geom.geometry, predicate="contains").T

    return match_list, geom


def check_mult_relops(string):
    relops = ["<", ">"]
    for i in relops:
        if i in string:
            raise Exception("Multiple relational operators found! Please check input")


def _check_threshold_kwargs(**kwargs):
    req_kwargs = ["var", "threshold"]
    for i in req_kwargs:
        if i not in kwargs:
            raise Exception("Missing expected keyword argument: %s" % i)


def _check_polygon_kwargs(**kwargs):
    req_kwargs = ["polygon"]
    for i in req_kwargs:
        if i not in kwargs:
            raise Exception("Missing expected keyword argument: %s" % i)


def check_density_kwargs(**kwargs):
    req_kwargs = ["x", "y", "dx", "dy"]
    for i in req_kwargs:
        if i not in kwargs:
            raise Exception("Missing expected keyword argument: %s" % i)
        if i in ["dx", "dy"]:
            if not isinstance(kwargs[i], float):
                raise Exception(
                    "Keyword argument %s should be an floating point number" % i
                )


def _validate_coords(ds, xlabel, ylabel):
    # Check if dataset has xlabel and ylabel

    for clabel in [xlabel, ylabel]:
        if clabel not in ds.coords.keys():
            if clabel in ds.data_vars.keys():
                logger.warning(
                    '"{}"was not found in coordinates, but in data variables. '
                    "We will proceed with the data variable. "
                    'Please consider registering "{}" in the coordinates using '
                    '"xarray.Dataset.assign".'.format(clabel, clabel)
                )
                return 2
            else:
                raise ValueError('Coordinate label "{}" was not found.'.format(clabel))
    return 1
