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
                check_threshold_kwargs(
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
        return data_xr_subset

    def geom_enrich(self, geom, fields, xlabel="lon", ylabel="lat"):
        # Check if coordinatelabel exists
        for clabel in [xlabel, ylabel]:
            if clabel not in self._obj.coords.keys():
                if clabel in self._obj.data_vars.keys():
                    logger.warning(
                        '"{}"was not found in coordinates, but in data variables. '
                        "We will proceed with the data variable. "
                        'Please consider registering "{}" in the coordinates using '
                        '"xarray.Dataset.assign".'.format(
                            clabel, clabel
                        )
                    )
                else:
                    raise ValueError(
                        'Coordinate label "{}" was not found.'.format(clabel)
                    )

        # Check if fields is a Iterable or a str
        if isinstance(fields, str):
            fields = [fields]
        elif not isinstance(fields, Iterable):
            raise ValueError('fields need to be a Iterable or a string')
        
        # Get geom type and the first row 
        if isinstance(geom, gpd.GeoDataFrame):
            type_geom = "GeoDataFrame"
            geom_one_row = geom.iloc[0]
        elif isinstance(geom, Path) or isinstance(geom, str):
            type_geom = "File"
            geom_one_row = gpd.read_file(geom, rows=1)
        else:
            raise NotImplementedError("Cannot recognize the input geometry.")
        
        # Check if fields exists in geom
        for field in fields:
                if field not in geom_one_row.columns:
                    raise ValueError('Field "{}" not found in the the input geometry'.format(field))

        # Enrich all fields
        chunks = (self._obj.chunksizes["points"][0],)
        ds = self._obj
        for field in fields:
            # Assign an empty field
            ds = ds.assign(
                {
                    field: (
                        ["points"],
                        da.from_array(
                            np.full(self._obj.points.shape, None), chunks=chunks
                        ),
                    )
                }
            )
            da_field = ds[field]
            da_field = da_field.map_blocks(
                _geom_enrich_block,
                args=[geom, field, xlabel, ylabel, type_geom],
                template=da_field,
            )
            ds = ds.assign({field: da_field})

        return ds


def _geom_enrich_block(da, geom, field, xlabel, ylabel, type_geom):
    # Crop the geom to the bounding box of the block
    xmin, ymin, xmax, ymax = [
        da[xlabel].data.min(),
        da[ylabel].data.min(),
        da[xlabel].data.max(),
        da[ylabel].data.max(),
    ]
    match type_geom:
        case "GeoDataFrame":
            geom = geom.clip_by_rect(xmin, ymin, xmax, ymax)
        case "File":
            geom = gpd.read_file(geom, bbox=(xmin, ymin, xmax, ymax))

    # Build STR tree for points
    pnttree = STRtree(gpd.GeoSeries(map(Point, zip(da[xlabel].data, da[ylabel].data))))

    intml = pnttree.query(geom.geometry, predicate="contains").T

    if intml.ndim == 2:  # geometry is an array_like
        intuids = np.unique(intml[:, 0])
        for intuid in intuids:
            intm = np.where(intml[:, 0] == intuid)[0]
            intmid = intml[intm, 1]
            da.data[intmid] = geom.iloc[intuid][field]
    elif intml.ndim == 1:  # geometry is a scalar
        da.data[intml] = geom[field]

    return da


def check_mult_relops(string):
    relops = ["<", ">"]
    for i in relops:
        if i in string:
            raise Exception("Multiple relational operators found! Please check input")


def check_threshold_kwargs(**kwargs):
    req_kwargs = ["var", "threshold"]
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
