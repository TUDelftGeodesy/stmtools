"""space-time matrix module."""

import logging
from collections.abc import Iterable
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pymorton as pm
import xarray as xr
from scipy.spatial import KDTree
from shapely.geometry import Point
from shapely.strtree import STRtree

from stmtools.metadata import DataVarTypes, STMMetaData
from stmtools.utils import _has_property

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor("stm")
class SpaceTimeMatrix:
    """Space-Time Matrix."""

    def __init__(self, xarray_obj):
        """Init."""
        self._obj = xarray_obj

    def add_metadata(self, metadata):
        """Assign metadata to the STM.

        Parameters
        ----------
        metadata : str or dict
            input metadata

        Returns
        -------
        xarray.Dataset
            STM with assigned attributes.

        """
        self._obj = self._obj.assign_attrs(metadata)
        return self._obj

    def regulate_dims(self, space_label=None, time_label=None):
        """Regulate the dimension of a Space-Time Matrix instance.

        An STM should have two dimensions: "space" and "time".

        If the inupt argument `space_label` or `time_label` is specified,
        and that dimension exists, the function will rename that dimension to "space" or "time".

        If either `space_label` or `time_label` are None, a "space" or "time" dimension with
        size 1 will be created.

        If both `space_label` or `time_label` are None. Data variables will also be regulated.

        For data variables with a name started with "pnt_", they are regared as
        point-only attribute and will not be affected by "time" dimension expansion.

        Parameters
        ----------
        space_label : str, optional
            Dimension to be renamed as "space", by default None.
        time_label : _type_, optional
            Dimension to be renamed as "time", by default None.

        Returns
        -------
        xarray.Dataset
            Regulated STM.

        """
        if (
            (space_label is None)
            and (time_label is None)
            and all([k not in self._obj.dims.keys() for k in ["space", "time"]])
        ):
            raise ValueError(
                'No "space" nor "time" dimension found. \
                You should specify either "space_label" or ""time_label'
            )

        # Check time dimension
        ds_reg = self._obj
        for key, label in zip(["space", "time"], [space_label, time_label], strict=True):
            if key not in self._obj.dims.keys():
                if label is None:
                    ds_reg = ds_reg.expand_dims({key: 1})
                elif isinstance(label, str):
                    ds_reg = ds_reg.rename_dims({label: key})
                else:
                    raise ValueError(f'"{key}" dimension label should be a string.')

        # Re-order dimensions
        ds_reg = ds_reg.transpose("space", "time")

        # Squeeze the time dimension for all point attibutes, if exists
        pnt_vars = [var for var in ds_reg.data_vars.keys() if var.startswith("pnt_")]
        for var in pnt_vars:
            if "time" in ds_reg[var].dims:
                ds_reg[var] = ds_reg[var].squeeze(dim="time")

        return ds_reg

    def subset(self, method: str, **kwargs):
        """Select a subset of the STM.

        Parameters
        ----------
        method : str
            Method of subsetting. Choose from "threshold", "density" and "polygon".
            - threshold: select all space entries with a threshold criterion, e.g.
                data_xr.stm.subset(method="threshold", var="thres", threshold='>1')
            - density: select one point in every [dx, dy] cell, e.g.
                data_xr.stm.subset(method="density", dx=0.1, dy=0.1)
            - polygon: select all space entries inside a given polygon, e.g.
                data_xr.stm.subset(method='polygon', polygon=path_polygon_file)
                or
                import geopandas as gpd
                polygon = gpd.read_file(path_polygon_file)
                data_xr.stm.subset(method='polygon', polygon=polygon)
        **kwargs:
            - when method="threshold": data variable "var" and  threshold "threshold"
            - when method="density": x and y density size: "dx" and "dy"
            - when method="polygon": polygon geodataframe instance or file: "polygon"

        Returns
        -------
        xarray.Dataset
            A subset of the original STM.

        """
        # Check if both "space" and "time" dimension exists
        for dim in ["space", "time"]:
            if dim not in self._obj.sizes.keys():
                raise KeyError(
                    f'Missing dimension: "{dim}". \
                    You can use the function ".regulate_dim()" to add it.'
                )

        match method:  # Match statements available only from python 3.10 onwards
            case "threshold":
                _check_threshold_kwargs(**kwargs)  # Check is all required kwargs are available
                if kwargs["threshold"][0] == "<":
                    str_parts = kwargs["threshold"].partition("<")
                    _check_mult_relops(
                        str_parts[2]
                    )  # Check to ensure multiple relational operators are not present
                    idx = (self._obj[kwargs["var"]] < float(str_parts[2])).compute()
                    data_xr_subset = self._obj.where(idx, drop=True)
                elif kwargs["threshold"][0] == ">":
                    str_parts = kwargs["threshold"].partition(">")
                    _check_mult_relops(str_parts[2])
                    idx = (self._obj[kwargs["var"]] > float(str_parts[2])).compute()
                    data_xr_subset = self._obj.where(idx, drop=True)
                else:
                    raise ValueError("Suitable relational operator not found! Please check input")
            case "polygon":
                _check_polygon_kwargs(**kwargs)
                if "xlabel" not in kwargs:
                    keyx = "lon"
                else:
                    keyx = kwargs["xlabel"]
                if "ylabel" not in kwargs:
                    keyy = "lat"
                else:
                    keyy = kwargs["ylabel"]
                mask = self._obj.stm._in_polygon(kwargs["polygon"], xlabel=keyx, ylabel=keyy)
                idx = self._obj.space.data[mask.data]
                data_xr_subset = self._obj.sel(space=idx)
            case _:
                raise NotImplementedError(f"Method: {method} is not implemented.")
        chunks = {
            "space": min(self._obj.chunksizes["space"][0], data_xr_subset.space.shape[0]),
            "time": min(self._obj.chunksizes["time"][0], data_xr_subset.time.shape[0]),
        }

        data_xr_subset = data_xr_subset.chunk(chunks)

        return data_xr_subset

    def enrich_from_polygon(self, polygon, fields, xlabel="lon", ylabel="lat"):
        """Enrich the SpaceTimeMatrix from one or more attribute fields of a (multi-)polygon.

        Each attribute in fields will be assigned as a data variable to the STM.

        If a point of the STM falls into the given polygon, the value of the specified field will
        be added.

        For space entries outside the (multi-)polygon, the value will be None.

        Parameters
        ----------
        polygon : geopandas.GeoDataFrame, str, or pathlib.Path
            Polygon or multi-polygon with contextual information for enrichment
        fields : str or list of str
            Field name(s) in the (multi-)polygon for enrichment
        xlabel : str, optional
            Name of the x-coordinates of the STM, by default "lon"
        ylabel : str, optional
            Name of the y-coordinates of the STM, by default "lat"

        Returns
        -------
        xarray.Dataset
            Enriched STM.

        """
        _ = _validate_coords(self._obj, xlabel, ylabel)

        # Check if fields is a Iterable or a str
        if isinstance(fields, str):
            fields = [fields]
        elif not isinstance(fields, Iterable):
            raise ValueError("fields need to be a Iterable or a string")

        # Get polygon type and the first row
        if isinstance(polygon, gpd.GeoDataFrame):
            type_polygon = "GeoDataFrame"
            polygon_one_row = polygon.iloc[0:1]
        elif isinstance(polygon, Path | str):
            type_polygon = "File"
            polygon_one_row = gpd.read_file(polygon, rows=1)
        else:
            raise NotImplementedError("Cannot recognize the input polygon.")

        # Check if fields exists in polygon
        for field in fields:
            if field not in polygon_one_row.columns:
                raise ValueError(f'Field "{field}" not found in the the input polygon')

        # Enrich all fields
        ds = self._obj
        chunks = (ds.chunksizes["space"][0],)  # Assign an empty fields to ds
        for field in fields:
            ds = ds.assign(
                {
                    field: (
                        ["space"],
                        da.from_array(np.full(ds.space.shape, None), chunks=chunks),
                    )
                }
            )
        ds = xr.map_blocks(
            _enrich_from_polygon_block,
            ds,
            args=(polygon, fields, xlabel, ylabel, type_polygon),
            template=ds,
        )

        return ds

    def _in_polygon(self, polygon, xlabel="lon", ylabel="lat"):
        """Test if a space entry of a STM is inside a given (multi-polygon).

        Return result as a boolean Dask array.

        Parameters
        ----------
        polygon : geopandas.GeoDataFrame, str, or pathlib.Path
            Polygon or multi-polygon for query
        xlabel : str, optional
            Name of the x-coordinates of the STM, by default "lon"
        ylabel : str, optional
            Name of the y-coordinates of the STM, by default "lat"

        Returns
        -------
        Dask.array
            A boolean Dask array. True where a space entry is inside the (multi-)polygon.

        """
        # Check if coords exists
        _ = _validate_coords(self._obj, xlabel, ylabel)

        # Get polygon type and the first row
        if isinstance(polygon, gpd.GeoDataFrame):
            type_polygon = "GeoDataFrame"
        elif isinstance(polygon, Path | str):
            type_polygon = "File"
        else:
            raise NotImplementedError("Cannot recognize the input polygon.")

        # Enrich all fields
        ds = self._obj
        chunks = (ds.chunksizes["space"][0],)  # Assign an empty fields to ds
        ds = ds.assign(
            {
                "mask": (
                    ["space"],
                    da.from_array(np.full(ds.space.shape, False), chunks=chunks),
                )
            }
        )
        mask = ds["mask"]
        mask = xr.map_blocks(
            _in_polygon_block,
            mask,
            args=(polygon, xlabel, ylabel, type_polygon),
            template=mask,
        )

        return mask

    def register_metadata(self, dict_meta: STMMetaData):
        """Register metadata.

        Parameters
        ----------
        dict_meta : STMMetaData
            Metatdata dictionaries. The schema is pre-defined as annotation.

        Returns
        -------
        xarray.Dataset
            STM with registered metadata.

        """
        ds_updated = self._obj.assign_attrs(dict_meta)

        return ds_updated

    def register_datatype(self, keys: str | Iterable, datatype: DataVarTypes):
        """Register the specified data variables as an attribute.

        Parameters
        ----------
        keys : Union[str, Iterable]
            Keys of the data variables to register
        datatype : str in DataVarTypes
            String of the datatype. Choose from ["obsData", "auxData", "pntAttrib", "epochAttrib"].

        Returns
        -------
        xarray.Dataset
            STM with registered metadata.

        """
        ds_updated = self._obj

        if isinstance(keys, str):
            keys = [keys]
        if _has_property(ds_updated, keys):
            ds_updated = ds_updated.assign_attrs({datatype: keys})
        else:
            raise ValueError("Not all given keys are data_vars of the STM.")
        return ds_updated

    def get_order(self, xlabel="azimuth", ylabel="range", xscale=1.0, yscale=1.0):
        """Compute an ordering on the points based on coordinates with xlabel and ylabel.

        This order is stored in a (new) point attribute "order".

        Note that this ordering is most intuitive for integer coordinates (e.g. pixel coordinates).
        For float coordinates (e.g. lat-lon), the coordinates should be scaled to determine the
        resolution of the ordering: only the whole-number part influences the order.
        While coordinates could also be offset, this has limited effect on the relative order.

        Parameters
        ----------
        self : SpaceTimeMatrix
            space time matrix to order
        xlabel : str
            Name of x coordinates to order by
        ylabel : str
            Name of y coordinates to order by
        xscale : float
            Scaling multiplier to the x coordinates before truncating them to integer values.
        yscale : float
            Scaling multiplier to the y coordinates before truncating them to integer values.

        """
        meta_arr = np.array((), dtype=np.int64)
        order = da.apply_gufunc(
            _compute_morton_code,
            "(),()->()",
            xscale * self._obj[xlabel].data,
            yscale * self._obj[ylabel].data,
            meta=meta_arr,
        )
        self._obj = self._obj.assign({"order": (("space"), order)})
        return self._obj

    def reorder(self, xlabel="azimuth", ylabel="range", xscale=1.0, yscale=1.0):
        """Compute and apply an ordering on the points based on coordinates with xlabel and ylabel.

        Note that this ordering is most intuitive for integer coordinates (e.g. pixel coordinates).
        For float coordinates (e.g. lat-lon), the coordinates should be scaled to determine the
        resolution of the ordering: only the whole-number part influences the order.
        While coordinates could also be offset, this has limited effect on the relative order.

        Parameters
        ----------
        self : SpaceTimeMatrix
            space time matrix to order
        xlabel : str
            Name of x coordinates to order by
        ylabel : str
            Name of y coordinates to order by
        xscale : float
            Scaling multiplier to the x coordinates before truncating them to integer values.
        yscale : float
            Scaling multiplier to the y coordinates before truncating them to integer values.

        """
        self._obj = self.get_order(xlabel, ylabel, xscale, yscale)
        
        # Sorting may split up the chunks, which may interfere with later operations
        # so we immediately restore the chunk sizes.
        chunks = {"space": self._obj.chunksizes["space"][0]}
        if "time" in self._obj.chunksizes:
            chunks = {
                "space": self._obj.chunksizes["space"][0],
                "time": self._obj.chunksizes["time"][0],
            }
        self._obj = self._obj.sortby(self._obj.order)
        self._obj = self._obj.chunk(chunks)

        return self._obj

    def enrich_from_dataset(self,
                            dataset: xr.Dataset | xr.DataArray,
                            fields: str | Iterable,
                            method="nearest",
                            ) -> xr.Dataset:
        """Enrich the SpaceTimeMatrix from one or more fields of a dataset.

        scipy is required. if dataset is raster, it uses
        _enrich_from_raster_block to do interpolation using method. if dataset
        is point, it uses _enrich_from_points_block to find the nearest points
        in space and time using Euclidean distance.

        Parameters
        ----------
        dataset : xarray.Dataset | xarray.DataArray
            Input data for enrichment
        fields : str or list of str
            Field name(s) in the dataset for enrichment
        method : str, optional
            Method of interpolation, by default "nearest", see
            https://docs.xarray.dev/en/stable/generated/xarray.Dataset.interp.html

        Returns
        -------
        xarray.Dataset
            Enriched STM.

        """
        # Check if fields is a Iterable or a str
        if isinstance(fields, str):
            fields = [fields]
        elif not isinstance(fields, Iterable):
            raise ValueError("fields need to be a Iterable or a string")

        # if dataset is a DataArray, convert it to a Dataset
        if isinstance(dataset, xr.DataArray):
            dataset = dataset.to_dataset()

        ds = self._obj
        # check if both dataset and ds have coords_labels keys
        for coord_label in ds.coords.keys():
            if coord_label not in dataset.coords.keys():
                raise ValueError(
                    f'Coordinate label "{coord_label}" was not found in the input dataset.'
                    )

        # check if dataset is point or raster if 'space' in dataset.dims:
        if "space" in dataset.dims:
            approch = "point"
        elif "lat" in dataset.dims and "lon" in dataset.dims:
            approch = "raster"
        elif "y" in dataset.dims and "x" in dataset.dims:
            approch = "raster"
        else:
            raise ValueError(
                "The input dataset is not a point or raster dataset."
                "The dataset should have either 'space' or 'lat/y' and 'lon/x' dimensions."
                "Consider renaming using "
                "https://docs.xarray.dev/en/latest/generated/xarray.Dataset.rename.html#xarray-dataset-rename"
                             )

        # check if dataset has time dimensions
        if "time" not in dataset.dims:
            raise ValueError('Missing dimension: "time" in the input dataset.')

        # check if dtype of time is the same
        if dataset.time.dtype != ds.time.dtype:
            raise ValueError("The input dataset and the STM have different time dtype.")

        # TODO: check if both ds and dataset has same coordinate system

        for field in fields:

            # check if dataset has the fields
            if field not in dataset.data_vars.keys():
                raise ValueError(f'Field "{field}" not found in the the input dataset')

            # check STM has the filed already
            if field in ds.data_vars.keys():
                raise ValueError(f'Field "{field}" already exists in the STM.')
            # TODO: overwrite the field in the STM

        if approch == "raster":
            return _enrich_from_raster_block(ds, dataset, fields, method)
        elif approch == "point":
            return _enrich_from_points_block(ds, dataset, fields)

    @property
    def num_points(self):
        """Get number of space entry of the stm.

        Returns
        -------
        int
            Number of space entry.

        """
        return self._obj.dims["space"]

    @property
    def num_epochs(self):
        """Get number of epochs of the stm.

        Returns
        -------
        int
            Number of epochs.

        """
        return self._obj.dims["time"]


def _in_polygon_block(mask, polygon, xlabel, ylabel, type_polygon):
    """Block-wise function for "_in_polygon"."""
    match_list, _ = _ml_str_query(mask[xlabel], mask[ylabel], polygon, type_polygon)
    intmid = np.unique(match_list[:, 1])  # incase overlapping polygons
    mask.data[intmid] = True

    return mask


def _enrich_from_polygon_block(ds, polygon, fields, xlabel, ylabel, type_polygon):
    """Block-wise function for "enrich_from_polygon"."""
    # Get the match list
    match_list, polygon = _ml_str_query(ds[xlabel], ds[ylabel], polygon, type_polygon)

    _ds = ds.copy(deep=True)

    if match_list.ndim == 2:
        intuids = np.unique(match_list[:, 0])
        for intuid in intuids:
            intm = np.where(match_list[:, 0] == intuid)[0]
            intmid = match_list[intm, 1]
            for field in fields:
                _ds[field].data[intmid] = polygon.iloc[intuid][field]

    return _ds


def _ml_str_query(xx, yy, polygon, type_polygon):
    """Test if a set of space entries is inside a (multi-)polygon.

    Sort-Tile-Recursive (STR) query is used, return the match list.

    Parameters
    ----------
    xx : array_like
        Vector array of the x coordinates
    yy : array_like
        Vector array of the y coordinates
    polygon : geopandas.GeoDataFrame, str, or pathlib.Path
        Polygon or multi-polygon for query
    type_polygon : str
        Choose from "GeoDataFrame" or "File"

    Returns
    -------
    array_like
        An array with two columns. The first column is the positional index into the list of
        polygons being used to query the tree. The second column is the positional index into
        the list of space entries for which the tree was constructed.

    """
    # Crop the polygon to the bounding box of the block
    xmin, ymin, xmax, ymax = [
        xx.data.min(),
        yy.data.min(),
        xx.data.max(),
        yy.data.max(),
    ]
    match type_polygon:
        case "GeoDataFrame":
            polygon = polygon.cx[xmin:xmax, ymin:ymax]
        case "File":
            polygon = gpd.read_file(polygon, bbox=(xmin, ymin, xmax, ymax))

    # Build STR tree
    pnttree = STRtree(gpd.GeoSeries(map(Point, zip(xx.data, yy.data, strict=True))))

    match_list = pnttree.query(polygon.geometry, predicate="contains").T

    return match_list, polygon


def _check_mult_relops(string):
    relops = ["<", ">"]
    for i in relops:
        if i in string:
            raise ValueError("Multiple relational operators found! Please check input")


def _check_threshold_kwargs(**kwargs):
    req_kwargs = ["var", "threshold"]
    for i in req_kwargs:
        if i not in kwargs:
            raise ValueError(f"Missing expected keyword argument: {i}")


def _check_polygon_kwargs(**kwargs):
    req_kwargs = ["polygon"]
    for i in req_kwargs:
        if i not in kwargs:
            raise ValueError(f"Missing expected keyword argument: {i}")


def _validate_coords(ds, xlabel, ylabel):
    """Check if dataset has coordinates xlabel and ylabel.

    Parameters
    ----------
    ds : xarray.dataset
        dataset to query
    xlabel : str
        Name of x coordinates
    ylabel : str
        Name of y coordinates

    Returns
    -------
    int
        If xlabel and ylabel are in the coordinates of ds, return 1.
        If the they are in the data variables, return 2 and raise a warning

    Raises
    ------
    ValueError
        If xlabel or ylabel neither exists in coordinates, raise ValueError

    """
    for clabel in [xlabel, ylabel]:
        if clabel not in ds.coords.keys():
            if clabel in ds.data_vars.keys():
                logger.warning(
                    f'"{clabel}"was not found in coordinates, but in data variables. '
                    "We will proceed with the data variable. "
                    f'Please consider registering "{clabel}" in the coordinates using '
                    '"xarray.Dataset.assign".'
                )
                return 2
            else:
                raise ValueError(f'Coordinate label "{clabel}" was not found.')
    return 1


def _compute_morton_code(xx, yy):
    """Compute Morton codes based on two coordinates.

    The Morton codes are computed using pymorton interleave on integer values.

    Parameters
    ----------
    xx : array_like
        Vector array of the x coordinates
    yy : array_like
        Vector array of the y coordinates

    Returns
    -------
    array_like
        An array with Morton codes per coordinate pair.

    """
    code = [pm.interleave(int(xi), int(yi)) for xi, yi in zip(xx, yy, strict=True)]
    return code


def _enrich_from_raster_block(ds, dataraster, fields, method):
    """Enrich the ds (SpaceTimeMatrix) from one or more fields of a raster dataset.

    scipy is required. It uses xarray.Dataset.interp_like to interpolate the
    raster dataset to the coordinates of ds.
    https://docs.xarray.dev/en/stable/generated/xarray.Dataset.interp.html

    Parameters
    ----------
    ds : xarray.Dataset
        SpaceTimeMatrix to enrich
    dataraster : xarray.Dataset | xarray.DataArray
        Input data for enrichment
    fields : str or list of str
        Field name(s) in the dataset for enrichment
    method : str, optional
        Method of interpolation, by default "nearest", see

    Returns
    -------
    xarray.Dataset

    """
    # interpolate the raster dataset to the coordinates of ds
    interpolated = dataraster.interp(ds.coords, method=method)

    # Assign these values to the corresponding points in ds
    for field in fields:
        ds[field] = xr.DataArray(interpolated[field].data, dims=ds.dims, coords=ds.coords)
    return ds


def _enrich_from_points_block(ds, datapoints, fields):
    """Enrich the ds (SpaceTimeMatrix) from one or more fields of a point dataset.

    Assumption is that dimensions of data are space and time.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree

    Parameters
    ----------
    ds : xarray.Dataset
        SpaceTimeMatrix to enrich
    datapoints : xarray.Dataset | xarray.DataArray
        Input data for enrichment
    fields : str or list of str
        Field name(s) in the dataset for enrichment

    Returns
    -------
    xarray.Dataset

    """
    # The reason that we use KDTRee instead of xarray.unstack is that the latter
    # is slow for large datasets

    # check the dimensions
    indexer = {}
    for dim in ["space", "time"]:
        if dim not in datapoints.coords:
            indexer[dim]= [
                coord for coord in datapoints.coords if dim in datapoints[coord].dims
                ]
        else:
            indexer[dim] = [dim]

    ## datapoints
    indexes = [datapoints[coord] for coord in indexer["space"]]
    dataset_points_coords = np.column_stack(indexes)

    # ds
    indexes = [ds[coord] for coord in indexer["space"]]
    ds_coords = np.column_stack(indexes)

    # Create a KDTree object for the spatial coordinates of datapoints
    # Find the indices of the nearest points in space in datapoints for each point in ds
    # it uses Euclidean distance
    tree = KDTree(dataset_points_coords)
    _, indices_space = tree.query(ds_coords)

    # Create a KDTree object for the temporal coordinates of datapoints
    # Find the indices of the nearest points in time in datapoints for each point in ds
    datapoints_times = datapoints.time.values.reshape(-1, 1)
    ds_times = ds.time.values.reshape(-1, 1)
    tree = KDTree(datapoints_times)
    _, indices_time = tree.query(ds_times)

    selections = datapoints.isel(space=indices_space, time=indices_time)

    # Assign these values to the corresponding points in ds
    for field in fields:
        ds[field] = xr.DataArray(
            selections[field].data, dims=ds.dims, coords=ds.coords
            )

    return ds
