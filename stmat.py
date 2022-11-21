import xarray as xr
from rasterio import features
from rasterio import enums
import math
import numpy as np
import geopandas as gpd
import affine


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
        match method:   #Match statements available only from python 3.10 onwards
            case 'threshold':
                check_threshold_kwargs(**kwargs) #Check is all required kwargs are available
                if kwargs['threshold'][0] == '<':
                    str_parts = kwargs['threshold'].partition('<')
                    check_mult_relops(str_parts[2]) #Check to ensure multiple relational operators are not present
                    data_xr_subset= self._obj.where(self._obj[kwargs['var']] < float(str_parts[2]), drop=True)
                elif kwargs['threshold'][0] == '>':
                    str_parts = kwargs['threshold'].partition('>')
                    check_mult_relops(str_parts[2])
                    data_xr_subset= self._obj.where(self._obj[kwargs['var']] > float(str_parts[2]), drop=True)
                else:
                    raise Exception('Suitable relational operator not found! Please check input') 
            case 'density':
                check_density_kwargs(**kwargs) #Check for all require kwargs
                gdf = gpd.GeoDataFrame(self._obj['points'], geometry=gpd.points_from_xy(self._obj[kwargs['x']],self._obj[kwargs['y']]))
                grid_cell  = ((shapes) for shapes in zip(gdf.geometry, gdf.index))
                out_x = math.ceil((max(self._obj[kwargs['x']]) - min(self._obj[kwargs['x']]))/kwargs['dx'])
                out_y = math.ceil((max(self._obj[kwargs['y']]) - min(self._obj[kwargs['y']]))/kwargs['dy'])
                raster = features.rasterize(shapes=grid_cell, out_shape=[out_x, out_y], fill=np.NAN, all_touched= True, default_value=1, transform=affine.Affine.from_gdal(min(self._obj[kwargs['x']]), kwargs['dx'], 0.0, max(self._obj[kwargs['y']]), 0.0, -1*kwargs['dy']))
                subset = [item for item in np.unique(raster) if not(math.isnan(item)) == True]
                data_xr_subset = self._obj.sel(points=subset)
        return data_xr_subset

    def from_stack(self, stack_obj):
        # Make  from a Stack Object
        # EXMAPLE of a stack object: https://bitbucket.org/grsradartudelft/rippl/src/main/rippl/SAR_sensors/sentinel/sentinel_stack.py

        return None

    def query_polygon(self, polygon, field):
        # query a field in the polygon with location of the points
        # return the field value for each point

        return None

def check_mult_relops(string):
    relops = ['<','>']
    for i in relops:
        if i in string:
            raise Exception('Multiple relational operators found! Please check input')

def check_threshold_kwargs(**kwargs):
    req_kwargs=['var','threshold']
    for i in req_kwargs:
        if i not in kwargs:
            raise Exception('Missing expected keyword argument: %s'%i)
 

def check_density_kwargs(**kwargs):
    req_kwargs=['x','y','dx','dy' ]
    for i in req_kwargs:
        if i not in kwargs:
            raise Exception('Missing expected keyword argument: %s'%i)
        if i in ['dx', 'dy']:
            if not isinstance(kwargs[i], float):
                raise Exception('Keyword argument %s should be an floating point number'%i)
  


  
