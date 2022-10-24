import xarray as xr


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

    def subset(self, **kwargs):
        # To be implemented
        # Spatial query, by polygon, bounding box (priority) 
        # Threshold Query
        # Density Query

        return None

    def from_stack(self, stack_obj):
        # Make  from a Stack Object
        # EXMAPLE of a stack object: https://bitbucket.org/grsradartudelft/rippl/src/main/rippl/SAR_sensors/sentinel/sentinel_stack.py

        return None

    def query_polygon(self, polygon, field):
        # query a field in the polygon with location of the points
        # return the field value for each point

        return None



        
 


  
