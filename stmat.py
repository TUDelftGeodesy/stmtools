import xarray as xr


# Inspiration: 
#   - https://docs.xarray.dev/en/stable/internals/extending-xarray.html
#   - https://corteva.github.io/rioxarray/html/_modules/rioxarray/raster_dataset.html


@xr.register_dataset_accessor("stmat")
class STMat:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def add_metadata(self, metadata):
        self._obj = self._obj.assign_attrs(metadata)
        return self._obj.copy()
        
 


  
