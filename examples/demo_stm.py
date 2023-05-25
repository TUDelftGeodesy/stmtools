


from dask.distributed import Client
import numpy as np
from pathlib import Path
import sarxarray
from matplotlib import pyplot as plt
import xarray as xr
import stm
import geopandas as gpd

## Setup processing
# SLC stack processed by Doris V5
path = Path('/project/caroline/Share/stacks/nl_veenweiden_s1_asc_t088/stack')

# Data file in each folder
f_slc = 'cint_srd.raw'

# Geo referenced coordinates
f_lat = [path/'20210730'/'phi.raw']
f_lon = [path/'20210730'/'lam.raw']

# Metadata of the stack
shape=(10018, 68656)
dtype = np.dtype([('re', np.float32), ('im', np.float32)])

# Reading chunk size
reading_chunks = (2000,2000)

# Subset size, processing only a subset of the data
azimuth_subset = range(2000, 6000)
range_subset = range(14000, 22000)

# Flag for zarr overwrite
overwrite = True

# Zarr storage for STM
path_stm = Path('./stm.zarr')

# Path to the BRP polygon of NL
# Need a abs path for cluster processing
path_polygon = Path('/project/caroline/Public/demo_sarxarray/data/brp/brpgewaspercelen_concept_2022_wgs84.gpkg')

if __name__ == "__main__":

    ## Data loading
    # Build slcs lists
    list_slcs = [p/f_slc for p in path.rglob(('[0-9]' * 8)) if not p.match('20200325')]
    list_slcs = list_slcs[0:100] # 100 images for example

    # Load complex data
    stack = sarxarray.from_binary(list_slcs, shape, dtype=dtype, chunks=reading_chunks)

    # Load coordinates and assign to stack
    lat = sarxarray.from_binary(f_lat, shape, vlabel="lat", dtype=np.float32, chunks=reading_chunks)
    lon = sarxarray.from_binary(f_lon, shape, vlabel="lon", dtype=np.float32, chunks=reading_chunks)
    stack = stack.assign_coords(lat = (("azimuth", "range"), lat.squeeze().lat.data), lon = (("azimuth", "range"), lon.squeeze().lon.data))

    ## Select a subset
    stack_subset = stack.sel(azimuth=azimuth_subset, range=range_subset)

    ## Mean reflection map
    mrm = stack_subset.slcstack.mrm()
    mrm = mrm.compute()

    fig, ax = plt.subplots()
    ax.imshow(mrm)
    ax.set_aspect(2)
    im = mrm.plot(ax=ax, cmap='gray')
    im.set_clim([0, 40000])
    fig.savefig('mrm.png')

    ## Point selection
    stack_subset2 = stack_subset.sel(azimuth=range(20_00,40_00), range=range(180_00,200_00))
    stmat = stack_subset2.slcstack.point_selection(threshold=2, method="amplitude_dispersion",chunks=20_000)

    fig, ax = plt.subplots()
    plt.scatter(stmat.lon.data, stmat.lat.data, s=0.005)
    fig.savefig('mrm.png')

    # Export point selection to Zarr
    if overwrite:
        stmat.to_zarr(path_stm, mode="w")
    else:
        if not path_stm.exists():
            stmat.to_zarr(path_stm)

    ## Loading SpaceTime Matrix from Zarr
    stm_demo = xr.open_zarr(path_stm)
    
    ## STM enrichment from Polygon file
    # Read one row and check columns
    polygons_one_row = gpd.read_file(path_polygon, rows=1)
    xmin, ymin, xmax, ymax = [
            stm_demo['lon'].data.min().compute(),
            stm_demo['lat'].data.min().compute(),
            stm_demo['lon'].data.max().compute(),
            stm_demo['lat'].data.max().compute(),
        ]
    polygons = gpd.read_file(path_polygon, bbox=(xmin, ymin, xmax, ymax))
    polygons.plot()

    # Data enrichment
    fields_to_query = ['gewas', 'gewascode']
    stm_demo = stm_demo.stm.enrich_from_polygon(polygons, fields_to_query)

    ## Subset by Polygons
    stm_demo_subset = stm_demo.stm.subset(method='polygon', polygon=path_polygon)
    gewascode = stm_demo_subset['gewascode'].compute()


    # Convert gewascode to classes
    idx = 1
    classes = gewascode
    for v in np.unique(gewascode):
        classes[np.where(gewascode==v)] = idx
        idx+=1


    ## Visualize the croptype code
    import matplotlib.cm as cm
    colormap = cm.jet
    fig, ax = plt.subplots()
    plt.scatter(stm_demo_subset.lon.data, stm_demo_subset.lat.data, c=classes, s=0.003, cmap=colormap)
    plt.colorbar()
    fig.savefig('crop_classes.png')

