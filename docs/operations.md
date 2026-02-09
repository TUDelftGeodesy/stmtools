# Operations on STM

STMTools supports various operations on an STM.

## Enrich an STM

Contextual data can be added to an STM by enrichment. STMTools supports enriching an STM by static polygons or a dataset.

### Enrich from a polygon

STMTools supports enriching an STM by static polygons. For example, if soil type
data (`soil_map.gpkg`) is available together with an STM, one can first read
`soil_map.gpkg` using the `GeoPandas` library as a `GeoDataFrame`, then add the
soil type and corresponding type ID to the STM, using the `enrich_from_polygon`
function.

```python
import geopandas as gpd
polygon = gpd.read_file('soil_map.gpkg')
fields_to_query = ['soil_type', 'type_id']
stmat_enriched = stmat.stm.enrich_from_polygon(polygon, fields_to_query)
```
Two attributes from `soil_map.gpkg`: `soil_type` and `type_id`, will be added as data variables to the STM.

In case of a large file `soil_map.gpkg`, one can directly pass the file path to `enrich_from_polygon` to trigger the chunked enrichment:

```python
path_polygon = Path('soil_map.gpkg')
fields_to_query = ['soil_type', 'type_id']
stmat_enriched = stmat.stm.enrich_from_polygon(path_polygon, fields_to_query)
```

### Enrich from a dataset

STMTools supports enriching an STM by a dataset or a data array. For example, if
a dataset (`meteo_data.nc`) is available together with an STM, one can first
read `meteo_data.nc` using the `Xarray` library, then add the dataset to the
STM, using the `enrich_from_dataset` function.

```python
import xarray as xr
dataset = xr.open_dataset('meteo_data.nc')

# one field
stmat_enriched = stmat.stm.enrich_from_dataset(dataset, 'temperature')

# multiple fields
stmat_enriched = stmat.stm.enrich_from_dataset(dataset, ['temperature', 'precipitation'])
```

By default `"nearest"` is used for the interpolation. But you can choose [any
method provided by
Xarray](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.interp.html).
For example, if you want to use `"linear"` interpolation, you can do it like
this:

```python
stmat_enriched = stmat.stm.enrich_from_dataset(dataset, 'temperature', method='linear')
```

## Subset an STM

A subset of an STM can be obtained based on 1) thresholding on an attribute, or 2) intersection with a background polygon.

### Subset by an attribute

For example, select entries with `pnt_enscoh` higher than 0.7:

```python
stmat_subset = stmat.stm.subset(method='threshold', var='pnt_enscoh', threshold='>0.7')
```

This is equivalent to Xarray filtering:

```python
mask = stmat['pnt_enscoh'] > 0.7
mask = mask.compute()
stmat_subset = stmat.where(mask, drop=True)
```

### Subset by a polygon

Select all entries inside the polygons in `example_polygon.shp`:

```python
import geopandas as gpd
polygon = gpd.read_file('example_polygon.shp')
stm_demo.stm.subset(method='polygon', polygon=polygon)
```

Subset can also operate on the polygon file directly if the file is too big to load in the memory:

```python
stmat_subset = stm_demo.stm.subset(method='polygon', polygon='example_polygon.gpkg')
```


## Regulate the dimensions of an STM

Use `regulate_dims` to add a missing `space` or `time` dimension.

```python
# An STM witout time dimension
nspace = 10
stm_only_space = xr.Dataset(data_vars=dict(data=(['space'], np.arange(nspace))))

stm_only_space
```

```output
<xarray.Dataset>
Dimensions:  (space: 10)
Dimensions without coordinates: space
Data variables:
    data     (space) int64 0 1 2 3 4 5 6 7 8 9
```

```python
stm_only_space.regulate_dims()
```

```output
<xarray.Dataset>
Dimensions:  (time: 1, space: 10)
Dimensions without coordinates: time, space
Data variables:
    data     (space, time) int64 0 1 2 3 4 5 6 7 8 9
```

## Assign metadata

Use `register_metadata` to assign metadata to an STM by a Python dictionary.

```python
metadata_normal = dict(techniqueId='ID0001', datasetId='ID_datasetID', crs=4326)
stmat_with_metadata = stmat.stm.register_metadata(metadata_normal)
```