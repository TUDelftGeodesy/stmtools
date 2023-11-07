# Operations on STM

STMTools supports various operations on STM.

## Enrich an STM

Contextual data can be added to an STM by enrichment. At present, stmtools supports enriching an STM by static polygons.

For example, if soil type data (`soil_map.gpkg`) is available together with an STM, then we can add the soil type and corresponding type code to the STM, using `enrich_from_polygon`:

```python
path_polygon = Path('soil_map.gpkg')
fields_to_query = ['soil_type', 'type_code']
stmat_enriched = stmat.stm.enrich_from_polygon(path_polygon, fields_to_query)
```

Two attributes from `soil_map.gpkg` to the STM: `soil_type` and `type_id`, will be added as data variables to the STM.

When `soil_map.gpkg` is small, loading it as an `GeoDataFrame` is faster:

```python
import geopandas as gpd
polygon = gpd.read_file('soil_map.gpkg')
fields_to_query = ['soil_type', 'type_code']
stmat_enriched = stmat.stm.enrich_from_polygon(polygon, fields_to_query)
```

## Subset an STM

A subset of an STM can be calculated based on 1) thresholding on a data-variable, or 2) intersection with a background polygon.

### By an attribute

For example, select entries with `pnt_enscoh` higher than 0.7:

```python
stmat_subset = stmat.stm.subset(method="threshold", var="pnt_enscoh", threshold='>0.7')
```

This is equivelent to Xarray filtering:

```python
mask = stmat["pnt_enscoh"] > 0.7
mask = mask.compute()
stmat_subset = stmat.where(mask, drop=True)
``` 

### By polygon

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

Use `regulate_dims` to add missing `space` or `time` dimension.

```python
# An STM witout time dimension
nspace = 10
stm_only_spcae = xr.Dataset(data_vars=dict(data=(["space"], np.arange(nspace))))

stm_only_spcae
```

```output
<xarray.Dataset>
Dimensions:  (space: 10)
Dimensions without coordinates: space
Data variables:
    data     (space) int64 0 1 2 3 4 5 6 7 8 9
```

```python
stm_only_spcae.regulate_dims()
```

```output
<xarray.Dataset>
Dimensions:  (time: 1, space: 10)
Dimensions without coordinates: time, space
Data variables:
    data     (space, time) int64 0 1 2 3 4 5 6 7 8 9
```

## Assign metadata

Use `register_metadata` to assign dictionary metadata to an STM.

```python
metadata_normal = dict(techniqueId="ID0001", datasetId="ID_datasetID", crs=4326)
stmat_with_metadata = stmat.stm.register_metadata(metadata_normal)
```