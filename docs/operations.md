# Operations on STM

## Regulate dimensions of an STM

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

## Subset an STM

## Enrich an STM

## Assign metadata
