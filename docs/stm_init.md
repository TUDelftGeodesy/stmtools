# Initiate a Space-Time Matrix

We implemented STM in Python as an [`Xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) object. An STM instance can be initiated as an `Xarray.Dataset` in different ways.

STMTools provides a reader to perform lazy loading from a csv file. However, we recommend to store STM in [`zarr`](https://zarr.readthedocs.io/en/stable/) format, and directly load them as an Xarray object by [`xarray.open_zarr`](https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html).

## Manually initiate an STM

When represented by `xarray.Dataset`, an STM is a `Dataset` object with "space" and "time" dimension. It can be initiated manually, e.g.:

```python
# Define dimension sizes
nspace = 10
ntime = 5

# Initialte STM as Dataset
stm = xr.Dataset(
    data_vars=dict(
        space_time_data=(
            ["space", "time"],
            np.arange(nspace * ntime).reshape((nspace, ntime)),
        ),
        space_data=(["space"], np.arange(nspace)),
        time_data=(["time"], np.arange(ntime)),
    ),
    coords=dict(
        x_coords=(["space"], np.arange(nspace)),
        y_coords=(["space"], np.arange(nspace)),
        time=(["time"], np.arange(ntime)),
    ),
)

stm
```

```output
<xarray.Dataset>
Dimensions:          (space: 10, time: 5)
Coordinates:
    x_coords         (space) int64 0 1 2 3 4 5 6 7 8 9
    y_coords         (space) int64 0 1 2 3 4 5 6 7 8 9
  * time             (time) int64 0 1 2 3 4
Dimensions without coordinates: space
Data variables:
    space_time_data  (space, time) int64 0 1 2 3 4 5 6 ... 43 44 45 46 47 48 49
    space_data       (space) int64 0 1 2 3 4 5 6 7 8 9
    time_data        (time) int64 0 1 2 3 4
```

## From a Zarr storage

If an STM is stored in `.zarr` format, it can be read by the `xarray.open_zarr` funtcion:

```python
stm = xr.open_zarr('./stm.zarr')
```

## From csv file

STM can also be intiated from a csv file. During this process, the following assumptions are made to the column names of the csv file:

1. All columns with space-only attributes share the same [Regular Expression (RE)](https://en.wikipedia.org/wiki/Regular_expression) pattern in the column names.
    E.g. Latitude, Longitude and height columns are named as "pnt_lat", "pnt_lon" and
    "pnt_height", sharing the same RE pattern "^pnt_";
2. Per space-time attribute, a common RE pattern is shared by all columns. E.g. for the
    time-series of amplitude data, the column names are "amp_20100101", "amp_20100110",
    "amp_20100119" ..., where "^amp_" is the common RE pattern;
3. There is no temporal-only (i.e. 1-row attribute) attribute present in the csv file.

Consider the [example csv data](./notebooks/data/example.csv). In this file, the
rows are points and the columns are time series of `deformation`, `amplitude`,
and `h2ph` variables. The columns names for these variables are `d_<timestamp>`,
`a_<timestamp>`, and `h2ph_<timestamp>` respectively. We can read this csv file
as an STM object in xarray format using the function `from_csv()`:

```python
import stmtools
stm = stmtools.from_csv('example.csv')
```

```output
<xarray.Dataset>
Dimensions:                (space: 2500, time: 11)
Coordinates:
  * space                  (space) int64 0 1 2 3 4 ... 2495 2496 2497 2498 2499
  * time                   (time) datetime64[ns] 2016-03-27 ... 2016-07-15
    lat                    (space) float64 dask.array<chunksize=(2500,), meta=np.ndarray>
    lon                    (space) float64 dask.array<chunksize=(2500,), meta=np.ndarray>
Data variables: (12/13)
    pnt_id                 (space) <U1 dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_flags              (space) int64 dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_line               (space) int64 dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_pixel              (space) int64 dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_height             (space) float64 dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_demheight          (space) float64 dask.array<chunksize=(2500,), meta=np.ndarray>
    ...                     ...
    pnt_enscoh             (space) float64 dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_ampconsist         (space) float64 dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_linear             (space) float64 dask.array<chunksize=(2500,), meta=np.ndarray>
    deformation            (space, time) float64 dask.array<chunksize=(2500, 11), meta=np.ndarray>
    amplitude              (space, time) float64 dask.array<chunksize=(2500, 11), meta=np.ndarray>
    h2ph                   (space, time) float64 dask.array<chunksize=(2500, 11), meta=np.ndarray>
```

By default, time values are extracted from the column names assumeing that the
names are in the format of `a_<YYYYMMDD>`, `d_<YYYYMMDD>`, and
`h2ph_<YYYYMMDD>`. Note that only a seperator "`_`", and a date format of
`YYYYMMDD` are supported. But if the names are different, for example
`amp_<YYYYMMDD>`, `def_<YYYYMMDD>`, and `h2ph_<YYYYMMDD>`, like [example2 csv
data](./notebooks/data/example2.csv), you can specify `spacetime_pattern`
argument as a dictionay mapping RE patterns of each space-time attribute to
corresponding variable names:

```python
import stmtools
stm = stmtools.from_csv('example2.csv', spacetime_pattern={
    '^amp_': 'amplitude',
    '^def_': 'deformation',
    '^h2ph_': 'h2ph'
})
```

```output
stm
<xarray.Dataset> Size: 910kB
Dimensions:                (space: 2500, time: 11)
Coordinates:
  * space                  (space) int64 20kB 0 1 2 3 4 ... 2496 2497 2498 2499
  * time                   (time) datetime64[ns] 88B 2016-03-27 ... 2016-07-15
    lat                    (space) float64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
    lon                    (space) float64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
Data variables: (12/13)
    pnt_id                 (space) <U1 10kB dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_flags              (space) int64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_line               (space) int64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_pixel              (space) int64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_height             (space) float64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_demheight          (space) float64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
    ...                     ...
    pnt_enscoh             (space) float64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_ampconsist         (space) float64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
    pnt_linear             (space) float64 20kB dask.array<chunksize=(2500,), meta=np.ndarray>
    amplitude              (space, time) float64 220kB dask.array<chunksize=(2500, 11), meta=np.ndarray>
    deformation            (space, time) float64 220kB dask.array<chunksize=(2500, 11), meta=np.ndarray>
    h2ph                   (space, time) float64 220kB dask.array<chunksize=(2500, 11), meta=np.ndarray>
```

## By pixel selection from an image stack

An STM can also be generated by selecting pixels from an SLC stack or interferogram stack. An example of the selection is the [`point_selection`](https://tudelftgeodesy.github.io/sarxarray/common_ops/#point-selection) implementation of [`sarxarray`](https://tudelftgeodesy.github.io/sarxarray/).