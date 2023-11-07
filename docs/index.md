# STMTools

STMTools (Space-Time Matrix Tools) is an Xarray extension for Space-Time Matrix (*Bruna et al., 2021; van Leijen et al., 2021*). It provides tools to read, write, enrich, and manipulate a Space-Time Matrix (STM).

An STM is a dataset containing data with a space (point, location) and time (epoch) component, as well as contextual data. STMTools utilizes Xarray’s multi-dimensional labeling feature, and Zarr's chunk storage feature, to efficiently read and write large Space-Time matrices.

The contextual data enrichment functionality is implemented with Dask. Therefore it can be performed in a paralleled style on High Performance Computing (HPC) systems.

At this stage, stmtools specifically focus on the implementation for radar interferometry measurements, e.g. Persistent Scatterer, Distributed Scatterer, etc, with the possibility to be extended to other measurements with space and time attributes.

## References
[1] Bruna, M. F. D., van Leijen, F. J., & Hanssen, R. F. (2021). A Generic Storage Method for Coherent Scatterers and Their Contextual Attributes. In 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS: Proceedings (pp. 1970-1973). [9553453] (International Geoscience and Remote Sensing Symposium (IGARSS); Vol. 2021-July). IEEE . https://doi.org/10.1109/IGARSS47720.2021.9553453

[2] van Leijen, F. J., van der Marel, H., & Hanssen, R. F. (2021). Towards the Integrated Processing of Geodetic Data. In 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS: Proceedings (pp. 3995-3998). [9554887] IEEE . https://doi.org/10.1109/IGARSS47720.2021.9554887
