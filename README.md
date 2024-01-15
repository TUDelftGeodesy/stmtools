# STMtools: Space Time Matrix Toolbox

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7717084.svg)](https://doi.org/10.5281/zenodo.7717084)
[![License](https://img.shields.io/github/license/TUDelftGeodesy/sarxarray)](https://opensource.org/licenses/Apache-2.0)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8027/badge?)](https://www.bestpractices.dev/projects/8027)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=MotionbyLearning_stmtools&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=MotionbyLearning_stmtools)
[![Build](https://github.com/TUDelftGeodesy/stmtools/actions/workflows/build.yml/badge.svg)](https://github.com/TUDelftGeodesy/stmtools/actions/workflows/build.yml)

STMTools (Space-Time Matrix Tools) is an Xarray extension for Space-Time Matrix (*Bruna et al., 2021; van Leijen et al., 2021*). It provides tools to read, write, enrich, and manipulate a Space-Time Matrix (STM).

A STM is a data array containing data with a space (point, location) and time (epoch) component, as well as contextual data. STMTools utilizes Xarray’s multi-dimensional labeling feature, and Zarr's chunk storage feature, to efficiently read and write large Space-Time matrices.

The contextual data enrichment functionality is implemented with Dask. Therefore it can be performed in a paralleled style on Hyper-Performance Computation (HPC) systems.

At this stage, stmtools specifically focus on the implementation for radar interferometry measurements, e.g. Persistent Scatterer, Distributed Scatterer, etc, with the possibility to be extended to other measurements with space and time attributes.

## Installation

STMtools can be installed from PyPI:

```sh
pip install stmtools
```

or from the source:

```sh
git clone git@github.com:TUDelftGeodesy/stmtools.git
cd stmtools
pip install .
```

Note that Python version `>=3.10` is required for STMtools.

## Documentation

For more information on usage and examples of STMTools, please refer to the [documentation](https://tudelftgeodesy.github.io/stmtools/).

## References
[1] Bruna, M. F. D., van Leijen, F. J., & Hanssen, R. F. (2021). A Generic Storage Method for Coherent Scatterers and Their Contextual Attributes. In 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS: Proceedings (pp. 1970-1973). [9553453] (International Geoscience and Remote Sensing Symposium (IGARSS); Vol. 2021-July). IEEE . https://doi.org/10.1109/IGARSS47720.2021.9553453

[2] van Leijen, F. J., van der Marel, H., & Hanssen, R. F. (2021). Towards the Integrated Processing of Geodetic Data. In 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS: Proceedings (pp. 3995-3998). [9554887] IEEE . https://doi.org/10.1109/IGARSS47720.2021.9554887
