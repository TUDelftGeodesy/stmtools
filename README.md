# STMtools: Space Time Matrix Toolbox

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7717088.svg)](https://doi.org/10.5281/zenodo.7717088)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8027/badge)](https://www.bestpractices.dev/projects/8027)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=MotionbyLearning_stmtools&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=MotionbyLearning_stmtools)
[![Build](https://github.com/MotionbyLearning/stmtools/actions/workflows/build.yml/badge.svg)](https://github.com/MotionbyLearning/stmtools/actions/workflows/build.yml)

`STMtools` is an open-source Xarray extension for Space-Time Matrix data. It is implemented for PSI-InSAR data processing.

## Installation

STMtools can be installed from PyPI:

```sh
pip install stmtools
```

or from the source:

```sh
git clone git@github.com:MotionbyLearning/stmtools.git
cd stmtools
pip install .
```

Note that Python version `>=3.10` is required for STMtools.

## Documentation

For more information on usage and examples of SARXarray, please refer to the [documentation](https://motionbylearning.github.io/stmtools/).

## References
[1] Bruna, M. F. D., van Leijen, F. J., & Hanssen, R. F. (2021). A Generic Storage Method for Coherent Scatterers and Their Contextual Attributes. In 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS: Proceedings (pp. 1970-1973). [9553453] (International Geoscience and Remote Sensing Symposium (IGARSS); Vol. 2021-July). IEEE . https://doi.org/10.1109/IGARSS47720.2021.9553453

[2] van Leijen, F. J., van der Marel, H., & Hanssen, R. F. (2021). Towards the Integrated Processing of Geodetic Data. In 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS: Proceedings (pp. 3995-3998). [9554887] IEEE . https://doi.org/10.1109/IGARSS47720.2021.9554887