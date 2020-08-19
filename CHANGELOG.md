# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2020-08-17
### Added
- Unittest for observations and validations at [parameters.py](./xcompact3d_toolbox/parameters.py), by [@fschuch](https://github.com/fschuch).

## Changed
- Temporarily disabling the link between parameters and their widgets (see [#2](https://github.com/fschuch/xcompact3d_toolbox/issues/2)), by [@fschuch](https://github.com/fschuch).

## [0.0.0] - 2020-08-14
### Added
- CHANGELOG.md by [@fschuch](https://github.com/fschuch).
- `class Parameters`  built on top of [traitlets](https://traitlets.readthedocs.io/en/stable/index.html), for type checking, dynamically calculated default values, and ‘on change’ callbacks, by [@fschuch](https://github.com/fschuch).
- [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) for all relevant parameters and two-ways linking with [traitlets](https://traitlets.readthedocs.io/en/stable/index.html) variables, by [@fschuch](https://github.com/fschuch).
- Accessors for [xarray](http://xarray.pydata.org/en/stable/)'s `Dataset` and `DataArray`, making possible high-order derivatives (with appropriated boundary conditions), I/O, parallel execution with pencil decomposition (powered by [dask](https://dask.org/)) and integration with `scipy.integrate.simps` and `scipy.integrate.cumtrapz`. By [@fschuch](https://github.com/fschuch).
- Ported genepsi.f90 to genepsi.py (powered by [Numba](http://numba.pydata.org/)), generating all the files necessary for our customized Immersed Boundary Method, by [@fschuch](https://github.com/fschuch).
- Support to *Sandbox Flow Configuration* (see [fschuch/Xcompact3d](https://github.com/fschuch/Xcompact3d/)), by [@fschuch](https://github.com/fschuch).
- Ahmed body as benchmark geometry, mirror and plotting tools, by [@momba98](https://github.com/momba98).

[Unreleased]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.0.0...v0.1.3
[0.0.0]: https://github.com/fschuch/xcompact3d_toolbox/releases/tag/v0.0.0
