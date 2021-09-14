# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2021-09-14

Xcompact3d-toolbox has evolved considerably in the last year.
The pre-release version has been employed in CFD research projects, and the feedback from the users helped to improve its interfaces and functionalities.
The integration between this Python package and the numerical solver [XCompact3d](https://github.com/xcompact3d/Incompact3d) were experimental, and many of the functionalities were only available in a forked repository for a test of concept ([fschuch/Xcompact3d](https://github.com/fschuch/Xcompact3d)). These features are now part of the main repository [XCompact3d](https://github.com/xcompact3d/Incompact3d) (see [PR #51](https://github.com/xcompact3d/Incompact3d/pull/51)).
With this, Xcompact3d-toolbox is ready for its first stable release.

### Added

- Support for stretched mesh at the xdmf writer and automatized tests for it, by [@fschuch](https://github.com/fschuch).
- A class to handle the binary filenames and its tests. Now all methods support different filenames, like the classic `ux000`, or the new `ux-0000.bin`, besides some combinations between them. By [@fschuch](https://github.com/fschuch).
- Classes to handle the coordinates and their tests, so they can be moved out of the parameters class, by [@fschuch](https://github.com/fschuch).
- New class for the 3D coordinate system, with useful methods and its testes, by [@fschuch](https://github.com/fschuch).
- Add `class Dataset` to read the binary files from XCompact3d on-demand, it is possible to loop through the snapshots, select any of them or read the entire time-series for a given variable. It returns them in a proper `xarray.Dataset`, by [@fschuch](https://github.com/fschuch).
- Dependency sets for extra functionalities, by [@fschuch](https://github.com/fschuch).
- More parameters are now covered in the Parameters class, by [@fschuch](https://github.com/fschuch).
- Add type hints to many functions and class methods, for a better development experience using modern IDEs, by [@fschuch](https://github.com/fschuch).

### Modified

- Writing methods for the binary files were moved from the xarray accessors to the Dataset class, by [@fschuch](https://github.com/fschuch).
- The new sandbox flow configuration is now set with `itype = 12`, following its addition the XCompact3d's main repository (see [#51](https://github.com/xcompact3d/Incompact3d/pull/51)), by [@fschuch](https://github.com/fschuch).
- The documentation was improved, by [@fschuch](https://github.com/fschuch).

### Fixed

- Not installing all dependencies with `pip install`, by [@fschuch](https://github.com/fschuch).
- Suppressed warning from `tqdm`, by [@fschuch](https://github.com/fschuch).
- The output format from `gene_epsi_3D` has changed, fixing some compatibility issues with XCompact3d as well (see [#51](https://github.com/xcompact3d/Incompact3d/pull/51)), by [@fschuch](https://github.com/fschuch).

## [0.1.11] - 2021-02-12

### Fixed

- Fix #8, a little incompatibility problem with xcompact3d was fixed, by [@fschuch](https://github.com/fschuch).

## [0.1.10] - 2021-02-11

### Added
- Add support for the previous parameters format `.prm` (#7), by [@fschuch](https://github.com/fschuch).
- Class `ParametersGui`, a subclass from `Parameters`, but with widgets, by [@fschuch](https://github.com/fschuch).
- Argument `loadfile` added to class `Parameters`, so it is possible to initialize and load form the disc with just one line of code `prm = x3d.Parameters(loadfile='example.i3d')`, by [@fschuch](https://github.com/fschuch).

### Changed
- Changed from method `__call__` to `__repr__` at `parameters.py` as the procedure to show the parameters on screen, by [@fschuch](https://github.com/fschuch).
- Functions to read binary fields where moved from `io.py` to methods at `parameters.py`, so the syntax is simplified from `x3d.read_field('filename', prm)` to just `prm.read_field('filename')`. by [@fschuch](https://github.com/fschuch).

### Fixed
- Scale factor for Ahmed Body at sandbox, by [@fschuch](https://github.com/fschuch).
- Fix #2, widgets are now working in a new class `ParametersGui`, by [@fschuch](https://github.com/fschuch).
- Fix #5, Bug at Ahmed body when using double precision, by [@fschuch](https://github.com/fschuch).
- Fix #6, The files describing the geometry are incompatible when running on Linux, by [@fschuch](https://github.com/fschuch).

## [0.1.9] - 2020-10-09

### Added
- `get_boundary_condition` at class `Parameters`. It returns the appropriate boundary parameters that are
expected by the derivative functions. by [@fschuch](https://github.com/fschuch).
- First and second derivatives for stretched mesh in y, by [@fschuch](https://github.com/fschuch).

### Changed
- Syntax for `<da,ds>.x3d.simps` and `<da,ds>.x3d.pencil_decomp`. It is now possible to use them over multiple dimensions with just one call (for instance `ds.x3d.simps('x', 'y', 'z')`), by [@fschuch](https://github.com/fschuch).

### Fixed
- First derivative was incorrect when `ncl1=1` and `ncln=2`. by [@fschuch](https://github.com/fschuch).

## [0.1.8] - 2020-09-29

### Fixed
- `param.mytype` was not updating properly [#4](https://github.com/fschuch/xcompact3d_toolbox/issues/4), by [@fschuch](https://github.com/fschuch).

## [0.1.7] - 2020-08-28

### Fixed
- BC parameters at ([param.py](xcompact3d_toolbox\param.py)), by [@fschuch](https://github.com/fschuch).

## [0.1.6] - 2020-08-28

### Fixed
- [python-publish](.github/workflows/python-package.yml) action does not trigger if the release was first drafted, then published, by [@fschuch](https://github.com/fschuch).

## [0.1.5] - 2020-08-28

### Added
- Support for stretched mesh in y, by [@fschuch](https://github.com/fschuch).

### Fixed
- `get_mesh`, by [@fschuch](https://github.com/fschuch).
- `write_xdmf`, by [@fschuch](https://github.com/fschuch).

## [0.1.4] - 2020-08-20
### Added
- [python-versioneer](https://github.com/warner/python-versioneer) removes the tedious and error-prone "update the embedded version string" step from our release process, by [@fschuch](https://github.com/fschuch).
- Docstrings for most classes, methods and functions, by [@fschuch](https://github.com/fschuch).
- Examples for Sandbox flow configuration (by [@fschuch](https://github.com/fschuch)):

  - [Turbidity Current in Axisymmetric Configuration](docs\examples\Axisymmetric_flow.ipynb);
  - [Flow Around a Complex Body](docs\examples\Cylinder.ipynb).

- Tutorials (by [@fschuch](https://github.com/fschuch)):

  - [Parameters](docs\tutorial\parameters.ipynb).


- Integration with [Read the Docs](https://xcompact3d-toolbox.readthedocs.io/en/latest/), by [@fschuch](https://github.com/fschuch).

### Changed
- Code style changed to [black](https://github.com/psf/black), by [@fschuch](https://github.com/fschuch).
- `param = {'mytype': np.float64}` changed to just `mytype = float64` ([param.py](xcompact3d_toolbox\param.py)), by [@fschuch](https://github.com/fschuch).

## [0.1.3] - 2020-08-17
No changes, just trying to get familiar with workflows and the release to Pypi.

## [0.1.2] - 2020-08-17
### Added
- Unittest for observations and validations at [parameters.py](./xcompact3d_toolbox/parameters.py), by [@fschuch](https://github.com/fschuch).

### Changed
- Temporarily disabling the link between parameters and their widgets (see [#2](https://github.com/fschuch/xcompact3d_toolbox/issues/2)), by [@fschuch](https://github.com/fschuch).

## [0.1.1] - 2020-08-14
No changes, just trying to get familiar with workflows and the release to Pypi.

## [0.0.0] - 2020-08-14
### Added
- CHANGELOG.md by [@fschuch](https://github.com/fschuch).
- `class Parameters`  built on top of [traitlets](https://traitlets.readthedocs.io/en/stable/index.html), for type checking, dynamically calculated default values, and ‘on change’ callbacks, by [@fschuch](https://github.com/fschuch).
- [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) for all relevant parameters and two-ways linking with [traitlets](https://traitlets.readthedocs.io/en/stable/index.html) variables, by [@fschuch](https://github.com/fschuch).
- Accessors for [xarray](http://xarray.pydata.org/en/stable/)'s `Dataset` and `DataArray`, making possible high-order derivatives (with appropriated boundary conditions), I/O, parallel execution with pencil decomposition (powered by [dask](https://dask.org/)) and integration with `scipy.integrate.simps` and `scipy.integrate.cumtrapz`. By [@fschuch](https://github.com/fschuch).
- Ported genepsi.f90 to genepsi.py (powered by [Numba](http://numba.pydata.org/)), generating all the files necessary for our customized Immersed Boundary Method, by [@fschuch](https://github.com/fschuch).
- Support to *Sandbox Flow Configuration* (see [fschuch/Xcompact3d](https://github.com/fschuch/Xcompact3d/)), by [@fschuch](https://github.com/fschuch).
- Ahmed body as benchmark geometry, mirror and plotting tools, by [@momba98](https://github.com/momba98).

[Unreleased]: https://github.com/fschuch/xcompact3d_toolbox/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.11...v1.0.0
[0.1.11]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/fschuch/xcompact3d_toolbox/compare/v0.0.0...v0.1.1
[0.0.0]: https://github.com/fschuch/xcompact3d_toolbox/releases/tag/v0.0.0
