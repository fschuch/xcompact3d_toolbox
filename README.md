# Xcompact3d Toolbox

![Build Status](https://github.com/fschuch/xcompact3d_toolbox/workflows/Python%20package/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/xcompact3d-toolbox/badge/?version=latest)](https://xcompact3d-toolbox.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/xcompact3d-toolbox.svg)](https://badge.fury.io/py/xcompact3d-toolbox)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

It is a Python package designed to handle the pre and postprocessing of
the high-order Navier-Stokes solver Xcompact3d. It aims to help users and
code developers with a set of tools and automated processes.

**Xcompact3d Toolbox** is still in pre-release, be aware that new features are
going to be added to it and the current features may change with no further notice.

The physical and computational parameters are built on top of [traitlets](https://traitlets.readthedocs.io/en/stable/index.html),
a framework that lets Python classes have attributes with type checking, dynamically calculated default values, and ‘on change’ callbacks.
In addition to [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) for an user friendly interface.

Data structure is provided by [xarray](http://xarray.pydata.org/en/stable/) (see [Why xarray?](http://xarray.pydata.org/en/stable/why-xarray.html)), that introduces labels in the form of dimensions, coordinates and attributes on top of raw [NumPy](https://numpy.org/)-like arrays, which allows for a more intuitive, more concise, and less error-prone developer experience. It integrates tightly with [dask](https://dask.org/) for parallel computing.

Finally, Xcompact3d Toolbox is fully integrated with the new *Sandbox Flow Configuration* (see [fschuch/Xcompact3d](https://github.com/fschuch/Xcompact3d/)). The idea is to easily provide everything that X3d needs from a [Jupyter Notebook](https://jupyter.org/), like initial conditions, solid geometry, boundary conditions, and the parameters ([see examples](https://xcompact3d-toolbox.readthedocs.io/en/latest/tutorial.html#sandbox-examples)). It makes life easier for beginners, that can run any new flow configuration without worrying about Fortran and [2decomp](http://www.2decomp.org/). For developers, it works as a rapid prototyping tool, to test concepts and then compare results to validate any future Fortran implementations.

## Installation

`pip install xcompact3d-toolbox`

## Useful links

* [Documentation](https://xcompact3d-toolbox.readthedocs.io/);
* [Suggestions for new features and bug report](https://github.com/fschuch/xcompact3d_toolbox/issues);
* [See what is coming next (Project page)](https://github.com/fschuch/xcompact3d_toolbox/projects/1);
* [Xcompact3d's repository](https://github.com/xcompact3d/Incompact3d);
* [My fork from Xcompact3d's repository](https://github.com/fschuch/Xcompact3d/);

## Copyright and License

(c) 2020 [Felipe N. Schuch](https://fschuch.github.io/). All content is under [GPL-3.0 License](https://github.com/fschuch/xcompact3d_toolbox/blob/master/LICENSE).
