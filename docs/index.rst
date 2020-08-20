.. Xcompact3d_toolbox documentation master file, created by
   sphinx-quickstart on Mon Aug 17 09:28:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Xcompact3d_toolbox's documentation!
==============================================

It is a Python package designed to handle the pre and postprocessing of
the high-order Navier-Stokes solver Xcompact3d_. It aims to help users and
code developers with a set of tools and automated processes.

**Xcompact3d Toolbox** is still in pre-release, be aware that new features are
going to be added to it and the current features may change with no further notice.

The physical and computational parameters are built on top of traitlets_,
a framework that lets Python classes have attributes with type checking, dynamically
calculated default values, and ‘on change’ callbacks.
In addition to ipywidgets_ for an user friendly interface.

Data structure is provided by xarray_ (see `Why xarray?`_), that introduces labels
in the form of dimensions, coordinates and attributes on top of raw NumPy_-like
arrays, which allows for a more intuitive, more concise, and less error-prone
developer experience. It integrates tightly with dask_ for parallel computing.

Finally, Xcompact3d Toolbox is fully integrated with the new *Sandbox Flow Configuration*
(see `fschuch/Xcompact3d`_). The idea is to easily provide everything that X3d needs
from a `Jupyter Notebook`_, like initial conditions, solid geometry, boundary
conditions, and the parameters. It makes life easier for beginners, that can run
any new flow configuration without worrying about Fortran and 2decomp_. For
developers, it works as a rapid prototyping tool, to test concepts and then
compare results to validate any future Fortran implementations.

Installation
------------

``pip install xcompact3d-toolbox``

Useful links
------------

* `View on GitHub`_;
* `Suggestions for new features and bug report`_;
* `See what is coming next (Project page)`_.

.. _`View on GitHub`: https://github.com/fschuch/xcompact3d_toolbox
.. _`Suggestions for new features and bug report`: https://github.com/fschuch/xcompact3d_toolbox/issues
.. _`See what is coming next (Project page)`: https://github.com/fschuch/xcompact3d_toolbox/projects/1

.. _2decomp: http://www.2decomp.org/
.. _dask: https://dask.org/
.. _`fschuch/Xcompact3d`: https://github.com/fschuch/Xcompact3d/
.. _ipywidgets: https://ipywidgets.readthedocs.io/en/latest/
.. _`Jupyter Notebook`: https://jupyter.org/
.. _Numpy: https://numpy.org/
.. _traitlets: https://traitlets.readthedocs.io/en/stable/index.html
.. _xarray: http://xarray.pydata.org/en/stable/
.. _Xcompact3d: https://github.com/xcompact3d/Incompact3d
.. _`Why xarray?`: http://xarray.pydata.org/en/stable/why-xarray.html

Table of Content
==================

.. toctree::
   :maxdepth: 4
   :glob:

   Docstrings
   tutorial

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
