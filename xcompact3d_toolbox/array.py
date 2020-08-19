# -*- coding: utf-8 -*-
"""The data structure is provided by `xarray`_, that introduces labels in the
form of dimensions, coordinates and attributes on top of raw `NumPy`_-like
arrays, which allows for a more intuitive, more concise, and less error-prone
developer experience. See `xarray`_'s User Guide for a complete overview about
its data structures and buit-in functions for indexing, selecting, computing,
plotting and much more.
It integrates tightly with `dask`_ for parallel computing.

**Xcompact3d_toolbox** adds extra functions on top of :obj:`xarray.DataArray`
and :obj:`xarray.Dataset`, all the details are described bellow.

.. _dask:
    https://dask.org/
.. _numpy:
    https://numpy.org/
.. _xarray:
    http://xarray.pydata.org/en/stable/
"""

from .param import mytype
from .derive import FirstDerivative, SecondDerivative
import xarray as xr
import numpy as np
import os.path


@xr.register_dataset_accessor("x3d")
class X3dDataset:
    """An accessor with extra utilities for :obj:`xarray.Dataset`.
    """

    def __init__(self, data_set):

        self._data_set = data_set

    def write(self, prm):
        """Write the arrays in this data set to binary files on the disc, in the
        same order that Xcompact3d would do, so they can be easily read with
        2DECOMP.

        In order to avoid overwriting any relevant field, only variables with
        an **attribute** called ``file_name`` will be written.

        See :obj:`xcompact3d_toolbox.array.X3dDataArray` for more information.

        Parameters
        ----------
        prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
            Contains the computational and physical parameters.

        Examples
        -------

        >>> ds.x3d.write(prm)

        """
        for key, val in self._data_set.items():
            if "file_name" in val.attrs:
                print("Writing : " + key)
                val.x3d.write(prm)

    def cumtrapz(self, dim):
        """Cumulatively integrate :obj:`xarray.Dataset` in direction ``dim``
        using the composite trapezoidal rule.
        It is a wrapper for :obj:`scipy.integrate.cumtrapz`.
        Initial value is defined to zero.

        Parameters
        ----------
        dim : str
            Coordinate used for the integration.

        Returns
        -------
        :obj:`xarray.Dataset`
            Integrated

        Examples
        -------

        >>> ds.x3d.cumtrapz('t')

        """
        from scipy.integrate import cumtrapz

        return xr.apply_ufunc(
            cumtrapz,
            self._data_set,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            output_dtypes=[mytype],
            kwargs={"x": self._data_set[dim], "axis": -1, "initial": 0.0},
        )

    def simps(self, dim):
        """Integrate :obj:`xarray.Dataset` in direction ``dim`` using the
        composite Simpson’s rule.
        It is a wrapper for :obj:`scipy.integrate.simps`.

        Parameters
        ----------
        dim : str
            Coordinate used for the integration.

        Returns
        -------
        :obj:`xarray.Dataset`
            Integrated

        Examples
        -------

        >>> ds.x3d.simps('x')

        """
        from scipy.integrate import simps

        return xr.apply_ufunc(
            simps,
            self._data_set,
            input_core_dims=[[dim]],
            dask="parallelized",
            output_dtypes=[mytype],
            kwargs={"x": self._data_set[dim], "axis": -1},
        )

    def pencil_decomp(self, dim=None, chunks=None):
        """Coerce all arrays in this dataset into dask arrays.

        It applies ``chunk=-1`` for all coordinates listed in ``dim``, which means
        no decompositon, and ``'auto'`` to the others, resulting in a pencil
        decomposition for parallel evaluation.

        If ``chunks`` is provided, the behavior will be just like
        :obj:`xarray.Dataset.chunk`.

        Parameters
        ----------
        dim : str or sequence of str
            Description of parameter `dim` (the default is None).
        chunks : int, 'auto' or maping, optional
            Description of parameter `chunks` (the default is None).

        Returns
        -------
        :obj:`xarray.Dataset`
            **chunked**

        Examples
        -------

        >>> ds.x3d.pencil_decomp('x') # Pencil decomposition
        >>> ds.x3d.pencil_decomp('t')
        >>> ds.x3d.pencil_decomp('y', 'z') # Slab decomposition

        """
        if chunks == None and dim != None:
            chunks = {}
            for var in self._data_set.dims:
                if var in dim:
                    # no chunking along this dimension
                    chunks[var] = -1
                else:
                    # allow the chunking in this dimension to accommodate ideal chunk sizes
                    chunks[var] = "auto"

        return self._data_set.chunk(chunks)


@xr.register_dataarray_accessor("x3d")
class X3dDataArray:
    """An acessor with extra utilities for :obj:`xarray.DataArray`.
    """

    def __init__(self, data_array):
        self._data_array = data_array

        self._Dx = {}
        self._Dxx = {}

    def write(self, prm, filename=None):
        """Write the array to binary files on the disc, in the same order that
        Xcompact3d would do, so they can be easily read with 2DECOMP.

        Coordinates are properly aligned before writing.

        If filename is not provided, it may be obtained from the **attribute**
        called ``file_name``.

        If ``n`` is a valid coordinate (for scalar fractions) in the array, one
        numerated binary file will be written for each scalar field.

        If ``t`` is a valid coordinate (for time) in the array, one numerated
        binary file will be written for each available time.

        Parameters
        ----------
        prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
            Contains the computational and physical parameters.
        filename : str, optional
            Filename for binary file (default is :obj:`None`).

        Examples
        -------

        >>> da.x3d.write(prm, './data/3d_snapshots/ux')

        """
        if filename == None:  # Try to get from atributes
            filename = self._data_array.attrs.get("file_name", None)
        if filename != None:
            if "n" in self._data_array.dims:
                for n in self._data_array.n:
                    self._data_array.sel(n=n).x3d.write(
                        prm, filename + str(n.values + 1)
                    )
            elif "t" in self._data_array.dims:

                from tqdm.notebook import tqdm as tqdm

                fmt = prm.ifilenameformat
                k = 0
                for t in tqdm(self._data_array.t.values, desc=filename):
                    num = str(int(t / prm.dt)).zfill(fmt)
                    self._data_array.isel(t=k).x3d.write(prm, filename + "-" + num)
                    k += 1
            else:
                align = []
                for i in reversed(sorted(self._data_array.dims)):
                    align.append(self._data_array.get_axis_num(i))
                self._data_array.values.astype(mytype).transpose(align).tofile(
                    filename + ".bin"
                )

    def cumtrapz(self, dim):
        """Cumulatively integrate :obj:`xarray.DataArray` in direction ``dim``
        using the composite trapezoidal rule.
        It is a wrapper for :obj:`scipy.integrate.cumtrapz`.
        Initial value is defined to zero.

        Parameters
        ----------
        dim : str
            Coordinate used for the integration.

        Returns
        -------
        :obj:`xarray.DataArray`
            Integrated

        Examples
        -------

        >>> da.x3d.cumtrapz('t')

        """
        ds = self._data_array._to_temp_dataset().x3d.cumtrapz(dim)
        return self._data_array._from_temp_dataset(ds)

    def simps(self, dim):
        """Integrate :obj:`xarray.DataArray` in direction ``dim`` using the
        composite Simpson’s rule.
        It is a wrapper for :obj:`scipy.integrate.simps`.

        Parameters
        ----------
        dim : str
            Coordinate used for the integration.

        Returns
        -------
        :obj:`xarray.DataArray`
            Integrated

        Examples
        -------

        >>> da.x3d.simps('x')

        """
        ds = self._data_array._to_temp_dataset().x3d.simps(dim)
        return self._data_array._from_temp_dataset(ds)

    def pencil_decomp(self, dim=None, chunks=None):
        """Coerce the array into dask array.

        It applies ``chunk=-1`` for all coordinates listed in ``dim``, which means
        no decompositon, and ``'auto'`` to the others, resulting in a pencil
        decomposition for parallel evaluation.

        If ``chunks`` is provided, the behavior will be just like
        :obj:`xarray.DataArray.chunk`.

        Parameters
        ----------
        dim : str or sequence of str
            Description of parameter `dim` (the default is None).
        chunks : int, 'auto' or maping, optional
            Description of parameter `chunks` (the default is None).

        Returns
        -------
        :obj:`xarray.DataArray`
            **chunked**

        Examples
        -------

        >>> da.x3d.pencil_decomp('x') # Pencil decomposition
        >>> da.x3d.pencil_decomp('t')
        >>> da.x3d.pencil_decomp('y', 'z') # Slab decomposition

        """
        ds = self._data_array._to_temp_dataset().x3d.pencil_decomp(dim, chunks)
        return self._data_array._from_temp_dataset(ds)

    def first_derivative(self, dim):
        """Compute first derivative with the 4th order accurate centered scheme.

        It is fully functional with all boundary conditions available on Xcompact3d.
        The **atribute** ``BC`` is used to store BC information in a dictionary
        (see examples), default is ``ncl1 = ncln = 2`` and ``npaire = 1``.

        Parameters
        ----------
        dim : str
            Coordinate used for the derivative.

        Returns
        -------
        :obj:`xarray.DataArray`
            **differentiated**

        Examples
        -------

        >>> da.attrs['BC'] = {
        ...     'x': {
        ...         'ncl1': 1,
        ...         'ncln': 1,
        ...         'npaire': 0
        ...     },
        ...     'y': {
        ...         'ncl1': 2,
        ...         'ncln': 1,
        ...         'npaire': 1
        ...     },
        ...     'z': {
        ...         'ncl1': 0,
        ...         'ncln': 0,
        ...         'npaire': 1
        ... }
        >>> da.x3d.first_derivative('x')

        Notes
        -----
        The **atribute** ``BC`` is automatically defined for ``ux``, ``uy``,
        ``uz``, ``pp`` and ``phi`` when read from the disc with
        :obj:`xcompact3d_toolbox.io.readfield`.
        """

        if dim not in self._Dx:
            try:
                ncl1 = self._data_array.attrs["BC"][dim]["ncl1"]
                ncln = self._data_array.attrs["BC"][dim]["ncln"]
                npaire = self._data_array.attrs["BC"][dim]["npaire"]
            except:
                ncl1, ncln, npaire = 2, 2, 1

            n = self._data_array[dim].size
            d = (self._data_array[dim][1] - self._data_array[dim][0]).values
            self._Dx[dim] = FirstDerivative(n, d, ncl1, ncln, npaire)

        return xr.apply_ufunc(
            lambda f: self._Dx[dim].dot(f),
            self._data_array,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[mytype],
        )

    def second_derivative(self, dim):
        """Compute second derivative with the 4th order accurate centered scheme.

        It is fully functional with all boundary conditions available on Xcompact3d.
        The **atribute** ``BC`` is used to store BC information in a dictionary
        (see examples), default is ``ncl1 = ncln = 2`` and ``npaire = 1``.

        Parameters
        ----------
        dim : str
            Coordinate used for the derivative.

        Returns
        -------
        :obj:`xarray.DataArray`
            **differentiated**

        Examples
        -------

        >>> da.attrs['BC'] = {
        ...     'x': {
        ...         'ncl1': 1,
        ...         'ncln': 1,
        ...         'npaire': 0
        ...     },
        ...     'y': {
        ...         'ncl1': 2,
        ...         'ncln': 1,
        ...         'npaire': 1
        ...     },
        ...     'z': {
        ...         'ncl1': 0,
        ...         'ncln': 0,
        ...         'npaire': 1
        ... }
        >>> da.x3d.second_derivative('x')

        Notes
        -----
        The **atribute** ``BC`` is automatically defined for ``ux``, ``uy``,
        ``uz``, ``pp`` and ``phi`` when read from the disc with
        :obj:`xcompact3d_toolbox.io.readfield`.
        """
        if dim not in self._Dxx:
            try:
                ncl1 = self._data_array.attrs["BC"][dim]["ncl1"]
                ncln = self._data_array.attrs["BC"][dim]["ncln"]
                npaire = self._data_array.attrs["BC"][dim]["npaire"]
            except:
                ncl1, ncln, npaire = 2, 2, 1

            n = self._data_array[dim].size
            d = (self._data_array[dim][1] - self._data_array[dim][0]).values
            self._Dxx[dim] = SecondDerivative(n, d, ncl1, ncln, npaire)

        return xr.apply_ufunc(
            lambda f: self._Dxx[dim].dot(f),
            self._data_array,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[mytype],
        )
