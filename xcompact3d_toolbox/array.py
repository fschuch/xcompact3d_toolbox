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

from .mesh import stretching
from .param import param
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

    def write(self, prm, **kwargs):
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
                val.x3d.write(prm, **kwargs)

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
            **Integrated**

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
            output_dtypes=[param["mytype"]],
            kwargs={"x": self._data_set[dim], "axis": -1, "initial": 0.0},
        )

    def simps(self, *args):
        """Integrate :obj:`xarray.Dataset` in direction(s) ``args`` using the
        composite Simpson’s rule.
        It is a wrapper for :obj:`scipy.integrate.simps`.

        Parameters
        ----------
        arg : str or sequence of str
            Dimension(s) to compute integration.

        Returns
        -------
        :obj:`xarray.Dataset`
            **Integrated**

        Raises
        -------
        ValueError
            args must be valid dimensions in the dataset

        Examples
        -------

        >>> ds.x3d.simps('x')
        >>> ds.x3d.simps('t')
        >>> ds.x3d.simps('x', 'y', 'z')

        """

        from scipy.integrate import simps

        def integrate(dataset, dim):
            return xr.apply_ufunc(
                simps,
                dataset,
                input_core_dims=[[dim]],
                dask="parallelized",
                output_dtypes=[param["mytype"]],
                kwargs={"x": self._data_set[dim], "axis": -1},
            )

        for var in args:
            if not var in self._data_set.dims:
                raise ValueError(
                    f'Invalid value for "args", it should be a valid dimension'
                )

        for i, var in enumerate(args):
            if i == 0:
                I = integrate(self._data_set, var)
            else:
                I = integrate(I, var)

        return I

    def pencil_decomp(self, *args):
        """Coerce all arrays in this dataset into dask arrays.

        It applies ``chunk=-1`` for all coordinates listed in ``args``, which
        means no decompositon, and ``'auto'`` to the others, resulting in a
        pencil decomposition for parallel evaluation.

        For customized ``chunks`` adjust, see :obj:`xarray.Dataset.chunk`.

        Parameters
        ----------
        arg : str or sequence of str
            Dimension(s) to apply no decompositon.

        Returns
        -------
        :obj:`xarray.Dataset`
            **chunked**

        Raises
        -------
        ValueError
            args must be valid dimensions in the dataset

        Examples
        -------

        >>> ds.x3d.pencil_decomp('x') # Pencil decomposition
        >>> ds.x3d.pencil_decomp('t')
        >>> ds.x3d.pencil_decomp('y', 'z') # Slab decomposition

        """

        for var in args:
            if not var in self._data_set.dims:
                raise ValueError(
                    f'Invalid value for "args", it should be a valid dimension'
                )

        chunks = {}
        for var in self._data_set.dims:
            if var in args:
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

    def write(
        self,
        prm,
        filename=None,
        steep="ioutput",
        progress_function="",
        separator="-",
        fileformat=".bin",
    ):
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
        if filename is not None:
            # If n is a dimension (for scalar), call write recursively to save
            # phi1, phi2, phi3, for instance.
            if "n" in self._data_array.dims:
                for n in self._data_array.n:
                    self._data_array.sel(n=n).x3d.write(
                        prm,
                        filename + str(n.values + 1),
                        steep=steep,
                        progress_function=progress_function,
                        fileformat=fileformat,
                    )
            # If t is a dimension (for time), call write recursively to save
            # ux-000000000.bin, ux-000000100.bin, ux-000000200.bin, for instance.
            elif "t" in self._data_array.dims:

                if not progress_function:
                    progress_function = lambda x, **kwargs: x
                elif progress_function.lower() == "notebook":
                    from tqdm.notebook import tqdm as progress_function
                elif progress_function.lower() == "tqdm":
                    from tqdm import tqdm as progress_function
                else:
                    raise ValueError(
                        'Invalid value for progress_function, try again with "", "notebook" or "tqdm".'
                    )

                i1, i2 = prm.ifilenameformat[2:-1].split(".")
                k = 0
                # <To do> put the logical tests outside the loop
                for t in progress_function(self._data_array.t.values, desc=filename):
                    # New filename format, see https://github.com/fschuch/Xcompact3d/issues/3
                    if prm.filenamedigits == 0:
                        num = str(int(t / prm.dt)).zfill(int(i1))
                    # Previous filename format
                    elif prm.filenamedigits == 1:
                        num = str(int(t / prm.dt / getattr(prm, steep))).zfill(int(i1))
                    self._data_array.isel(t=k).x3d.write(
                        prm,
                        filename + separator + num,
                        steep=steep,
                        progress_function="",
                        fileformat=fileformat,
                    )
                    k += 1
            # and finally writes to the disc
            else:
                align = []
                for i in reversed(sorted(self._data_array.dims)):
                    align.append(self._data_array.get_axis_num(i))
                self._data_array.values.astype(param["mytype"]).transpose(align).tofile(
                    filename + fileformat
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

    def simps(self, *args):
        """Integrate :obj:`xarray.DataArray` in direction(s) ``args`` using the
        composite Simpson’s rule.
        It is a wrapper for :obj:`scipy.integrate.simps`.

        Parameters
        ----------
        arg : str or sequence of str
            Dimension(s) to compute integration.

        Returns
        -------
        :obj:`xarray.DataArray`
            Integrated

        Raises
        -------
        ValueError
            args must be valid dimensions in the data array

        Examples
        -------

        >>> da.x3d.simps('x')
        >>> da.x3d.simps('t')
        >>> da.x3d.simps('x', 'y', 'z')

        """
        ds = self._data_array._to_temp_dataset().x3d.simps(*args)
        return self._data_array._from_temp_dataset(ds)

    def pencil_decomp(self, *args):
        """Coerce the data array into dask array.

        It applies ``chunk=-1`` for all coordinates listed in ``args``, which
        means no decompositon, and ``'auto'`` to the others, resulting in a
        pencil decomposition for parallel evaluation.

        For customized ``chunks`` adjust, see :obj:`xarray.DataArray.chunk`.

        Parameters
        ----------
        arg : str or sequence of str
            Dimension(s) to apply no decompositon.

        Returns
        -------
        :obj:`xarray.DataArray`
            **chunked**

        Raises
        -------
        ValueError
            args must be valid dimensions in the data array

        Examples
        -------

        >>> da.x3d.pencil_decomp('x') # Pencil decomposition
        >>> da.x3d.pencil_decomp('t')
        >>> da.x3d.pencil_decomp('y', 'z') # Slab decomposition

        """
        ds = self._data_array._to_temp_dataset().x3d.pencil_decomp(*args)
        return self._data_array._from_temp_dataset(ds)

    def first_derivative(self, dim):
        """Compute first derivative with the 4th order accurate centered scheme.

        It is fully functional with all boundary conditions available on
        Xcompact3d and stretched mesh in y direction.
        The **atribute** ``BC`` is used to store Boundary Condition information
        in a dictionary (see examples), default is ``ncl1 = ncln = 2`` and
        ``npaire = 1``.

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
        ...         'npaire': 1,
        ...         'istret': 0,
        ...         'beta': 1.0
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
        :obj:`xcompact3d_toolbox.Parameters.read_field` and
        :obj:`xcompact3d_toolbox.Parameters.read_all_fields` or initialized at
        :obj:`xcompact3d_toolbox.sandbox.init_dataset`.
        """

        if dim not in self._Dx:
            try:
                ncl1 = self._data_array.attrs["BC"][dim]["ncl1"]
                ncln = self._data_array.attrs["BC"][dim]["ncln"]
                npaire = self._data_array.attrs["BC"][dim]["npaire"]
            except:
                ncl1, ncln, npaire = 2, 2, 1

            n = self._data_array[dim].size
            m = n if ncl1 == 0 and ncln == 0 else n - 1
            d = (self._data_array[dim][-1] - self._data_array[dim][0]).values / m
            self._Dx[dim] = FirstDerivative(n, d, ncl1, ncln, npaire)

        try:
            istret = self._data_array.attrs["BC"][dim]["istret"]
            beta = self._data_array.attrs["BC"][dim]["beta"]
        except:
            istret = 0
            beta = 1.0

        if istret == 0:

            return xr.apply_ufunc(
                lambda f: self._Dx[dim].dot(f),
                self._data_array,
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                dask="parallelized",
                vectorize=True,
                output_dtypes=[param["mytype"]],
            )

        else:

            yly = (self._data_array[dim][-1] - self._data_array[dim][0]).values

            yp, ppy, pp2y, pp4y = stretching(istret, beta, yly, m, n)

            da_ppy = xr.DataArray(ppy, coords=[self._data_array[dim]], name="ppy")

            return da_ppy * xr.apply_ufunc(
                lambda f: self._Dx[dim].dot(f),
                self._data_array,
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                dask="parallelized",
                vectorize=True,
                output_dtypes=[param["mytype"]],
            )

    def second_derivative(self, dim):
        """Compute second derivative with the 4th order accurate centered scheme.

        It is fully functional with all boundary conditions available on
        Xcompact3d and stretched mesh in y direction.
        The **atribute** ``BC`` is used to store Boundary Condition information
        in a dictionary (see examples), default is ``ncl1 = ncln = 2`` and
        ``npaire = 1``.

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
        ...         'istret': 0,
        ...         'beta': 1.0
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
        :obj:`xcompact3d_toolbox.io.readfield` or initialized at
        :obj:`xcompact3d_toolbox.sendbox.init_dataset`.
        """
        if dim not in self._Dxx:
            try:
                ncl1 = self._data_array.attrs["BC"][dim]["ncl1"]
                ncln = self._data_array.attrs["BC"][dim]["ncln"]
                npaire = self._data_array.attrs["BC"][dim]["npaire"]
            except:
                ncl1, ncln, npaire = 2, 2, 1

            n = self._data_array[dim].size
            m = n if ncl1 == 0 and ncln == 0 else n - 1
            d = (self._data_array[dim][-1] - self._data_array[dim][0]).values / m
            self._Dxx[dim] = SecondDerivative(n, d, ncl1, ncln, npaire)

        try:
            istret = self._data_array.attrs["BC"][dim]["istret"]
            beta = self._data_array.attrs["BC"][dim]["beta"]
        except:
            istret = 0
            beta = 1.0

        if istret == 0:

            return xr.apply_ufunc(
                lambda f: self._Dxx[dim].dot(f),
                self._data_array,
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                dask="parallelized",
                vectorize=True,
                output_dtypes=[param["mytype"]],
            )

        else:

            yly = (self._data_array[dim][-1] - self._data_array[dim][0]).values

            yp, ppy, pp2y, pp4y = stretching(istret, beta, yly, m, n)

            da_pp2y = xr.DataArray(pp2y, coords=[self._data_array[dim]], name="pp2y")
            da_pp4y = xr.DataArray(pp4y, coords=[self._data_array[dim]], name="pp4y")

            return da_pp2y * xr.apply_ufunc(
                lambda f: self._Dxx[dim].dot(f),
                self._data_array,
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                dask="parallelized",
                vectorize=True,
                output_dtypes=[param["mytype"]],
            ) - da_pp4y * self._data_array.x3d.first_derivative(dim)
