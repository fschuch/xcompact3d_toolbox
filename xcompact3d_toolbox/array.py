"""The data structure is provided by `xarray`_, that introduces labels in the
form of dimensions, coordinates and attributes on top of raw `NumPy`_-like
arrays, which allows for a more intuitive, more concise, and less error-prone
developer experience.

See `xarray`_'s User Guide for a complete overview about
its data structures and built-in functions for indexing, selecting, computing,
plotting and much more.
It integrates tightly with `dask`_ for parallel computing.

Consider using hvPlot_ to explore your data interactively,
see how to plot `Gridded Data`_.

Xcompact3d-toolbox adds extra functions on top of :obj:`xarray.DataArray`
and :obj:`xarray.Dataset`, all the details are described below.

.. _dask: https://dask.org/
.. _numpy: https://numpy.org/
.. _xarray: http://docs.xarray.dev/en/stable
.. _hvPlot : https://hvplot.holoviz.org/
.. _`Gridded Data` : https://hvplot.holoviz.org/user_guide/Gridded_Data.html

"""

import xarray as xr
from deprecated.sphinx import deprecated, versionadded

try:
    from scipy.integrate import cumulative_trapezoid, simpson
except ImportError:
    # to be removed in Scipy 1.14.0
    from scipy.integrate import cumtrapz as cumulative_trapezoid  # no cov
    from scipy.integrate import simps as simpson  # no cov

from xcompact3d_toolbox.derive import first_derivative, second_derivative
from xcompact3d_toolbox.mesh import Istret, _stretching
from xcompact3d_toolbox.param import param


@xr.register_dataset_accessor("x3d")
class X3dDataset:
    """An accessor with extra utilities for :obj:`xarray.Dataset`."""

    def __init__(self, data_set):
        self._data_set = data_set

    @deprecated(
        version="1.2.0",
        reason=(
            "`SciPy 1.12.0 <https://docs.scipy.org/doc/scipy/release/1.12.0-notes.html#deprecated-features>`_ "
            "deprecated :obj:`scipy.integrate.cumtrapz` in favor of :obj:`scipy.integrate.cumulative_trapezoid`. "
            "Since it is running under the hood, it makes sense to rename the method on xcompact3d-toolbox as well."
        ),
        category=FutureWarning,
    )
    def cumtrapz(self, dim):
        """See :obj:`cumulative_trapezoid`."""
        return self.cumulative_trapezoid(dim)

    @versionadded(version="1.2.0")
    def cumulative_trapezoid(self, dim):
        """Cumulatively integrate all arrays in this dataset
        in direction ``dim`` using the composite trapezoidal rule.
        It is a wrapper for :obj:`scipy.integrate.cumulative_trapezoid`.
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

        >>> ds.x3d.cumulative_trapezoid("t")

        """

        return xr.apply_ufunc(
            cumulative_trapezoid,
            self._data_set,
            input_core_dims=[[dim]],
            output_core_dims=[[dim]],
            dask="parallelized",
            output_dtypes=[param["mytype"]],
            kwargs={"x": self._data_set[dim], "axis": -1, "initial": 0.0},
        )

    @deprecated(
        version="1.2.0",
        reason=(
            "`SciPy 1.12.0 <https://docs.scipy.org/doc/scipy/release/1.12.0-notes.html#deprecated-features>`_ "
            "deprecated :obj:`scipy.integrate.simps` in favor of :obj:`scipy.integrate.simpson`. "
            "Since it is running under the hood, it makes sense to rename the method on xcompact3d-toolbox as well."
        ),
        category=FutureWarning,
    )
    def simps(self, *args):
        """See :obj:`simpson`."""
        return self.simpson(*args)

    @versionadded(version="1.2.0")
    def simpson(self, *args):
        """Integrate all arrays in this dataset in direction(s) ``args``
        using the composite Simpson's rule.
        It is a wrapper for :obj:`scipy.integrate.simpson`.

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

        >>> ds.x3d.simpson("x")
        >>> ds.x3d.simpson("t")
        >>> ds.x3d.simpson("x", "y", "z")

        """

        def integrate(dataset, dim):
            return xr.apply_ufunc(
                simpson,
                dataset,
                input_core_dims=[[dim]],
                dask="parallelized",
                output_dtypes=[param["mytype"]],
                kwargs={"x": self._data_set[dim], "axis": -1},
            )

        for var in args:
            if var not in self._data_set.dims:
                msg = 'Invalid value for "args", it should be a valid dimension'
                raise ValueError(msg)

        result = 0
        for i, var in enumerate(args):
            result = integrate(self._data_set, var) if i == 0 else integrate(result, var)

        return result

    def pencil_decomp(self, *args):
        """Coerce all arrays in this dataset into dask arrays.

        It applies ``chunk=-1`` for all coordinates listed in ``args``, which
        means no decomposition, and ``"auto"`` to the others, resulting in a
        pencil decomposition for parallel evaluation.

        For customized ``chunks`` adjust, see :obj:`xarray.Dataset.chunk`.

        Parameters
        ----------
        arg : str or sequence of str
            Dimension(s) to apply no decomposition.

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

        >>> ds.x3d.pencil_decomp("x")  # Pencil decomposition
        >>> ds.x3d.pencil_decomp("t")
        >>> ds.x3d.pencil_decomp("y", "z")  # Slab decomposition

        """

        if not set(args).issubset(set(self._data_set.dims)):
            msg = 'Invalid value for "args", it should be a valid dimension'
            raise ValueError(msg)

        return self._data_set.chunk(chunks={dim: "auto" if dim in args else -1 for dim in self._data_set.dims})


@xr.register_dataarray_accessor("x3d")
class X3dDataArray:
    """An accessor with extra utilities for :obj:`xarray.DataArray`."""

    def __init__(self, data_array):
        self._data_array = data_array

        self._Dx = {}
        self._Dxx = {}

    @deprecated(
        version="1.2.0",
        reason=(
            "`SciPy 1.12.0 <https://docs.scipy.org/doc/scipy/release/1.12.0-notes.html#deprecated-features>`_ "
            "deprecated :obj:`scipy.integrate.cumtrapz` in favor of :obj:`scipy.integrate.cumulative_trapezoid`. "
            "Since it is running under the hood, it makes sense to rename the method on xcompact3d-toolbox as well."
        ),
        category=FutureWarning,
    )
    def cumtrapz(self, dim):
        """See :obj:`cumulative_trapezoid`."""
        return self.cumulative_trapezoid(dim)

    @versionadded(version="1.2.0")
    def cumulative_trapezoid(self, dim):
        """Cumulatively integrate :obj:`xarray.DataArray` in direction ``dim``
        using the composite trapezoidal rule.
        It is a wrapper for :obj:`scipy.integrate.cumulative_trapezoid`.
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

        >>> da.x3d.cumulative_trapezoid("t")

        """
        ds = self._data_array._to_temp_dataset().x3d.cumulative_trapezoid(dim)  # noqa: SLF001
        return self._data_array._from_temp_dataset(ds)  # noqa: SLF001

    @deprecated(
        version="1.2.0",
        reason=(
            "`SciPy 1.12.0 <https://docs.scipy.org/doc/scipy/release/1.12.0-notes.html#deprecated-features>`_ "
            "deprecated :obj:`scipy.integrate.simps` in favor of :obj:`scipy.integrate.simpson`. "
            "Since it is running under the hood, it makes sense to rename the method on xcompact3d-toolbox as well."
        ),
        category=FutureWarning,
    )
    def simps(self, *args):
        """See :obj:`simpson`."""
        return self.simpson(*args)

    @versionadded(version="1.2.0")
    def simpson(self, *args):
        """Integrate :obj:`xarray.DataArray` in direction(s) ``args`` using the
        composite Simpson's rule.
        It is a wrapper for :obj:`scipy.integrate.simpson`.

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

        >>> da.x3d.simpson("x")
        >>> da.x3d.simpson("t")
        >>> da.x3d.simpson("x", "y", "z")

        """
        ds = self._data_array._to_temp_dataset().x3d.simpson(*args)  # noqa: SLF001
        return self._data_array._from_temp_dataset(ds)  # noqa: SLF001

    def pencil_decomp(self, *args):
        """Coerce the data array into dask array.

        It applies ``chunk=-1`` for all coordinates listed in ``args``, which
        means no decomposition, and ``'auto'`` to the others, resulting in a
        pencil decomposition for parallel evaluation.

        For customized ``chunks`` adjust, see :obj:`xarray.DataArray.chunk`.

        Parameters
        ----------
        arg : str or sequence of str
            Dimension(s) to apply no decomposition.

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

        >>> da.x3d.pencil_decomp("x")  # Pencil decomposition
        >>> da.x3d.pencil_decomp("t")
        >>> da.x3d.pencil_decomp("y", "z")  # Slab decomposition

        """
        ds = self._data_array._to_temp_dataset().x3d.pencil_decomp(*args)  # noqa: SLF001
        return self._data_array._from_temp_dataset(ds)  # noqa: SLF001

    def first_derivative(self, dim):
        """Compute first derivative with the 4th order accurate centered scheme.

        It is fully functional with all boundary conditions available on
        XCompact3d and stretched mesh in the vertical direction (y).
        The **attribute** ``BC`` is used to store Boundary Condition information
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
        >>> da.x3d.first_derivative("x")

        or just:

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> da.attrs["BC"] = prm.get_boundary_condition("ux")
        >>> da.x3d.first_derivative("x")

        """

        if dim not in self._Dx:
            try:
                ncl1 = self._data_array.attrs["BC"][dim]["ncl1"]
                ncln = self._data_array.attrs["BC"][dim]["ncln"]
                npaire = self._data_array.attrs["BC"][dim]["npaire"]
            except KeyError:
                ncl1, ncln, npaire = 2, 2, 1

            n = self._data_array[dim].size
            m = n if ncl1 == 0 and ncln == 0 else n - 1
            d = (self._data_array[dim][-1] - self._data_array[dim][0]).values / m
            self._Dx[dim] = first_derivative(n, d, ncl1, ncln, npaire)

        try:
            istret = self._data_array.attrs["BC"][dim]["istret"]
            beta = self._data_array.attrs["BC"][dim]["beta"]
        except KeyError:
            istret = Istret.NO_REFINEMENT
            beta = 1.0

        if istret == Istret.NO_REFINEMENT:
            return xr.apply_ufunc(
                lambda f: self._Dx[dim].dot(f),
                self._data_array,
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                dask="parallelized",
                vectorize=True,
                output_dtypes=[param["mytype"]],
            )

        yly = (self._data_array[dim][-1] - self._data_array[dim][0]).values

        _, ppy, _, _ = _stretching(istret, beta, yly, m, n)

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
        The **attribute** ``BC`` is used to store Boundary Condition information
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
        >>> da.x3d.second_derivative("x")

        or just:

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> da.attrs["BC"] = prm.get_boundary_condition("ux")
        >>> da.x3d.second_derivative("x")
        """
        if dim not in self._Dxx:
            try:
                ncl1 = self._data_array.attrs["BC"][dim]["ncl1"]
                ncln = self._data_array.attrs["BC"][dim]["ncln"]
                npaire = self._data_array.attrs["BC"][dim]["npaire"]
            except KeyError:
                ncl1, ncln, npaire = 2, 2, 1

            n = self._data_array[dim].size
            m = n if ncl1 == 0 and ncln == 0 else n - 1
            d = (self._data_array[dim][-1] - self._data_array[dim][0]).values / m
            self._Dxx[dim] = second_derivative(n, d, ncl1, ncln, npaire)

        try:
            istret = self._data_array.attrs["BC"][dim]["istret"]
            beta = self._data_array.attrs["BC"][dim]["beta"]
        except KeyError:
            istret = Istret.NO_REFINEMENT
            beta = 1.0

        if istret == Istret.NO_REFINEMENT:
            return xr.apply_ufunc(
                lambda f: self._Dxx[dim].dot(f),
                self._data_array,
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                dask="parallelized",
                vectorize=True,
                output_dtypes=[param["mytype"]],
            )

        yly = (self._data_array[dim][-1] - self._data_array[dim][0]).values

        _, _, pp2y, pp4y = _stretching(istret, beta, yly, m, n)

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
