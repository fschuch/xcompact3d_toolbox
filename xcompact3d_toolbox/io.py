"""Useful objects to read and write the binary fields produced by XCompact3d."""

from __future__ import annotations

import glob
import io
import os
import os.path
import warnings
from typing import Iterator

import numpy as np
import pandas as pd
import traitlets
import xarray as xr
from tqdm.auto import tqdm

from xcompact3d_toolbox.mesh import Istret, Mesh3D
from xcompact3d_toolbox.param import param


class FilenameProperties(traitlets.HasTraits):
    """Filename properties are important to guarantee consistency for input/output operations.
    This class makes xcompact3d-toolbox work with different types of file names for the binary
    fields produced from the numerical simulations and their pre/postprocessing.

    Parameters
    ----------
    separator : str
        The string used as separator between the name of the variable and its numeration, it
        can be an empty string (default is ``"-"``).
    file_extension : str
        The file extension that identify the raw binary files from XCompact3d, it
        can be an empty string (default is ``".bin"``).
    number_of_digits : int
        The number of numerical digits used to identify the time series (default is ``3``).
    scalar_num_of_digits : int
        The number of numerical digits used to identify each scalar field (default is ``1``).

    Notes
    -----
        :obj:`FilenameProperties` is in fact an attribute of
        :obj:`xcompact3d_toolbox.io.Dataset`, so there is no
        need to initialize it manually for most of the common
        use cases.
    """

    separator = traitlets.Unicode(default_value="-")
    file_extension = traitlets.Unicode(default_value=".bin")
    number_of_digits = traitlets.Int(default_value=3, min=1)
    scalar_num_of_digits = traitlets.Int(default_value=1, min=1)

    def __init__(self, **kwargs):
        """Initializes the object.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the parameters, like :obj:`separator`, :obj:`file_extension` and so on.

        Raises
        ------
        KeyError
            Raises an error if the user tries to set an invalid parameter.

        Returns
        -------
        :obj:`xcompact3d_toolbox.io.FilenameProperties`
            Filename properties
        """
        super(FilenameProperties).__init__()

        self.set(**kwargs)

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for name in self.trait_names():
            if name.startswith("_"):
                continue
            string += f"    {name} = {getattr(self, name)!r},\n"
        string += ")"
        return string

    def set(self, **kwargs) -> None:
        """Set new values for any of the properties after the initialization.

        Parameters
        ----------
        **kwargs
            Keyword arguments for parameters, like :obj:`separator`, :obj:`file_extension` and so on.

        Raises
        ------
        KeyError
            Raises an error if the user tries to set an invalid parameter.

        Examples
        --------

        If the simulated fields are named like ``ux-000.bin``, they are in the default
        configuration, there is no need to specify filename properties. But just in case,
        it would be like:

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> prm.dataset.filename_properties.set(
        ...     separator="-", file_extension=".bin", number_of_digits=3
        ... )

        If the simulated fields are named like ``ux0000``, the parameters are:

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> prm.dataset.filename_properties.set(
        ...     separator="", file_extension="", number_of_digits=4
        ... )
        """
        for key, arg in kwargs.items():
            if key not in self.trait_names():
                msg = f"{key} is not a valid argument for FilenameProperties"
                raise KeyError(msg)
            setattr(self, key, arg)

    def get_filename_for_binary(self, prefix: str, counter: int, data_path="") -> str:
        """Get the filename for an array.

        Parameters
        ----------
        prefix : str
            Name of the array.
        counter : int or str
            The number that identifies this array, it can be the string
            ``"*"`` if the filename is going to be used with :obj:`glob.glob`.
        data_path : str
            Path to the folder where the data is stored.

        Returns
        -------
        str
            The filename

        Examples
        --------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> prm.dataset.filename_properties.set(
        ...     separator="-", file_extension=".bin", number_of_digits=3
        ... )
        >>> prm.dataset.filename_properties.get_filename_for_binary("ux", 10)
        'ux-010.bin'
        >>> prm.dataset.filename_properties.get_filename_for_binary("ux", "*")
        'ux-???.bin'

        >>> prm.dataset.filename_properties.set(
        ...     separator="", file_extension="", number_of_digits=4
        ... )
        >>> prm.dataset.filename_properties.get_filename_for_binary("ux", 10)
        'ux0010'
        >>> prm.dataset.filename_properties.get_filename_for_binary("ux", "*")
        'ux????'
        """
        if counter == "*":
            counter = "?" * self.number_of_digits
        filename = f"{prefix}{self.separator}{str(counter).zfill(self.number_of_digits)}{self.file_extension}"
        if data_path:
            filename = os.path.join(data_path, filename)
        return filename

    def get_info_from_filename(self, filename: str) -> tuple[int, str]:
        """Get information from the name of a binary file.

        Parameters
        ----------
        filename : str
            The name of the array.

        Returns
        -------
        tuple[int, str]
            A tuple with the name of the array and the number that identifies it.

        Examples
        --------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> prm.dataset.filename_properties.set(
        ...     separator="-", file_extension=".bin", number_of_digits=3
        ... )
        >>> prm.dataset.filename_properties.get_info_from_filename("ux-010.bin")
        (10, 'ux')

        >>> prm.dataset.filename_properties.set(
        ...     separator="", file_extension="", number_of_digits=4
        ... )
        >>> prm.dataset.filename_properties.get_info_from_filename("ux0010")
        (10, 'ux')

        """
        _filename = os.path.basename(filename)
        if self.file_extension:
            _filename = _filename[: -len(self.file_extension)]
        if self.separator:
            prefix, counter = _filename.split(self.separator)
        else:
            prefix = _filename[: -self.number_of_digits]
            counter = _filename[-self.number_of_digits :]
        return int(counter), prefix

    def get_num_from_filename(self, filename: str) -> int:
        """Same as :obj:`get_info_from_filename`, but just returns the number."""
        num, _ = self.get_info_from_filename(filename)
        return num

    def get_name_from_filename(self, filename: str) -> str:
        """Same as :obj:`get_info_from_filename`, but just returns the name."""
        _, name = self.get_info_from_filename(filename)
        return name


class Dataset(traitlets.HasTraits):
    """An object that reads and writes the raw binary files from XCompact3d on-demand.

    Parameters
    ----------
    data_path : str
        The path to the folder where the binary fields are located (default is ``"./data/"``).

    .. note::
        The default ``"./data/"`` is relative to the path to the parameters
        file when initialized from :obj:`xcompact3d_toolbox.parameters.ParametersExtras`.

    drop_coords : str
        If working with two-dimensional planes, specify which of the coordinates should be
        dropped, i.e., ``"x"``, ``"y"`` or ``"z"``, or leave it empty for 3D fields (default is ``""``).
    filename_properties : :obj:`FilenameProperties`
        Specifies filename properties for the binary files, like the separator, file extension and
        number of digits.
    set_of_variables : set
        The methods in this class will try to find all
        variables per snapshot, use this parameter
        to work with just a few specified variables if you
        need to speedup your application
        (default is an empty set).
    snapshot_counting : str
        The parameter that controls the number of timesteps used to produce the datasets
        (default is ``"ilast"``).
    snapshot_step : str
        The parameter that controls the number of timesteps between each snapshot, it is often
        ``"ioutput"`` or ``"iprocessing"`` (default is ``"ioutput"``).
    stack_scalar : bool
        When :obj:`True`, the scalar fields will be stacked in a new coordinate ``n``, otherwise returns one
        array per scalar fraction (default is :obj:`False`).
    stack_velocity : bool
        When :obj:`True`, the velocity will be stacked in a new coordinate ``i`` , otherwise returns one
        array per velocity component (default is :obj:`False`).

    Notes
    -----
    * :obj:`Dataset` is in fact an attribute of :obj:`xcompact3d_toolbox.parameters.ParametersExtras`,
      so there is no need to initialize it manually for most of the common use cases.

    * All arrays are wrapped into Xarray objects (:obj:`xarray.DataArray`
      or :obj:`xarray.Dataset`), take a look at `xarray's documentation`_,
      specially, see `Why xarray?`_
      Xarray has many useful methods for indexing, comparisons, reshaping
      and reorganizing, computations and plotting.

    * Consider using hvPlot_ to explore your data interactively,
      see how to plot `Gridded Data`_.

    .. _`xarray's documentation`: http://docs.xarray.dev/en/stable
    .. _`Why xarray?`: http://docs.xarray.dev/en/stable/why-xarray.html
    .. _hvPlot : https://hvplot.holoviz.org/
    .. _`Gridded Data` : https://hvplot.holoviz.org/user_guide/Gridded_Data.html
    """

    data_path = traitlets.Unicode(default_value="./data/")
    drop_coords = traitlets.Unicode(default_value="")
    filename_properties = traitlets.Instance(klass=FilenameProperties)
    set_of_variables = traitlets.Set()
    snapshot_counting = traitlets.Unicode(default_value="ilast")
    snapshot_step = traitlets.Unicode(default_value="ioutput")
    stack_scalar = traitlets.Bool(default_value=False)
    stack_velocity = traitlets.Bool(default_value=False)

    _mesh = traitlets.Instance(klass=Mesh3D)
    _prm = traitlets.Instance(klass="xcompact3d_toolbox.parameters.Parameters", allow_none=True)

    def __init__(self, **kwargs):
        """Initializes the Dataset class.

        Parameters
        ----------
        filename_properties : dict, optional
            Keyword arguments for :obj:`FilenamePropertie`, like
            :obj:`separator`, :obj:`file_extension` and so on.
        **kwargs
            Keyword arguments for the parameters, like :obj:`data_path`, :obj:`drop_coords` and so on.

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid parameter.

        Returns
        -------
        :obj:`Dataset`
            An object to read and write the raw binary files from XCompact3d on-demand.

        """

        self.set_of_variables = set()
        self.filename_properties = FilenameProperties()
        self._mesh = Mesh3D()
        self._prm = None

        self.set(**kwargs)

    def __call__(self, *args) -> Iterator[xr.Dataset]:
        """Yields selected snapshots, so the application can iterate over them,
        loading one by one, with the same arguments of a classic Python :obj:`range`.

        Parameters
        ----------
        *args : int
            Same arguments used for a :obj:`range`.

        Yields
        ------
        :obj:`xarray.Dataset`
            Dataset containing the arrays loaded from the disc with the appropriate dimensions,
            coordinates and attributes.

        Examples
        --------

        Initial setup:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="input.i3d")
        >>> prm.dataset.set(
        ...     filename_properties=dict(
        ...         separator="-", file_extension=".bin", number_of_digits=3
        ...     ),
        ...     stack_scalar=True,
        ...     stack_velocity=True,
        ... )

        Iterate over some snapshots, loading them one by one, with the same
        arguments of a classic Python :obj:`range`, for instance,
        from 0 to 100 with a step of 5:

          >>> for ds in prm.dataset(0, 101, 5):
          ...     vort = ds.uy.x3d.first_derivative("x") - ds.ux.x3d.first_derivative("y")
          ...     prm.dataset.write(data=vort, file_prefix="w3")
        """
        for t in range(*args):
            yield self.load_snapshot(t)

    def __getitem__(self, arg: int | slice | str) -> xr.DataArray | xr.Dataset:
        """Get specified items from the disc.

        .. note:: Make sure to have enough memory to load many files at the same time.

        Parameters
        ----------
        arg : :obj:`int` or :obj:`slice` or :obj:`str`
            Specifies the items to load from the disc, depending on the type of the argument:

            * :obj:`int` returns the specified snapshot in a :obj:`xarray.Dataset`.
              It is equivalent to :obj:`Dataset.load_snapshot`;
            * :obj:`slice` returns the specified snapshots in a :obj:`xarray.Dataset`;
            * :obj:`str` returns the entire time series for a given variable in a
              :obj:`xarray.DataArray`.
              It is equivalent to :obj:`Dataset.load_time_series`;

        Returns
        -------
        :obj:`xarray.Dataset` or :obj:`xarray.DataArray`
            Xarray objects containing values loaded from the disc with the appropriate dimensions,
            coordinates and attributes.

        Raises
        ------
        TypeError
            Raises type error if arg is not an integer, string or slice

        Examples
        --------

        Initial setup:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="input.i3d")
        >>> prm.dataset.set(
        ...     filename_properties=dict(
        ...         separator="-", file_extension=".bin", number_of_digits=3
        ...     ),
        ...     drop_coords="z",
        ...     stack_scalar=True,
        ...     stack_velocity=True,
        ... )

        * Load the entire time series for a given variable:

          >>> ux = prm.dataset["ux"]
          >>> uy = prm.dataset["uy"]
          >>> uz = prm.dataset["uz"]

          or organize them using a dataset:

          >>> dataset = xarray.Dataset()
          >>> for var in "ux uy uz".split():
          ...     dataset[var] = prm.dataset[var]

        * Load all variables from a given snapshot:

          >>> snapshot = prm.dataset[10]

        * Load many snapshots at once with a :obj:`slice`, for instance, from 0 to 100
          with a step of 10:

          >>> snapshots = prm.dataset[0:101:10]

        * Or simply load all snapshots at once (if you have enough memory):

          >>> snapshots = prm.dataset[:]

        """
        if isinstance(arg, int):
            return self.load_snapshot(arg)
        if isinstance(arg, slice):
            start, stop, step = arg.indices(len(self))
            return xr.concat((self.load_snapshot(t) for t in range(start, stop, step)), "t")
        if isinstance(arg, str):
            return self.load_time_series(arg)
        msg = "Dataset indices should be integers, string or slices"
        raise TypeError(msg)

    def __len__(self) -> int:
        """Make the dataset work with the Python function :obj:`len`.

        Returns
        -------
        int
            Total of snapshots as a function of :obj:`snapshot_counting` and :obj:`snapshot_step`.
        """
        # Test environment
        if self._prm is None:
            return 11
        return getattr(self._prm, self.snapshot_counting) // getattr(self._prm, self.snapshot_step) + 1

    def __iter__(self):
        """Yields all the snapshots, so the application can iterate over them.

        Yields
        ------
        :obj:`xarray.Dataset`
            Dataset containing the arrays loaded from the disc with the appropriate dimensions,
            coordinates and attributes.

        Examples
        --------

        Initial setup:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="input.i3d")
        >>> prm.dataset.set(
        ...     filename_properties=dict(
        ...         separator="-", file_extension=".bin", number_of_digits=3
        ...     ),
        ...     stack_scalar=True,
        ...     stack_velocity=True,
        ... )

        Iterate over all snapshots, loading them one by one:

        >>> for ds in prm.dataset:
        ...     vort = ds.uy.x3d.first_derivative("x") - ds.ux.x3d.first_derivative("y")
        ...     prm.dataset.write(data=vort, file_prefix="w3")
        """
        for t in range(len(self)):
            yield self.load_snapshot(t)

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for name in self.trait_names():
            if name.startswith("_"):
                continue
            string += f"    {name} = {getattr(self, name)!r},\n"
        string += ")"
        return string

    @property
    def _time_step(self) -> float:
        """Time step value between snapshots"""
        # test environment
        if self._prm is None:
            return 1.0
        return self._prm.dt * getattr(self._prm, self.snapshot_step)

    def set(self, **kwargs):
        """Set new values for any of the properties after the initialization.

        Parameters
        ----------
        filename_properties : dict, optional
            Keyword arguments for :obj:`FilenameProperties`, like
            :obj:`separator`, :obj:`file_extension` and so on.
        **kwargs
            Keyword arguments for the parameters, like :obj:`data_path`, :obj:`drop_coords` and so on.

        Raises
        ------
        KeyError
            Exception is raised when an Keyword arguments is not a valid parameter.

        Examples
        --------

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="input.i3d")
        >>> prm.dataset.set(
        ...     filename_properties=dict(
        ...         separator="-", file_extension=".bin", number_of_digits=3
        ...     ),
        ...     stack_scalar=True,
        ...     stack_velocity=True,
        ... )
        """

        if "filename_properties" in kwargs:
            self.filename_properties.set(**kwargs.pop("filename_properties"))

        for key, arg in kwargs.items():
            if key not in self.trait_names():
                msg = f"{key} is not a valid argument for Dataset"
                raise KeyError(msg)
            setattr(self, key, arg)

    def load_array(self, filename: str, attrs: dict | None = None, *, add_time: bool = True) -> type[xr.DataArray]:
        """This method reads a binary field from XCompact3d with :obj:`numpy.fromfile`
        and wraps it into a :obj:`xarray.DataArray` with the appropriate dimensions,
        coordinates and attributes.

        Parameters
        ----------
        filename : str
            Name of the file.
        add_time : bool, optional
            Add time as a coordinate (default is :obj:`True`).
        attrs : dict_like, optional
            Attributes to assign to the new instance :obj:`xarray.DataArray`.

        Returns
        -------
        :obj:`xarray.DataArray`
            Data array containing values loaded from the disc.

        Examples
        --------

        Initial setup:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="input.i3d")
        >>> prm.dataset.set(
        ...     filename_properties=dict(
        ...         separator="-", file_extension=".bin", number_of_digits=3
        ...     ),
        ...     stack_scalar=True,
        ...     stack_velocity=True,
        ... )

        Load one array from the disc:

        >>> ux = prm.dataset.load_array("ux-000.bin")

        .. versionchanged:: 1.2.0
            The argument ``add_time`` changed to keyword-only.

        """

        coords = self._mesh.drop(*self.drop_coords)

        if add_time:
            time_int, name = self.filename_properties.get_info_from_filename(filename)
            coords["t"] = [param["mytype"](self._time_step * time_int)]
        else:
            name = None

        # We obtain the shape for np.fromfile from the coordinates
        shape = [len(value) for value in coords.values()]

        # This is necessary if the file is a link
        if os.path.islink(filename):
            filename = os.readlink(filename)

        # Finally, we read the array and wrap it into a xarray object
        return xr.DataArray(
            np.fromfile(filename, dtype=param["mytype"]).reshape(shape, order="F"),
            dims=coords.keys(),
            coords=coords,
            name=name,
            attrs=attrs,
        )

    def load_snapshot(
        self,
        numerical_identifier: int,
        list_of_variables: list | None = None,
        stack_scalar: bool | None = None,
        stack_velocity: bool | None = None,
        *,
        add_time: bool = True,
    ) -> type[xr.Dataset]:
        """Load the variables for a given snapshot.

        Parameters
        ----------
        numerical_identifier : int
            The number of the snapshot.
        list_of_variables : list, optional
            List of variables to be loaded, if None, it uses :obj:`Dataset.set_of_variables`,
            if :obj:`Dataset.set_of_variables` is empty, it automatically loads all arrays from
            this snapshot, by default None.
        add_time : bool, optional
            Add time as a coordinate, by default True.
        stack_scalar : bool, optional
            When true, the scalar fields will be stacked in a new coordinate ``n``, otherwise returns one array per
            scalar fraction. If none, it uses :obj:`Dataset.stack_scalar`, by default None.
        stack_velocity : bool, optional
            When true, the velocity will be stacked in a new coordinate ``i``, otherwise returns one array per velocity
            component. If none, it uses :obj:`Dataset.stack_velocity`, by default None.

        Returns
        -------
        :obj:`xarray.Dataset`
            Dataset containing the arrays loaded from the disc with the appropriate dimensions,
            coordinates and attributes.

        Raises
        ------
        IOError
            Raises IO error if it does not find any variable for this snapshot.

        Examples
        --------

        Initial setup:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="input.i3d")
        >>> prm.dataset.set(
        ...     filename_properties=dict(
        ...         separator="-", file_extension=".bin", number_of_digits=3
        ...     ),
        ...     stack_scalar=True,
        ...     stack_velocity=True,
        ... )

        Load all variables from a given snapshot:

        >>> snapshot = prm.dataset.load_snapshot(10)

        or just

        >>> snapshot = prm.dataset[10]

        """
        dataset = xr.Dataset()

        if list_of_variables is not None:
            set_of_variables = set(list_of_variables)
        elif self.set_of_variables:
            set_of_variables = self.set_of_variables.copy()
        else:
            target_filename = self.filename_properties.get_filename_for_binary("*", numerical_identifier)

            list_of_variables = glob.glob(os.path.join(self.data_path, target_filename))
            set_of_variables = {self.filename_properties.get_name_from_filename(i) for i in list_of_variables}

        if not set_of_variables:
            msg = f"No file found corresponding to {self.data_path}/{target_filename}"
            raise OSError(msg)

        if stack_scalar is None:
            stack_scalar = self.stack_scalar
        if stack_velocity is None:
            stack_velocity = self.stack_velocity

        def stack_variables(variables, **kwargs):
            return self.load_snapshot(
                numerical_identifier=numerical_identifier,
                list_of_variables=variables,
                add_time=add_time,
                **kwargs,
            )

        def is_scalar(name):
            prefix = "phi"
            if len(name) != (len(prefix) + self.filename_properties.scalar_num_of_digits):
                return False
            if not name.startswith(prefix):
                return False
            if not name[len(prefix) :].isdigit():
                return False
            return True

        def is_velocity(name):
            return name in {"ux", "uy", "uz"}

        if stack_scalar:
            scalar_variables = sorted(
                filter(
                    is_scalar,
                    set_of_variables,
                )
            )

            if scalar_variables:
                dataset["phi"] = (
                    stack_variables(scalar_variables, stack_scalar=False)
                    .to_array(dim="n")
                    .assign_coords(n=[int(var[-1]) for var in scalar_variables])
                )
                set_of_variables -= set(scalar_variables)

        if stack_velocity:
            velocity_variables = sorted(filter(is_velocity, set_of_variables))
            if velocity_variables:
                dataset["u"] = (
                    stack_variables(velocity_variables, stack_velocity=False)
                    .to_array(dim="i")
                    .assign_coords(i=[var[-1] for var in velocity_variables])
                )
                set_of_variables -= set(velocity_variables)

        for var in sorted(set_of_variables):
            filename = self.filename_properties.get_filename_for_binary(var, numerical_identifier)
            filename = os.path.join(self.data_path, filename)
            dataset[var] = self.load_array(filename=filename, add_time=add_time)

        return dataset

    def load_time_series(self, array_prefix: str) -> type[xr.DataArray]:
        """Load the entire time series for a given variable.

        .. note:: Make sure to have enough memory to load all files at the same time.

        Parameters
        ----------
        array_prefix : str
            Name of the variable, for instance ``ux``, ``uy``, ``uz``, ``pp``, ``phi1``.

        Returns
        -------
        :obj:`xarray.DataArray`
            DataArray containing the time series loaded from the disc,
            with the appropriate dimensions, coordinates and attributes.

        Raises
        ------
        IOError
            Raises IO error if it does not find any snapshot for this variable.

        Examples
        --------

        Initial setup:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="input.i3d")
        >>> prm.dataset.set(
        ...     filename_properties=dict(
        ...         separator="-", file_extension=".bin", number_of_digits=3
        ...     ),
        ...     stack_scalar=True,
        ...     stack_velocity=True,
        ... )

        Load the entire time series for a given variable:

        >>> ux = prm.dataset.load_time_series("ux")
        >>> uy = prm.dataset.load_time_series("uy")
        >>> uz = prm.dataset.load_time_series("uz")

        or just:

        >>> ux = prm.dataset["ux"]
        >>> uy = prm.dataset["uy"]
        >>> uz = prm.dataset["uz"]

        You can organize them using a dataset:

        >>> dataset = xarray.Dataset()
        >>> for var in "ux uy uz".split():
        ...     dataset[var] = prm.dataset[var]

        """
        target_filename = self.filename_properties.get_filename_for_binary(array_prefix, "*")
        filename_pattern = os.path.join(self.data_path, target_filename)
        filename_list = sorted(glob.glob(filename_pattern))

        if not filename_list:
            msg = f"No file was found corresponding to {filename_pattern}."
            raise OSError(msg)

        return xr.concat(
            (self.load_array(file, add_time=True) for file in tqdm(filename_list, desc=filename_pattern)),
            dim="t",
        )

    def load_wind_turbine_data(self, file_pattern: str | None = None) -> type[xr.Dataset]:
        """Load the data produced by wind turbine simulations.

        .. note:: This feature is experimental

        Parameters
        ----------
        file_pattern : str, optional
            A filename pattern used to locate the files with :obj:`glob.iglob`.
            If None, it is obtained from ``datapath``, i.g.,
            if ``datapath = ./examples/Wind-Turbine/data`` then
            ``file_pattern = ./examples/Wind-Turbine/data/../*.perf``.
            By default None.

        Returns
        -------
        :obj:`xarray.Dataset`
            A dataset with all variables as a function of the time

        Examples
        --------

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="NREL-5MW.i3d")
        >>> ds = prm.dataset.load_wind_turbine_data()
        >>> ds
        <xarray.Dataset>
        Dimensions:          (t: 21)
        Coordinates:
        * t                (t) float64 0.0 2.0 4.0 6.0 8.0 ... 34.0 36.0 38.0 40.0
        Data variables: (12/14)
            Number of Revs   (t) float64 0.0 0.4635 0.9281 1.347 ... 7.374 7.778 8.181
            GeneratorSpeed   (t) float64 0.0 149.3 133.6 123.2 ... 122.9 122.9 123.0
            GeneratorTorque  (t) float64 0.0 2.972e+04 3.83e+04 ... 4.31e+04 4.309e+04
            BladePitch1      (t) float64 0.0 12.0 14.21 13.21 ... 11.44 11.44 11.44
            BladePitch2      (t) float64 0.0 12.0 14.21 13.21 ... 11.44 11.44 11.44
            BladePitch3      (t) float64 0.0 12.0 14.21 13.21 ... 11.44 11.44 11.44
            ...               ...
            Ux               (t) float64 0.0 15.0 15.0 15.0 15.0 ... 15.0 15.0 15.0 15.0
            Uy               (t) float64 0.0 -1.562e-05 2.541e-05 ... 7.28e-07 7.683e-07
            Uz               (t) float64 0.0 1.55e-06 ... -1.828e-06 2.721e-06
            Thrust           (t) float64 9.39e+05 1.826e+05 ... 4.084e+05 4.066e+05
            Torque           (t) float64 8.78e+06 1.268e+06 ... 4.231e+06 4.203e+06
            Power            (t) float64 1.112e+07 1.952e+06 ... 5.362e+06 5.328e+06
        """

        def get_dataset(filename):
            time = os.path.basename(filename).split("_")[0]

            time = float(time) * self._prm.iturboutput * self._prm.dt

            ds = xr.Dataset(coords={"t": [time]})
            ds.t.attrs = {"name": "t", "long_name": "Time", "units": "s"}

            for name, values in pd.read_csv(filename).to_dict().items():
                ds[name.strip()] = xr.DataArray(
                    data=np.float64(values[1]),
                    coords=ds.t.coords,
                    attrs={"units": values[0].strip()[1:-1]},
                )

            return ds

        if file_pattern is None:
            file_pattern = os.path.join(self.data_path, "..", "*.perf")

        filenames = glob.iglob(file_pattern)

        return xr.concat((get_dataset(file) for file in filenames), dim="t").sortby("t")

    def write(self, data: xr.DataArray | xr.Dataset, file_prefix: str | None = None):
        """Write an array or dataset to raw binary files on the disc, in the
        same order that Xcompact3d would do, so they can be easily read with
        2DECOMP.

        In order to avoid overwriting any relevant field, only the variables in a dataset
        with an **attribute** called ``file_name`` will be written.

        Coordinates are properly aligned before writing.

        If ``n`` is a valid coordinate (for scalar fractions) in the array, one
        numerated binary file will be written for each scalar field.

        If ``i`` is a valid coordinate (for velocity components) in the array, one
        binary file will be written for each of them (x, y or z).

        If ``t`` is a valid coordinate (for time) in the array, one numerated
        binary file will be written for each available time.

        Parameters
        ----------
        data : :obj:`xarray.DataArray` or :obj:`xarray.Dataset`
            Data to be written
        file_prefix : str, optional
            filename prefix for the array, if data is :obj:`xarray.DataArray`, by default None

        Raises
        ------
        IOError
            Raises IO error data is not an :obj:`xarray.DataArray` or :obj:`xarray.Dataset`.

        Examples
        --------

        Initial setup:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="input.i3d")
        >>> prm.dataset.set(
        ...     filename_properties=dict(
        ...         separator="-", file_extension=".bin", number_of_digits=3
        ...     ),
        ...     stack_scalar=True,
        ...     stack_velocity=True,
        ... )

        * From a dataset, write only the variables with the attribute ``file_name``,
          notice that ``ux`` and ``uy`` will not be overwritten because them do not
          have the attribute ``file_name``:

            >>> for ds in prm.dataset:
            ...     ds["vort"] = ds.uy.x3d.first_derivative("x") - ds.ux.x3d.first_derivative(
            ...         "y"
            ...     )
            ...     ds["vort"].attrs["file_name"] = "vorticity"
            ...     prm.dataset.write(ds)

        * Write an array:

          >>> for ds in prm.dataset:
          ...     vort = ds.uy.x3d.first_derivative("x") - ds.ux.x3d.first_derivative("y")
          ...     vort.attrs["file_name"] = "vorticity"
          ...     prm.dataset.write(vort)

          or

          >>> for ds in prm.dataset:
          ...     vort = ds.uy.x3d.first_derivative("x") - ds.ux.x3d.first_derivative("y")
          ...     prm.dataset.write(data=vort, file_prefix="vorticity")

        .. note :: It is not recommended to load the arrays with
            ``add_time = False`` when planning to write the results in a
            time series (e.g., `vort-000.bin`, `vort-001.bin`, `vort-002.bin`, ...)


        """
        if isinstance(data, xr.Dataset):
            os.makedirs(self.data_path, exist_ok=True)
            self._write_dataset(data)
        elif isinstance(data, xr.DataArray):
            os.makedirs(self.data_path, exist_ok=True)
            self._write_array(data, file_prefix)
        else:
            msg = "Invalid type for data, try with: xarray.Dataset or xarray.DataArray"
            raise OSError(msg)

    def _write_dataset(self, dataset) -> None:
        for array_name, array in dataset.items():
            if "file_name" in array.attrs:
                self._write_array(array)
            else:
                warnings.warn(f"Can't write array {array_name}, no filename provided", stacklevel=1)

    def _write_array(self, data_array, filename: str | None = None) -> None:
        if filename is None:  # Try to get from attributes
            filename = data_array.attrs.get("file_name", None)
        if filename is None:
            msg = "Can't write field without a filename"
            raise OSError(msg)
        # If n is a dimension (for scalar), call write recursively to save
        # phi1, phi2, phi3, for instance.
        if "n" in data_array.dims:
            for n, n_val in enumerate(data_array.n.data):
                self._write_array(
                    data_array.isel(n=n, drop=True),
                    filename=f"{filename}{str(n_val).zfill(self.filename_properties.scalar_num_of_digits)}",
                )
        # If i is a dimension, call write recursively to save
        # ux, uy and uz, for instance
        elif "i" in data_array.dims:
            for i, i_val in enumerate(data_array.i.data):
                self._write_array(data_array.isel(i=i, drop=True), filename=f"{filename}{i_val}")
        # If t is a dimension (for time), call write recursively to save
        # ux-0000.bin, ux-0001.bin, ux-0002.bin, for instance.
        elif "t" in data_array.dims:
            dt = self._time_step
            if data_array.t.size == 1:
                loop_itr = enumerate(data_array.t.data)
            else:
                loop_itr = enumerate(tqdm(data_array.t.data, desc=filename))
            for k, time in loop_itr:
                self._write_array(
                    data_array.isel(t=k, drop=True),
                    self.filename_properties.get_filename_for_binary(
                        prefix=filename,
                        counter=int(time / dt),
                    ),
                )
        # and finally writes to the disc
        else:
            fileformat: str = self.filename_properties.file_extension
            if fileformat and not filename.endswith(fileformat):
                filename += fileformat
            align = [data_array.get_axis_num(i) for i in sorted(data_array.dims, reverse=True)]
            filename_and_path = os.path.join(self.data_path, filename)
            data_array.values.astype(param["mytype"]).transpose(align).tofile(filename_and_path)

    def write_xdmf(self, xdmf_name: str = "snapshots.xdmf", *, float_precision: int | None = None) -> None:
        """Write the xdmf file, so the results from the simulation and its postprocessing
        can be opened in an external visualization tool, like Paraview.

        Make sure to set all the parameters in this object properly.

        If :obj:`set_of_objects` is empty, the files are obtained automatically with :obj:`glob.glob`.

        Parameters
        ----------
        xdmf_name : str, optional
            Filename for the xdmf file, by default "snapshots.xdmf"
        float_precision : int, optional
            Number of digits for the float precision on the output file, by default None.
            If None, it uses 8 for single precision and 16 for double precision arrays.

        Raises
        ------
        IOError
            Raises IO error if it does not find any file for this simulation.

        Examples
        --------

        Initial setup:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="input.i3d")
        >>> prm.dataset.set(
        ...     filename_properties=dict(
        ...         separator="-", file_extension=".bin", number_of_digits=3
        ...     ),
        ...     stack_scalar=True,
        ...     stack_velocity=True,
        ... )

        It is possible to produce a new xdmf file, so all data
        can be visualized on any external tool:

        >>> prm.dataset.write_xdmf()

        .. versionchanged:: 1.2.0
            Added argument ``float_precision``.
        """
        if self.set_of_variables:
            time_numbers = list(range(len(self)))
            var_names = sorted(self.set_of_variables)
        else:
            filename_pattern = self.filename_properties.get_filename_for_binary(
                prefix="*", counter="*", data_path=self.data_path
            )
            filename_list = glob.glob(filename_pattern)

            if not filename_list:
                msg = f"No file was found corresponding to {filename_pattern}."
                raise OSError(msg)

            properties = [self.filename_properties.get_info_from_filename(f) for f in filename_list]

            time_numbers = sorted({r[0] for r in properties})
            var_names = sorted({r[1] for r in properties})

        nx = self._mesh.x.grid_size
        ny = self._mesh.y.grid_size
        nz = self._mesh.z.grid_size

        dx = self._mesh.x.delta
        dy = self._mesh.y.delta
        dz = self._mesh.z.delta

        dt = self._time_step

        if "x" in self.drop_coords:
            nx, dx = 1, 0.0
        if "y" in self.drop_coords:
            ny, dy = 1, 0.0
        if "z" in self.drop_coords:
            nz, dz = 1, 0.0

        prec = 8 if param["mytype"] == np.float64 else 4
        if float_precision is None:
            float_precision = 2 * prec

        float_format = f".{float_precision}e"

        def get_filename(var_name, num):
            return self.filename_properties.get_filename_for_binary(var_name, num, self.data_path)

        with open(xdmf_name, "w") as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write(' <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
            f.write(' <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">\n')
            f.write(" <Domain>\n")
            if self._mesh.y.istret == Istret.NO_REFINEMENT:
                f.write('     <Topology name="topo" TopologyType="3DCoRectMesh"\n')
                f.write(f'         Dimensions="{nz} {ny} {nx}">\n')
                f.write("     </Topology>\n")
                f.write('     <Geometry name="geo" Type="ORIGIN_DXDYDZ">\n')
                f.write("         <!-- Origin -->\n")
                f.write('         <DataItem Format="XML" Dimensions="3">\n')
                f.write("         0.0 0.0 0.0\n")
                f.write("         </DataItem>\n")
                f.write("         <!-- DxDyDz -->\n")
                f.write('         <DataItem Format="XML" Dimensions="3">\n')
                array = " ".join(format(i, float_format) for i in (dz, dy, dx))
                f.write(f"           {array}\n")
                f.write("         </DataItem>\n")
                f.write("     </Geometry>\n")
            else:
                f.write('     <Topology name="topo" TopologyType="3DRectMesh"\n')
                f.write(f'         Dimensions="{nz} {ny} {nx}">\n')
                f.write("     </Topology>\n")
                f.write('     <Geometry name="geo" Type="VXVYVZ">\n')
                f.write(f'         <DataItem Dimensions="{nx}" NumberType="Float" Precision="{prec}" Format="XML">\n')
                array = " ".join(format(i, float_format) for i in self._mesh.x.vector) if nx > 1 else "0.0"
                f.write(f"         {array}\n")
                f.write("          </DataItem>\n")
                f.write(f'         <DataItem Dimensions="{ny}" NumberType="Float" Precision="{prec}" Format="XML">\n')
                array = " ".join(format(j, float_format) for j in self._mesh.y.vector) if ny > 1 else "0.0"
                f.write(f"         {array}\n")
                f.write("          </DataItem>\n")
                f.write(f'         <DataItem Dimensions="{nz}" NumberType="Float" Precision="{prec}" Format="XML">\n')
                array = " ".join(format(k, float_format) for k in self._mesh.z.vector) if nz > 1 else "0.0"
                f.write(f"         {array}\n")
                f.write("         </DataItem>\n")
                f.write("     </Geometry>\n")
            f.write("\n")
            f.write('     <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n')
            f.write('         <Time TimeType="HyperSlab">\n')
            f.write('             <DataItem Format="XML" NumberType="Float" Dimensions="3">\n')
            f.write("             <!--Start, Stride, Count-->\n")
            f.write(f"             0.0 {format(dt, float_format)}\n")
            f.write("             </DataItem>\n")
            f.write("         </Time>\n")
            for suffix in tqdm(time_numbers, desc=xdmf_name):
                f.write("\n")
                f.write("\n")
                f.write(f'         <Grid Name="{suffix}" GridType="Uniform">\n')
                f.write('             <Topology Reference="/Xdmf/Domain/Topology[1]"/>\n')
                f.write('             <Geometry Reference="/Xdmf/Domain/Geometry[1]"/>\n')
                for prefix in var_names:
                    f.write(f'             <Attribute Name="{prefix}" Center="Node">\n')
                    f.write('                <DataItem Format="Binary"\n')
                    f.write(f'                 DataType="Float" Precision="{prec}" Endian="little" Seek="0"\n')
                    f.write(f'                 Dimensions="{nz} {ny} {nx}">\n')
                    f.write(f"                   {get_filename(prefix, suffix)}\n")
                    f.write("                </DataItem>\n")
                    f.write("             </Attribute>\n")
                f.write("         </Grid>\n")
            f.write("\n")
            f.write("     </Grid>\n")
            f.write(" </Domain>\n")
            f.write("</Xdmf>\n")


def prm_to_dict(filename="incompact3d.prm"):
    dict_outer = {}

    with open(filename) as f:
        for raw_line in f:
            # Remove spaces
            line = " ".join(raw_line.split())

            if line == "":  # Cycle if line is empty
                continue
            if line[0] == "#":  # Cycle if starts with a comment
                continue

            line = line.split("#")
            # Get variable's name and value
            parameter = line[1].strip()
            value = line[0].strip()

            try:
                # Converting from string according to datatype
                if value[0] == "'" and value[-1] == "'":  # String
                    value = value[1:-1]
                elif value.lower() == ".false.":  # Bool
                    value = False
                elif value.lower() == ".true.":  # Bool
                    value = True
                elif "." in value:  # Float
                    value = float(value)
                else:  # Int
                    value = int(value)
            except TypeError:
                warnings.warn(f"Can't convert {parameter} : {value}", stacklevel=1)
                continue

            if "(" in parameter and parameter[-1] == ")":  # Param is a list
                parameter = parameter.split("(")[0]
                if parameter not in dict_outer:
                    dict_outer[parameter] = []
                dict_outer[parameter].append(value)
            else:  # Not a list
                dict_outer[parameter] = value

    return dict_outer


def i3d_to_dict(filename="input.i3d", string=None):
    if string is None:
        with open(filename) as f:
            string = f.read()

    return i3d_string_to_dict(string)


def i3d_string_to_dict(string):
    dict_outer = {}

    for raw_line in io.StringIO(string):
        # Remove comments
        line = raw_line.partition("!")[0].replace(" ", "")
        # Remove spaces
        line = " ".join(line.split())

        if line == "":  # Cycle if line is empty
            continue

        # Beginning of a new group
        if line[0] == "&":
            key = line[1:]
            dict_inner = {}
            continue

        # End of the group
        if line.lower() == "/end":
            dict_outer[key] = dict_inner
            continue

        # Get variable's name and value
        parameter = line.partition("=")[0]
        value = line.partition("=")[-1]

        try:
            # Converting from string according to datatype
            if value[0] == "'" and value[-1] == "'":  # String
                value = value[1:-1]
            elif value.lower() == ".false.":  # Bool
                value = False
            elif value.lower() == ".true.":  # Bool
                value = True
            elif "." in value:  # Float
                value = float(value)
            else:  # Int
                value = int(value)
        except TypeError:
            warnings.warn(f"Can't convert {parameter} : {value}", stacklevel=1)
            continue

        if "(" in parameter and parameter[-1] == ")":  # Param is a list
            parameter = parameter.split("(")[0]
            if parameter not in dict_inner:
                dict_inner[parameter] = []
            dict_inner[parameter].append(value)
        else:  # Not a list
            dict_inner[parameter] = value

    return dict_outer
