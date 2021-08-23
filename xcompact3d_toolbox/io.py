# -*- coding: utf-8 -*-
"""
Usefull functions to read/write binary fields and parameters files that
are compatible with XCompact3d.

Notes
----

* Writing is handled by the methods at :obj:`xcompact3d_toolbox.array`;

* The parameters file ``.i3d`` is handled by the methods at :obj:`xcompact3d_toolbox.parameters.Parameters`.

* The integration with Xcompact3d is in prerelease as well (see `fschuch/Xcompact3d`_).
  `#3`_ was proposed in order to increase the synergy between Xcompact3d and this
  Python Package. Following the new folder structure makes it possible to
  automatically obtain the shape and timestamp for each file.
  If necessary, the conversion is exemplified in `convert_filenames_x3d_toolbox`_.

.. _`convert_filenames_x3d_toolbox`: https://gist.github.com/fschuch/5a05b8d6e9787d76655ecf25760e7289

.. _#3:
    https://github.com/fschuch/Xcompact3d/issues/3

.. _fschuch/Xcompact3d:
    https://github.com/fschuch/Xcompact3d

"""

from __future__ import annotations

import glob
import os
import os.path
import warnings
from typing import Type

import numpy as np
import traitlets
import xarray as xr
from tqdm.autonotebook import tqdm

from .param import param
from .mesh import Mesh3D


class FilenameProperties(traitlets.HasTraits):
    """[summary]

    Parameters
    ----------
    separator : str
        [description]
    file_extension : str
        [description]
    number_of_digits : int or None
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    IOError
        It is not possible to get information from filename if `separator`
        is an empty string and `number_of_digits` is None.
    
    Examples
    --------

    If the simulated fields are named like `ux-000.bin`, they are in the default
    configuration, there is no need to specify filename properties. But just in case,
    it would be like:

    >>> xcompact3d_toolbox.FilenameProperties(
    ...     separator = "-",
    ...     file_extension = ".bin",
    ...     number_of_digits = 3
    ... )

    If the simulated fields are named like `ux0000`:

    >>> xcompact3d_toolbox.FilenameProperties(
    ...     separator = "",
    ...     file_extension = "",
    ...     number_of_digits = 4
    ... )

    """

    separator = traitlets.Unicode(default_value="-")
    file_extension = traitlets.Unicode(default_value=".bin")
    number_of_digits = traitlets.Int(default_value=3, min=1, allow_none=True)

    def __init__(self, **kwargs):
        super(FilenameProperties).__init__()

        self.set(**kwargs)

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for name in self.trait_names():
            if name.startswith("_"):
                continue
            string += f"    {name} = {repr(getattr(self, name))},\n"
        string += ")"
        return string

    def set(self, **kwargs) -> None:
        """[summary]
        """
        for key, arg in kwargs.items():
            if key not in self.trait_names():
                warnings.warn(f"{key} is not a valid parameter and was not loaded")
            setattr(self, key, arg)

    def get_filename_for_binary(self, prefix: str, counter: int, data_path="") -> str:
        """[summary]

        Parameters
        ----------
        prefix : str
            [description]
        counter : int
            [description]

        Returns
        -------
        str
            [description]
        """
        if counter == "*":
            counter = "?" * self.number_of_digits
        filename = f"{prefix}{self.separator}{str(counter).zfill(self.number_of_digits)}{self.file_extension}"
        if data_path:
            filename = os.path.join(data_path, filename)
        return filename

    def get_num_from_filename(self, filename: str) -> int:
        num, _ = self.get_info_from_filename(filename)
        return num

    def get_name_from_filename(self, filename: str) -> int:
        _, name = self.get_info_from_filename(filename)
        return name

    def get_info_from_filename(self, filename: str) -> tuple[int, str]:
        """[summary]

        Parameters
        ----------
        filename : str
            [description]

        Returns
        -------
        tuple[int, str]
            [description]

        Raises
        ------
        IOError
            It is not possible to get information from filename if `separator`
            is an empty string and `number_of_digits` is None.
        """
        _filename = os.path.basename(filename)
        if self.file_extension:
            _filename = _filename[: -len(self.file_extension)]
        if self.separator:
            prefix, counter = _filename.split(self.separator)
        elif self.number_of_digits is not None:
            prefix = _filename[: -self.number_of_digits]
            counter = _filename[-self.number_of_digits :]
        else:
            raise IOError(
                "Impossible to get time from filename without separator or number of digits"
            )
        return int(counter), prefix


class Dataset(traitlets.HasTraits):

    data_path = traitlets.Unicode(default_value="./data/")
    snapshot_step = traitlets.Unicode(default_value="ioutput")
    snapshot_counting = traitlets.Unicode(default_value="ilast")
    set_of_variables = traitlets.Set()
    drop_coords = traitlets.Unicode(default_value="")
    stack_velocity = traitlets.Bool(default_value=False)
    stack_scalar = traitlets.Bool(default_value=False)
    filename_properties = traitlets.Instance(klass=FilenameProperties)

    _mesh = traitlets.Instance(klass=Mesh3D)
    _prm = traitlets.Instance(
        klass="xcompact3d_toolbox.parameters.Parameters", allow_none=True
    )

    def __init__(self, **kwargs):

        self.set_of_variables = set()
        self.filename_properties = FilenameProperties()
        self._mesh = Mesh3D()
        self._prm = None

        self.set(**kwargs)

    def __call__(self, *args) -> Type(xr.Dataset):
        for t in self._range(*args):
            yield self.load_snapshot(t)

    def __getitem__(self, arg) -> Type(xr.Dataset):
        if isinstance(arg, int):
            return self.load_snapshot(arg)
        elif isinstance(arg, slice):
            start, stop, step = arg.indices(len(self))
            return xr.concat(
                [self.load_snapshot(t) for t in range(start, stop, step)], "t"
            )
        elif isinstance(arg, str):
            return self.load_time_series(arg)
        raise TypeError("Dataset indices should be integers, string or slices")

    def __len__(self):
        # Test environment
        if self._prm is None:
            return 11
        return (
            getattr(self._prm, self.snapshot_counting)
            // getattr(self._prm, self.snapshot_step)
            + 1
        )

    def __iter__(self):
        for t in range(len(self)):
            yield self.load_snapshot(t)

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for name in self.trait_names():
            if name.startswith("_"):
                continue
            string += f"    {name} = {repr(getattr(self, name))},\n"
        string += ")"
        return string

    @property
    def _time_step(self):
        # test environment
        if self._prm is None:
            return 1.0
        return self._prm.dt * getattr(self._prm, self.snapshot_step)

    def set(self, **kwargs):

        for key, arg in kwargs.items():
            if key not in self.trait_names():
                warnings.warn(f"{key} is not a valid parameter and was not defined")
            setattr(self, key, arg)

    def load_array(self, filename: str, add_time=True, attrs: dict = None):

        coords = self._mesh.drop(*self.drop_coords)

        time_int, name = self.filename_properties.get_info_from_filename(filename)

        if add_time:
            coords["t"] = [param["mytype"](self._time_step * time_int)]

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
        numerical_identifier,
        list_of_variables: list = None,
        add_time: bool = True,
        stack_scalar: bool = None,
        stack_velocity: bool = None,
    ) -> Type[xr.Dataset]:

        dataset = xr.Dataset()

        if list_of_variables is not None:
            set_of_variables = set(list_of_variables)
        elif self.set_of_variables:
            set_of_variables = self.set_of_variables.copy()
        else:
            target_filename = self.filename_properties.get_filename_for_binary(
                "*", numerical_identifier
            )

            list_of_variables = glob.glob(os.path.join(self.data_path, target_filename))
            list_of_variables = map(
                self.filename_properties.get_name_from_filename, list_of_variables
            )
            set_of_variables = set(list_of_variables)

        if not set_of_variables:
            raise IOError(
                f"No file found corresponding to {self.data_path}/{target_filename}"
            )

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

        if stack_scalar:
            scalar_variables = sorted(
                list(
                    filter(
                        lambda name: len(name) == 4 and "phi" in name, set_of_variables
                    )
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
            velocity_variables = sorted(
                list(filter(lambda name: name in {"ux", "uy", "uz"}, set_of_variables))
            )
            if velocity_variables:
                dataset["u"] = (
                    stack_variables(velocity_variables, stack_velocity=False)
                    .to_array(dim="i")
                    .assign_coords(i=[var[-1] for var in velocity_variables])
                )
                set_of_variables -= set(velocity_variables)

        for var in sorted(list(set_of_variables)):
            filename = self.filename_properties.get_filename_for_binary(
                var, numerical_identifier
            )
            filename = os.path.join(self.data_path, filename)
            dataset[var] = self.load_array(filename=filename, add_time=add_time)

        return dataset

    def load_time_series(self, array_prefix: str):

        target_filename = self.filename_properties.get_filename_for_binary(
            array_prefix, "*"
        )
        filename_pattern = os.path.join(self.data_path, target_filename)
        filename_list = sorted(glob.glob(filename_pattern))

        if not filename_list:
            raise IOError(f"No file was found corresponding to {filename_pattern}.")

        return xr.concat(
            [
                self.load_array(file, add_time=True)
                for file in tqdm(filename_list, desc=filename_pattern)
            ],
            dim="t",
        )

    def write_array(self, dataArray, filename: str = None) -> None:
        if filename is None:  # Try to get from atributes
            filename = dataArray.attrs.get("file_name", None)
        if filename is None:
            warnings.warn(f"Can't write field without a filename")
            return
        # If n is a dimension (for scalar), call write recursively to save
        # phi1, phi2, phi3, for instance.
        if "n" in dataArray.dims:
            for n, n_val in enumerate(dataArray.n.data):
                self.write_array(
                    dataArray.isel(n=n, drop=True), filename=f"{filename}{n_val}"
                )
        # If i is a dimension, call write recursively to save
        # ux, uy and uz, for instance
        elif "i" in dataArray.dims:
            for i, i_val in enumerate(dataArray.i.data):
                self.write_array(
                    dataArray.isel(i=i, drop=True), filename=f"{filename}{i_val}"
                )
        # If t is a dimension (for time), call write recursively to save
        # ux-0000.bin, ux-0001.bin, ux-0002.bin, for instance.
        elif "t" in dataArray.dims:
            dt = self._time_step
            for k, time in enumerate(tqdm(dataArray.t.data, desc=filename)):
                self.write_array(
                    dataArray.isel(t=k, drop=True),
                    self.filename_properties.get_filename_for_binary(
                        prefix=filename, counter=int(time / dt),
                    ),
                )
        # and finally writes to the disc
        else:
            fileformat = self.filename_properties.file_extension
            if fileformat and not filename.endswith(fileformat):
                filename += fileformat
            align = [
                dataArray.get_axis_num(i) for i in sorted(dataArray.dims, reverse=True)
            ]
            filename_and_path = os.path.join(self.data_path, filename)
            dataArray.values.astype(param["mytype"]).transpose(align).tofile(
                filename_and_path
            )

    def write_xdmf(self, xdmf_name: str = "snapshots.xdmf",) -> None:

        if self.set_of_variables:
            filename_pattern = f'[{",".join(self.set_of_variables)}]'
        else:
            filename_pattern = "*"

        filename_pattern = self.filename_properties.get_filename_for_binary(
            filename_pattern, "*", self.data_path
        )
        filename_list = glob.glob(filename_pattern)

        if not filename_list:
            raise IOError(f"No file was found corresponding to {filename_pattern}.")

        properties = zip(
            *map(self.filename_properties.get_info_from_filename, filename_list,)
        )
        time_numbers, var_names = properties
        time_numbers = sorted(list(set(time_numbers)))
        var_names = sorted(list(set(var_names)))

        nx = self._mesh.x.grid_size
        ny = self._mesh.y.grid_size
        nz = self._mesh.z.grid_size

        dx = self._mesh.x.delta
        dy = self._mesh.y.delta
        dz = self._mesh.z.delta

        dt = self._time_step

        if "x" in self.drop_coords:
            nx, dx = 1, 0.0
        elif "y" in self.drop_coords:
            ny, dy = 1, 0.0
        elif "z" in self.drop_coords:
            nz, dz = 1, 0.0

        prec = 8 if param["mytype"] == np.float64 else 4

        def get_filename(var_name, num):
            return self.filename_properties.get_filename_for_binary(
                var_name, num, self.data_path
            )

        with open(xdmf_name, "w") as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write(' <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
            f.write(
                ' <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">\n'
            )
            f.write(" <Domain>\n")
            if self._mesh.y.istret == 0:
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
                f.write(f"           {dz}  {dy}  {dx}\n")
                f.write("         </DataItem>\n")
                f.write("     </Geometry>\n")
            else:
                f.write('     <Topology name="topo" TopologyType="3DRectMesh"\n')
                f.write(f'         Dimensions="{nz} {ny} {nx}">\n')
                f.write("     </Topology>\n")
                f.write('     <Geometry name="geo" Type="VXVYVZ">\n')
                f.write(
                    f'         <DataItem Dimensions="{nx}" NumberType="Float" Precision="{prec}" Format="XML">\n'
                )
                f.write(
                    f'         {" ".join(map(str, self._mesh.x.vector)) if nx > 1 else 0.0}\n'
                )
                f.write("          </DataItem>\n")
                f.write(
                    f'         <DataItem Dimensions="{ny}" NumberType="Float" Precision="{prec}" Format="XML">\n'
                )
                f.write(
                    f'         {" ".join(map(str, self._mesh.y.vector)) if ny > 1 else 0.0}'
                )
                f.write("          </DataItem>\n")
                f.write(
                    f'         <DataItem Dimensions="{nz}" NumberType="Float" Precision="{prec}" Format="XML">\n'
                )
                f.write(
                    f'         {" ".join(map(str, self._mesh.z.vector)) if nz > 1 else 0.0}'
                )
                f.write("         </DataItem>\n")
                f.write("     </Geometry>\n")
            f.write("\n")
            f.write(
                '     <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n'
            )
            f.write('         <Time TimeType="HyperSlab">\n')
            f.write(
                '             <DataItem Format="XML" NumberType="Float" Dimensions="3">\n'
            )
            f.write("             <!--Start, Stride, Count-->\n")
            f.write(f"             0.0 {dt}\n")
            f.write("             </DataItem>\n")
            f.write("         </Time>\n")
            for suffix in tqdm(time_numbers, desc=xdmf_name):
                f.write("\n")
                f.write("\n")
                f.write(f'         <Grid Name="{suffix}" GridType="Uniform">\n')
                f.write(
                    '             <Topology Reference="/Xdmf/Domain/Topology[1]"/>\n'
                )
                f.write(
                    '             <Geometry Reference="/Xdmf/Domain/Geometry[1]"/>\n'
                )
                for prefix in var_names:
                    f.write(f'             <Attribute Name="{prefix}" Center="Node">\n')
                    f.write('                <DataItem Format="Binary"\n')
                    f.write(
                        f'                 DataType="Float" Precision="{prec}" Endian="little" Seek="0" \n'
                    )
                    f.write(f'                 Dimensions="{nz} {ny} {nx}">\n')
                    f.write(f"                   {get_filename(prefix, suffix)}\n")
                    f.write("                </DataItem>\n")
                    f.write("             </Attribute>\n")
                f.write("         </Grid>\n")
            f.write("\n")
            f.write("     </Grid>\n")
            f.write(" </Domain>\n")
            f.write("</Xdmf>")


def prm_to_dict(filename="incompact3d.prm"):

    f = open(filename)

    dict_outer = {}

    for line in f:
        # Remove spaces
        line = " ".join(line.split())

        if line == "":  # Cycle if line is empty
            continue
        if line[0] == "#":  # Cycle if starts with a comment
            continue

        line = line.split("#")
        # Get variable's name and value
        param = line[1].strip()
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
        except:
            warnings.warn(f"Can't convert {param} : {value}")
            continue

        if "(" in param and ")" == param[-1]:  # Param is a list
            param = param.split("(")[0]
            if param not in dict_outer:
                dict_outer[param] = []
            dict_outer[param].append(value)
        else:  # Not a list
            dict_outer[param] = value

    f.close()

    return dict_outer


def i3d_to_dict(filename="input.i3d"):

    f = open(filename)

    dict_outer = {}

    for line in f:
        # Remove comments
        line = line.partition("!")[0].replace(" ", "")
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
        param = line.partition("=")[0]
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
        except:
            warnings.warn(f"Can't convert {param} : {value}")
            continue

        if "(" in param and ")" == param[-1]:  # Param is a list
            param = param.split("(")[0]
            if param not in dict_inner:
                dict_inner[param] = []
            dict_inner[param].append(value)
        else:  # Not a list
            dict_inner[param] = value

    f.close()

    return dict_outer
