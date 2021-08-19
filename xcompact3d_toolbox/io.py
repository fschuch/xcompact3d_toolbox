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
    numeration_steep = traitlets.Unicode(default_value="ioutput")

    def __init__(self, **kwargs):
        super(FilenameProperties).__init__()

        self.set(**kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f'    separator = "{self.separator}",\n'
            f'    file_extension = "{self.file_extension}",\n'
            f"    number_of_digits = {self.number_of_digits},\n"
            f'    numeration_steep = "{self.numeration_steep}",\n'
            ")"
        )

    def set(self, **kwargs) -> None:
        """[summary]
        """
        for key, arg in kwargs.items():
            if key not in self.trait_names():
                warnings.warn(f"{key} is not a valid parameter and was not loaded")
            setattr(self, key, arg)

    def get_filename_for_binary(self, prefix: str, counter: int) -> str:
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
        return f"{prefix}{self.separator}{str(counter).zfill(self.number_of_digits)}{self.file_extension}"

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
        if self.file_extension:
            filename = filename[: -len(self.file_extension)]
        if self.separator:
            prefix, counter = filename.split(self.separator)
        elif self.number_of_digits is not None:
            prefix = filename[: -self.number_of_digits]
            counter = filename[-self.number_of_digits :]
        else:
            raise IOError(
                "Impossible to get time from filename without separator or number of digits"
            )
        return int(counter), prefix


def read_field(
    prm,
    filename: str,
    drop_coords: str = "",
    name: str = "",
    add_time: bool = False,
    attrs: dict = {},
) -> Type[xr.DataArray]:

    file = os.path.basename(filename)
    coords = prm.mesh.drop(*drop_coords)

    # if name is empty, we obtain it from the filename
    # time coordinate is added as well
    if not name:
        time_int, name = prm.filename_properties.get_info_from_filename(file)
        if add_time:
            step = getattr(prm, prm.filename_properties.numeration_steep)
            coords["t"] = [param["mytype"](time_int * step * prm.dt)]

    # Include atributes for boundary conditions, useful to compute the derivatives
    if "phi" in name:
        attrs["BC"] = prm.get_boundary_condition("phi")
    else:
        attrs["BC"] = prm.get_boundary_condition(name)

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


def write_field(dataArray, prm, filename: str = None) -> None:
    if filename is None:  # Try to get from atributes
        filename = dataArray.attrs.get("file_name", None)
    if filename is None:
        warnings.warn(f"Can't write field without a filename")
        return
    # If n is a dimension (for scalar), call write recursively to save
    # phi1, phi2, phi3, for instance.
    if "n" in dataArray.n:
        for n in range(dataArray.n.size):
            write_field(dataArray.sel(n=n), prm, filename=f"{filename}{n.data}")
    # If i is a dimension, call write recursively to save
    # ux, uy and uz, for instance
    if "i" in dataArray.dims:
        for i in dataArray.i:
            write_field(dataArray.sel(i=i), prm, filename=f"{filename}{i}")
    # If t is a dimension (for time), call write recursively to save
    # ux-0000.bin, ux-0001.bin, ux-0002.bin, for instance.
    elif "t" in dataArray.dims:
        step = getattr(prm, prm.filename_properties.numeration_steep)
        dt = prm.dt * step
        for k, time in enumerate(tqdm(dataArray.t.values, desc=filename)):
            write_field(
                dataArray.isel(t=k),
                prm,
                prm.filename_properties.get_filename_for_binary(
                    prefix=filename, counter=int(time / dt),
                ),
            )
    # and finally writes to the disc
    else:
        fileformat = prm.filename_properties.file_extension
        if fileformat and not filename.endswith(fileformat):
            filename += fileformat
        align = [
            dataArray.get_axis_num(i) for i in sorted(dataArray.dims, reverse=True)
        ]

        dataArray.values.astype(param["mytype"]).transpose(align).tofile(filename)


def read_time_series(
    prm, filename_pattern: str = None, filename_list: list = None, **kwargs,
) -> Type[xr.DataArray]:
    if filename_list is None:
        filename_list = sorted(glob.glob(filename_pattern))

    if not filename_list:
        raise IOError(f"No file was found corresponding to {filename_pattern}.")

    # set or sobescribe, time is needed for the concatenation
    kwargs["add_time"] = True

    return xr.concat(
        [
            read_field(prm, file, **kwargs)
            for file in tqdm(filename_list, desc=filename_pattern)
        ],
        dim="t",
    )


def read_snapshot(
    prm,
    numerical_identifier: int,
    list_of_variables: list = None,
    data_path: str = "",
    drop_coords: str = "",
    add_time: bool = False,
    stack_velocity: bool = True,
    stack_scalar: bool = True,
) -> Type[xr.Dataset]:

    dataset = xr.Dataset()

    if list_of_variables is None:
        target_filename = prm.filename_properties.get_filename_for_binary(
            "*", numerical_identifier
        )

        list_of_variables = glob.glob(os.path.join(data_path, target_filename))
        list_of_variables = map(
            prm.filename_properties.get_name_from_filename, list_of_variables
        )

    if list_of_variables is None:
        raise IOError(f"No file found corresponding to .{data_path}/{target_filename}")

    list_of_variables = set(list_of_variables)

    if stack_scalar:
        scalar_variables = sorted(
            list(
                filter(lambda name: len(name) == 4 and "phi" in name, list_of_variables)
            )
        )
        if scalar_variables:
            dataset["phi"] = (
                read_snapshot(
                    prm,
                    numerical_identifier,
                    scalar_variables,
                    data_path,
                    drop_coords,
                    add_time,
                    stack_velocity=False,
                    stack_scalar=False,
                )
                .to_array(dim="n")
                .assign_coords(n=[int(var[-1]) for var in scalar_variables])
            )
            list_of_variables -= set(scalar_variables)

    if stack_velocity:
        velocity_variables = sorted(
            list(filter(lambda name: name in {"ux", "uy", "uz"}, list_of_variables))
        )
        if velocity_variables:
            dataset["u"] = (
                read_snapshot(
                    prm,
                    numerical_identifier,
                    velocity_variables,
                    data_path,
                    drop_coords,
                    add_time,
                    stack_velocity=False,
                    stack_scalar=False,
                )
                .to_array(dim="i")
                .assign_coords(i=[var[-1] for var in velocity_variables])
            )
            list_of_variables -= set(velocity_variables)

    for var in list_of_variables:
        filename = prm.filename_properties.get_filename_for_binary(
            var, numerical_identifier
        )
        dataset[var] = read_field(
            prm=prm, filename=filename, drop_coords=drop_coords, add_time=add_time
        )

    return dataset


def write_xdmf(
    prm,
    xdmf_name: str = "snapshots.xdmf",
    filename_pattern: str = "./data/*",
    drop_coords: str = "",
) -> None:

    filename_list = glob.glob(filename_pattern)

    if not filename_list:
        raise IOError(f"No file was found corresponding to {filename_pattern}.")

    data_path = os.path.dirname(filename_pattern)

    properties = zip(
        *map(
            lambda file: prm.filename_properties.get_info_from_filename(
                os.path.basename(file)
            ),
            filename_list,
        )
    )
    time_numbers, var_names = properties
    time_numbers = sorted(list(set(time_numbers)))
    var_names = sorted(list(set(var_names)))

    nx = prm.nx
    ny = prm.ny
    nz = prm.nz

    dx = prm.dx
    dy = prm.dy
    dz = prm.dz

    step = getattr(prm, prm.filename_properties.numeration_steep)
    dt = prm.dt * step

    if drop_coords == "x":
        nx, dx = 1, 0.0
    elif drop_coords == "y":
        ny, dy = 1, 0.0
    elif drop_coords == "z":
        nz, dz = 1, 0.0

    prec = 8 if param["mytype"] == np.float64 else 4

    def get_filename(var_name, num):
        string = ""
        if data_path:
            string += data_path + "/"
        string += prm.filename_properties.get_filename_for_binary(var_name, num)
        return string

    with open(xdmf_name, "w") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write(' <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        f.write(' <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">\n')
        f.write(" <Domain>\n")
        if prm.istret == 0:
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
                f'         {" ".join(map(str, prm.mesh.x.vector)) if nx > 1 else 0.0}\n'
            )
            f.write("          </DataItem>\n")
            f.write(
                f'         <DataItem Dimensions="{ny}" NumberType="Float" Precision="{prec}" Format="XML">\n'
            )
            f.write(
                f'         {" ".join(map(str, prm.mesh.y.vector)) if ny > 1 else 0.0}'
            )
            f.write("          </DataItem>\n")
            f.write(
                f'         <DataItem Dimensions="{nz}" NumberType="Float" Precision="{prec}" Format="XML">\n'
            )
            f.write(
                f'         {" ".join(map(str, prm.mesh.z.vector)) if nz > 1 else 0.0}'
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
            f.write('             <Topology Reference="/Xdmf/Domain/Topology[1]"/>\n')
            f.write('             <Geometry Reference="/Xdmf/Domain/Geometry[1]"/>\n')
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


def dict_to_i3d(dict, filename="input.i3d"):
    try:
        with open(filename, "w") as file:

            file.write("! -*- mode: f90 -*-\n")

            for blockkey, block in dict.items():
                if blockkey == "auxiliar":
                    continue

                file.write("\n")
                file.write("!===================\n")
                file.write("&" + blockkey + "\n")
                file.write("!===================\n")
                file.write("\n")

                for paramkey, param in block.items():
                    # Check if param is a list or not
                    if isinstance(param, list):
                        for n, p in enumerate(param):
                            file.write(f"{paramkey}({n+1}) = {p}\n")
                    elif isinstance(param, str):
                        file.write(f"{paramkey} = '{param}'\n")
                    else:
                        file.write(f"{paramkey} = {param}\n")
                file.write("\n")
                file.write("/End\n")

    except IOError as e:
        print("Couldn't open or write to file (%s)." % e)
