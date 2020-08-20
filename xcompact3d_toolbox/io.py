# -*- coding: utf-8 -*-
"""
Usefull functions to read binary fields from the disc.

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

from .param import mytype, boundary_condition
import numpy as np
import xarray as xr
import os.path
import glob


def readfield(filename, prm, dims="auto", coords=None, name=None, attrs=None):
    """This functions reads a binary field from Xcompact3d with :obj:`numpy.fromfile`
    and wraps it into a :obj:`xarray.DataArray` with the appropriate dimensions,
    coordinates and attributes.

    The properties are automatically inferted if the
    file is inside Xcompact3d's output folders structure, i.e.:

    * 3d_snapshots (nx, ny, nz);
    * xy_planes (nx, ny);
    * xz_planes (nx, nz);
    * yz_planes (ny, nz).

    Attributes include the proper boundary conditions for derivatives if the
    file prefix is ``ux``, ``uy``, ``uz``, ``phi`` or ``pp``. Data type is
    defined by :obj:`xcompact3d_toolbox.mytype`.

    Parameters
    ----------
    filename : str
        Name of the file to be read.
    prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
        Contains the computational and physical parameters.
    dims : 'auto' or hashable or sequence of hashable
        Name(s) of the data dimension(s). Must be either a hashable
        (only for 1D data) or a sequence of hashables with length equal to the
        number of dimensions (see :obj:`xarray.DataArray`). If dims='auto' (default),
        dimensions are inferted from the folder structure.
    coords : sequence or dict of array_like objects, optional
        Coordinates (tick labels) to use for indexing along each dimension (see
        :obj:`xarray.DataArray`). If dims='auto' (default), coordinates are inferred
        from the folder structure.
    name : str or None, optional
        Name of this array. If dims='auto' (default), name is inferred
        from filename.
    attrs : dict_like or None, optional
        Attributes to assign to the new instance. If dims='auto' (default),
        boundary conditions for derivatives are included.

    Returns
    -------
    :obj:`xarray.DataArray`
        Data array containing values read from the disc.

    Examples
    -------

    >>> prm = x3d.Parameters()

    >>> xcompact3d_toolbox.mytype = np.float64 # if x3d was compiled with `-DDOUBLE_PREC`
    >>> xcompact3d_toolbox.mytype = np.float32 # otherwise

    In the following cases, coordinates and dimensions are infered from the
    folder containing the file:

    >>> ux = x3d.readfield('./data/3d_snapshots/ux-00000400.bin', prm)
    >>> uy = x3d.readfield('./data/xy_planes/uy-00000400.bin', prm)
    >>> uz = x3d.readfield('./data/xz_planes/uz-00000400.bin', prm)

    It is possible to handle the filenames from previous X3d's versions by
    setting coordinates manually:

    >>> ux = x3d.readfield(
    ...     './data/ux0010',
    ...     prm,
    ...     dims = ["x", "y", "z"],
    ...     coords = prm.get_mesh()
    ... )
    """

    if dims.lower() == "auto":

        path, file = os.path.split(filename)
        path = os.path.basename(path)

        name = os.path.basename(file).split("-")[0]
        if "phi" in name:
            name = "phi"

        mesh = prm.get_mesh()

        if path == "3d_snapshots":
            pass
        elif path == "xy_planes":
            del mesh["z"]
        elif path == "xz_planes":
            del mesh["y"]
        elif path == "yz_planes":
            del mesh["x"]

        shape = []
        for key, value in mesh.items():
            shape.append(value.size)

        dims = mesh.keys()
        coords = mesh

        # Setting BC
        if attrs == None:
            attrs = {}
        attrs["BC"] = boundary_condition(prm, name)

    else:
        shape = []
        for key, value in coords.items():
            shape.append(value.size)

    return xr.DataArray(
        np.fromfile(filename, dtype=mytype).reshape(shape, order="F"),
        dims=dims,
        coords=coords,
        name=name,
        attrs=attrs,
    )


def read_all(filename_pattern, prm):
    """Reads all files matching the ``filename_pattern`` with
    :obj:`xcompact3d_toolbox.io.readfield` and concatenates them into a time series.

    .. note:: Make sure to have enough memory to load all files at same time.

    .. note:: This is only compatible with the new filename structure,
        the conversion is exemplified in `convert_filenames_x3d_toolbox`_.

    .. _`convert_filenames_x3d_toolbox`: https://gist.github.com/fschuch/5a05b8d6e9787d76655ecf25760e7289

    Parameters
    ----------
    filename_pattern : str
        A specified pattern according to the rules used by the Unix shell.
    prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
        Contains the computational and physical parameters.

    Returns
    -------
    :obj:`xarray.DataArray`
        Data array containing values read from the disc.

    Examples
    -------

    >>> prm = x3d.Parameters()

    >>> x3d.mytype = np.float64 # if x3d was compiled with `-DDOUBLE_PREC`
    >>> x3d.mytype = np.float32 # otherwise

    In the following cases, coordinates and dimensions are infered from the
    folder containing the file and time from the filenames:

    >>> ux = x3d.read_all('./data/3d_snapshots/ux-*.bin', prm)
    >>> uy = x3d.read_all('./data/xy_planes/uy-*.bin', prm)
    >>> uz = x3d.read_all('./data/xz_planes/uz-0000??00.bin', prm)

    """

    filenames = sorted(glob.glob(filename_pattern))

    dt = prm.dt
    t = dt * np.array(
        [
            mytype(os.path.basename(file).split("-")[-1].split(".")[0])
            for file in filenames
        ],
        dtype=mytype,
    )

    # numscalar = prm.dict['BasicParam'].get('numscalar', 0)

    # if numscalar > 1:
    #    mesh['n'] = [int(os.path.basename(file).split('-')[0][-1])]

    from tqdm.notebook import tqdm as tqdm

    return xr.concat(
        [readfield(file, prm) for file in tqdm(filenames, desc=filename_pattern)],
        dim="t",
    ).assign_coords(coords={"t": t})


def write_xdmf(prm):
    """Writes four xdmf files:

    * ``./data/3d_snapshots.xdmf`` for 3D snapshots in ``./data/3d_snapshots/*``;
    * ``./data/xy_planes.xdmf`` for planes in ``./data/xy_planes/*``;
    * ``./data/xz_planes.xdmf`` for planes in ``./data/xz_planes/*``;
    * ``./data/yz_planes.xdmf`` for planes in ``./data/yz_planes/*``.

    Shape and time are inferted from folder structure and filenames.
    File list is obtained automatically with :obj:`glob`.

    .. note:: This is only compatible with the new filename structure,
        the conversion is exemplified in `convert_filenames_x3d_toolbox`_.

    .. _`convert_filenames_x3d_toolbox`: https://gist.github.com/fschuch/5a05b8d6e9787d76655ecf25760e7289

    Parameters
    ----------
    prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
        Contains the computational and physical parameters.

    Examples
    -------

    >>> prm = x3d.Parameters()
    >>> x3d.write_xdmf(prm)

    """
    from tqdm.notebook import tqdm as tqdm

    for folder in ["3d_snapshots", "xy_planes", "xz_planes", "yz_planes"]:

        xdmf = os.path.join("data", f"{folder}.xdmf")
        filepath = os.path.join("data", folder, "*")

        filenames = glob.glob(filepath)
        if len(filenames) == 0:
            continue

        prefixes = sorted(
            set([os.path.basename(file).split("-")[0] for file in filenames])
        )
        suffixes = sorted(
            set(
                [
                    os.path.basename(file).split("-")[-1].split(".")[0]
                    for file in filenames
                ]
            )
        )

        mesh = prm.mesh

        nx, ny, nz = [mesh[d].size for d in mesh.keys()]
        dx, dy, dz = [mesh[d][1] - mesh[d][0] for d in mesh.keys()]

        dt = prm.dt * (float(suffixes[1]) - float(suffixes[0]))

        if folder == "xy_planes":
            nz, dz = 0, 0
        elif folder == "xz_planes":
            ny, dy = 0, 0
        elif folder == "yz_planes":
            nx, dx = 0, 0

        prec = 8 if mytype == np.float64 else 4

        ibm_flag = "ibm" in prefixes

        with open(xdmf, "w") as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write(' <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
            f.write(
                ' <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">\n'
            )
            f.write(" <Domain>\n")
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
            for suffix in tqdm(suffixes, desc=xdmf):
                f.write("\n")
                f.write("\n")
                f.write(f'         <Grid Name="{suffix}" GridType="Uniform">\n')
                f.write(
                    '             <Topology Reference="/Xdmf/Domain/Topology[1]"/>\n'
                )
                f.write(
                    '             <Geometry Reference="/Xdmf/Domain/Geometry[1]"/>\n'
                )
                for prefix in prefixes:
                    f.write(f'             <Attribute Name="{prefix}" Center="Node">\n')
                    f.write('                <DataItem Format="Binary"\n')
                    f.write(
                        f'                 DataType="Float" Precision="{prec}" Endian="little" Seek="0" \n'
                    )
                    f.write(f'                 Dimensions="{nz} {ny} {nx}">\n')
                    if ibm_flag and prefix == "ibm":
                        f.write(
                            f"                   ./{folder}/{prefix}-{suffixes[0]}.bin\n"
                        )
                    else:
                        f.write(
                            f"                   ./{folder}/{prefix}-{suffix}.bin\n"
                        )
                    f.write("                </DataItem>\n")
                    f.write("             </Attribute>\n")
                f.write("         </Grid>\n")
            f.write("\n")
            f.write("     </Grid>\n")
            f.write(" </Domain>\n")
            f.write("</Xdmf>")


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
            print(f"Can't convert {param} : {value}")
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
