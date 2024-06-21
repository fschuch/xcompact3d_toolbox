"""The new **Sandbox Flow Configuration** (``itype = 12``) aims to break many of the
barriers to entry in a Navier-Stokes solver.
The idea is to easily provide everything that XCompact3d needs from a Python Jupyter
Notebook, like initial conditions, solid geometry, boundary conditions, and the
parameters. For students in computational fluid dynamics, it provides a
direct hands-on experience and a safe place for practicing and learning, while
for advanced users and code developers, it works as a rapid prototyping tool.
For more details, see:

   * `"A Jupyter sandbox environment coupled into the high-order Navier-Stokes\
   solver Xcompact3d", by F.N. Schuch, F.D. Vianna, A. Mombach, J.H. Silvestrini.\
   JupyterCon 2020. <https://www.fschuch.com/en/publication/2020-jupytercon/>`_

   * `"Sandbox flow configuration: A rapid prototyping tool inside XCompact3d",\
   by F.N. Schuch. XCompact3d 2021 Online Showcase Event.\
   <https://www.fschuch.com/en/talk/sandbox-flow-configuration-a-rapid-prototyping-tool-inside-xcompact3d/>`_
"""

from __future__ import annotations

import os.path
from typing import TYPE_CHECKING

import numba
import numpy as np
import stl
import xarray as xr

from xcompact3d_toolbox.param import param

if TYPE_CHECKING:
    from xcompact3d_toolbox.parameters import Parameters


class DimensionNotFoundError(KeyError):
    """Raised when a dimension is not found in the DataArray."""

    def __init__(self, dim):
        self.dim = dim
        super().__init__(f'Invalid key for "kwargs", "{dim}" is not a valid dimension')


def init_epsi(prm: Parameters, *, dask: bool = False) -> dict[str, xr.DataArray]:
    """Initializes the :math:`\\epsilon` arrays that define the solid geometry
    for the Immersed Boundary Method.

    Parameters
    ----------
    prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
        Contains the computational and physical parameters.

    dask : bool
        Defines the lazy parallel execution with dask arrays.
        See :obj:`xcompact3d_toolbox.array.x3d.pencil_decomp()`.

    Returns
    -------
    :obj:`dict` of :obj:`xarray.DataArray`
        A dictionary containing the epsi(s) array(s):

        * epsi (nx, ny, nz) if :obj:`iibm` != 0;
        * xepsi (nxraf, ny, nz) if :obj:`iibm` = 2;
        * yepsi (nx, nyraf, nz) if :obj:`iibm` = 2;
        * zepsi (nx, ny, nzraf) if :obj:`iibm` = 2.

        Each one initialized with np.zeros(dtype=np.bool) and wrapped into a
        :obj:`xarray.DataArray` with the proper size, dimensions and coordinates.
        They are used to define the object(s) that is(are) going to be inserted
        into the cartesian domain by the Immersed Boundary Method (IBM).
        They should be set to one (True) at the solid points and stay
        zero (False) at the fluid points, some standard geometries are provided
        by the accessor :obj:`xcompact3d_toolbox.sandbox.Geometry`.

    Examples
    -------

    >>> prm = xcompact3d_toolbox.Parameters()
    >>> epsi = xcompact3d_toolbox.init_epsi(prm)

    .. versionchanged:: 1.2.0
        The argument ``dask`` changed to keyword-only.
    """

    epsi: dict[str, xr.DataArray] = {}

    if prm.iibm == 0:
        return epsi

    from os import makedirs

    makedirs(os.path.join(prm.dataset.data_path, "geometry"), exist_ok=True)

    mesh = prm.get_mesh()

    # the epsi array in the standard mesh (nx, ny, nz)
    fields = {"epsi": (mesh["x"], mesh["y"], mesh["z"])}

    if prm.iibm == 2:  # noqa: PLR2004
        # Getting refined mesh
        mesh_raf = prm.get_mesh(refined_for_ibm=True)
        # Three additional versions are needed if iibm = 2,
        # each one refined in one dimension by a factor nraf
        fields["xepsi"] = (mesh_raf["x"], mesh["y"], mesh["z"])
        fields["yepsi"] = (mesh["x"], mesh_raf["y"], mesh["z"])
        fields["zepsi"] = (mesh["x"], mesh["y"], mesh_raf["z"])

    # Data type defined to boolean for simplicity, since epsi should be zero at
    # the fluid points and one at the solid points. The algorithm should work
    # for integer ao float as well
    for key, (x, y, z) in fields.items():
        epsi[key] = xr.DataArray(
            np.zeros((x.size, y.size, z.size), dtype=bool),
            dims=["x", "y", "z"],
            coords={"x": x, "y": y, "z": z},
            name=key,
        )

    # With 'file_name' attribute, we make sure that epsi will be written to disc,
    # while the refined versions are not needed
    epsi["epsi"].attrs = {"file_name": os.path.join("geometry", "epsilon")}

    # Turns on lazy parallel execution with dask arrays
    if dask is True:
        for key in epsi:
            if key == "epsi":
                # Decomposition in any direction would work for epsi
                epsi[key] = epsi[key].x3d.pencil_decomp("x")
            else:
                # notice that key[0] will be x, y and z for
                # xepsi, yepsi and zepsi, respectively
                epsi[key] = epsi[key].x3d.pencil_decomp(key[0])

    return epsi


def init_dataset(prm: Parameters) -> xr.Dataset:
    """This function initializes a :obj:`xarray.Dataset` including all variables
    that should be provided to XCompact3d and the sandbox flow configuration,
    according to the computational and physical parameters.

    Parameters
    ----------
    prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
        Contains the computational and physical parameters.

    Returns
    -------
    :obj:`xarray.Dataset`
        Each variable is initialized with
        ``np.zeros(dtype=xcompact3d_toolbox.param["mytype"])`` and wrapped into a
        :obj:`xarray.Dataset` with the proper size, dimensions, coordinates and
        attributes, check them for more details. The variables are:

        * ``bxx1``, ``bxy1``, ``bxz1`` - Inflow boundary condition for ux, uy
          and uz, respectively (if nclx1 = 2);
        * ``noise_mod_x1`` - for random noise modulation at inflow boundary
          condition (if nclx1 = 2);
        * ``bxphi1`` - Inflow boundary condition for scalar field(s)
          (if nclx1 = 2 and numscalar > 0);
        * ``byphi1`` - Bottom boundary condition for scalar field(s)
          (if nclyS1 = 2, numscalar > 0 and uset = 0);
        * ``byphin`` - Top boundary condition for scalar field(s)
          (if nclySn = 2, numscalar > 0 and uset = 0);
        * ``ux``, ``uy``, ``uz`` - Initial condition for velocity field;
        * ``phi`` - Initial condition for scalar field(s) (if numscalar > 0);
        * ``vol_frc`` - Integral operator employed for flow rate control in case
          of periodicity in x direction (nclx1 = 0 and nclxn = 0).
          Xcompact3d will compute the volumetric integration as
          I = sum(vol_frc * ux) and them will correct streamwise velocity
          as ux = ux / I, so, set ``vol_frc`` properly.

    Examples
    --------

    >>> prm = xcompact3d_toolbox.Parameters()
    >>> dataset = xcompact3d_toolbox.init_dataset(prm)
    >>> #
    >>> # Code here your customized flow configuration
    >>> #
    >>> prm.dataset.write(dataset)  # write the files to the disc

    """

    from os import makedirs

    makedirs(prm.dataset.data_path, exist_ok=True)

    # Init dataset
    ds = xr.Dataset(coords=prm.get_mesh()).assign_coords(n=[n + 1 for n in range(prm.numscalar)])

    ds.x.attrs = {"name": "Streamwise coordinate", "long_name": r"$x_1$"}
    ds.y.attrs = {"name": "Vertical coordinate", "long_name": r"$x_2$"}
    ds.z.attrs = {"name": "Spanwise coordinate", "long_name": r"$x_3$"}
    ds.n.attrs = {"name": "Scalar fraction", "long_name": r"$\ell$"}

    description = {0: "Streamwise", 1: "Vertical", 2: "Spanwise"}

    # Boundary conditions
    if prm.nclx1 == 2:  # noqa: PLR2004
        for i, var in enumerate("bxx1 bxy1 bxz1 noise_mod_x1".split()):
            ds[var] = xr.DataArray(
                param["mytype"](0.0),
                dims=["y", "z"],
                coords=[ds.y, ds.z],
                attrs={
                    "file_name": var,
                    "name": f"Inflow Plane for {description.get(i,'')} Velocity",
                    "long_name": rf"$u_{i+1} (x_1=0,x_2,x_3)$",
                },
            )
        ds.noise_mod_x1.attrs["name"] = "Modulation function for Random Numbers at Inflow Plane"
        ds.noise_mod_x1.attrs["long_name"] = r"mod $ (x_1=0,x_2,x_3)$"

    if prm.numscalar != 0:
        if prm.nclxS1 == 2:  # noqa: PLR2004
            ds["bxphi1"] = xr.DataArray(
                param["mytype"](0.0),
                dims=["n", "y", "z"],
                coords=[ds.n, ds.y, ds.z],
                attrs={
                    "file_name": "bxphi1",
                    "name": "Inflow Plane for Scalar field(s)",
                    "long_name": r"$\varphi (x_1=0,x_2,x_3,n)$",
                },
            )
        if prm.nclyS1 == 2:  # noqa: PLR2004
            ds["byphi1"] = xr.DataArray(
                param["mytype"](0.0),
                dims=["n", "x", "z"],
                coords=[ds.n, ds.x, ds.z],
                attrs={
                    "file_name": "byphi1",
                    "name": "Bottom Boundary Condition for Scalar field(s)",
                    "long_name": r"$\varphi (x_1,x_2=0,x_3,n)$",
                },
            )
        if prm.nclySn == 2:  # noqa: PLR2004
            ds["byphin"] = xr.DataArray(
                param["mytype"](0.0),
                dims=["n", "x", "z"],
                coords=[ds.n, ds.x, ds.z],
                attrs={
                    "file_name": "byphin",
                    "name": "Top Boundary Condition for Scalar field(s)",
                    "long_name": r"$\varphi (x_1,x_2=L_2,x_3,n)$",
                },
            )
    # Initial Condition
    for i, var in enumerate(["ux", "uy", "uz"]):
        ds[var] = xr.DataArray(
            param["mytype"](0.0),
            dims=["x", "y", "z"],
            coords=[ds.x, ds.y, ds.z],
            attrs={
                "file_name": var,
                "name": f"Initial Condition for {description.get(i,'')} Velocity",
                "long_name": rf"$u_{i+1!s} (x_1,x_2,x_3,t=0)$",
                "BC": prm.get_boundary_condition(var),
            },
        )
    if prm.numscalar != 0:
        ds["phi"] = xr.DataArray(
            param["mytype"](0.0),
            dims=["n", "x", "y", "z"],
            coords=[ds.n, ds.x, ds.y, ds.z],
            attrs={
                "file_name": "phi",
                "name": "Initial Condition for Scalar field(s)",
                "long_name": r"$\varphi (x_1,x_2,x_3,n,t=0)$",
                "BC": prm.get_boundary_condition("phi"),
            },
        )
    # Flowrate control
    if prm.nclx1 == 0 and prm.nclxn == 0:
        ds["vol_frc"] = xr.DataArray(
            param["mytype"](0.0),
            dims=["x", "y", "z"],
            coords=[ds.x, ds.y, ds.z],
            attrs={
                "file_name": "vol_frc",
                "name": "Integral Operator for Flow Rate Control",
            },
        )

    return ds


@xr.register_dataarray_accessor("geo")
class Geometry:
    """An accessor with some standard geometries for :obj:`xarray.DataArray`.
    Use them in combination with the arrays initialized at
    :obj:`xcompact3d_toolbox.sandbox.init_epsi` and the new
    :obj:`xcompact3d_toolbox.genepsi.gene_epsi_3d`.
    """

    def __init__(self, data_array: xr.DataArray):
        self._data_array = data_array

    def from_stl(
        self,
        *,
        filename: str | None = None,
        stl_mesh: stl.mesh.Mesh | None = None,
        origin: dict | None = None,
        rotate: dict | None = None,
        scale: float | None = None,
        user_tol: float = 2.0 * np.pi,
        remp: bool = True,
    ) -> xr.DataArray:
        r"""Load a STL file and compute if the nodes of the computational
        mesh are inside or outside the object. In this way, the
        customized geometry can be used at the flow solver.

        The methodology is based on the work of:

        * Jacobson, A., Kavan, L., & Sorkine-Hornung,
          O. (2013). Robust inside-outside segmentation
          using generalized winding numbers. ACM
          Transactions on Graphics (TOG), 32(4), 1-12.

        The Python implementation is an adaptation from
        `inside-3d-mesh <https://github.com/marmakoide/inside-3d-mesh>`_
        (licensed under the MIT License),
        by `@marmakoide <https://github.com/marmakoide>`_.

        To maximize the performance here at the toolbox, :obj:`from_stl` is powered by
        `Numba`_, that translates Python functions to optimized machine code at runtime.
        This method is compatible with `Dask`_ for parallel computation.
        In addition, just the subdomain near the object is tested, to save computational
        time.

        .. note:: The precision of the method is influenced by the
           complexity of the STL mesh, there is no guarantee it will work
           for all geometries. This feature is experimental, its
           interface may change in future releases.


        Parameters
        ----------
        filename : str, optional
            Filename of the STL file to be loaded and included in the cartesian
            domain, by default None
        scale : float, optional
            This parameters can be used to scale up the object when greater than
            one and scale it down when smaller than one, by default None
        rotate : dict, optional
            Rotate the object, including keyword arguments that are
            expected by :obj:`stl.mesh.Mesh.rotate`, like ``axis``,
            ``theta`` and ``point``.
            For more details, see `numpy-stl's documentation`_.
            By default None
        origin : dict, optional
            Specify the location of the origin point for the geometry.
            It is considered as the minimum value computed from all
            points in the object for each coordinate, after scaling
            and rotating them. The keys of the dictionary are the
            coordinate names (``x``, ``y`` and ``z``) and the values are the
            origin on that coordinate.
            For missing keys, the value is assumed as zero.
            By default None
        stl_mesh : stl.mesh.Mesh, optional
            For very customizable control over the 3D object, you
            can provide it directly. Note that none of the arguments
            above are applied in this case.
            For more details about how to create and modify the
            geometry, see `numpy-stl's documentation`_.
            By default None
        user_tol : float, optional
            Control the tolerance used to compute if a mesh node is
            inside or outside the object. Values smaller than the default may reduce the number of false negatives.
            By default :math:`2\pi`
        remp : bool, optional
            Add the geometry to the :obj:`xarray.DataArray` if
            :obj:`True` and removes it if :obj:`False`, by default True

        Returns
        -------
        :obj:`xarray.DataArray`
            Array with(out) the customized geometry

        Raises
        -------
        ValueError
            If neither :obj:`filename` or :obj:`stl_mesh` are specified
        ValueError
            If :obj:`stl_mesh` is not valid, the test is performed by :obj:`stl.mesh.Mesh.check`
        ValueError
            If :obj:`stl_mesh` is not closed, the test is performed by
            :obj:`stl.mesh.Mesh.is_closed`


        Examples
        --------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> epsi = xcompact3d_toolbox.init_epsi(prm, dask=True)
        >>> for key in epsi.keys():
        >>>     epsi[key] = epsi[key].geo.from_stl(
        ...         filename="My_file.stl",
        ...         scale=1.0,
        ...         rotate=dict(axis=[0, 0.5, 0], theta=math.radians(90)),
        ...         origin=dict(x=2, y=1, z=0),
        ...     )

        .. _`Dask`: https://dask.org/
        .. _`numpy-stl's documentation`: https://numpy-stl.readthedocs.io/en/latest/
        .. _`Numba`: http://numba.pydata.org/

        .. versionchanged:: 1.2.0
            All arguments changed to keyword-only.
        """

        def get_boundary(mesh_coord, coord):
            """Get the boundaries of the subdomain for a given direction (x, y, z)
            near the object. It returns a tuple of integers, representing the min and max
            indexes of the coordinate where we need to loop through.
            """
            min_val = coord.searchsorted(mesh_coord.min(), "left")
            max_val = coord.searchsorted(mesh_coord.max(), "right")
            return min_val, max_val

        if filename is not None and stl_mesh is None:
            stl_mesh = stl.mesh.Mesh.from_file(filename)

            if scale is not None:
                stl_mesh.vectors *= scale

            if rotate is not None:
                stl_mesh.rotate(**rotate)

            if origin is None:
                origin = {}

            stl_mesh.translate(
                [
                    origin.get("x", 0.0) - stl_mesh.x.min(),
                    origin.get("y", 0.0) - stl_mesh.y.min(),
                    origin.get("z", 0.0) - stl_mesh.z.min(),
                ]
            )

        if stl_mesh is None:
            msg = "Please, specify filename or stl_mesh"
            raise ValueError(msg)

        if not stl_mesh.check():
            msg = "stl_mesh is not valid"
            raise ValueError(msg)

        if not stl_mesh.is_closed():
            msg = "stl_mesh is not closed"
            raise ValueError(msg)

        x = self._data_array.x.data
        y = self._data_array.y.data
        z = self._data_array.z.data

        return self._data_array.where(
            ~_geometry_inside_mesh(
                stl_mesh.vectors.astype(np.double),
                x.astype(np.double),
                y.astype(np.double),
                z.astype(np.double),
                user_tol,
                get_boundary(stl_mesh.x, x),
                get_boundary(stl_mesh.y, y),
                get_boundary(stl_mesh.z, z),
            ),
            remp,
        )

    def cylinder(
        self, *, radius: float = 0.5, axis: str = "z", height: float | None = None, remp: bool = True, **kwargs
    ) -> xr.DataArray:
        r"""Draw a cylinder.

        Parameters
        ----------
        radius : float
            Cylinder's radius (the default is 0.5).
        axis : str
            Cylinder's axis (the default is ``"z"``).
        height : float or None
            Cylinder's height (the default is None), if None, it will take
            the entire axis, otherwise :math:`\pm h/2` is considered from the center.
        remp : bool
            Adds the geometry to the :obj:`xarray.DataArray` if True and removes
            it if False (the default is True).
        **kwargs : float
            Cylinder's center point.

        Returns
        -------
        :obj:`xarray.DataArray`
            Array with(out) the cylinder

        Raises
        -------
        KeyError
            Center coordinates must be valid dimensions

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> epsi = xcompact3d_toolbox.init_epsi(prm)
        >>> for key in epsi.keys():
        >>>     epsi[key] = epsi[key].geo.cylinder(x=4.0, y=5.0)

        .. versionchanged:: 1.2.0
            All arguments changed to keyword-only.
        """

        for key in kwargs:
            if key not in self._data_array.dims:
                raise DimensionNotFoundError(key)

        dis = 0.0
        for d in self._data_array.dims:
            if d == axis:
                continue
            dis = dis + (self._data_array[d] - kwargs.get(d, 0.0)) ** 2.0
        dis = np.sqrt(dis)

        if height is not None:
            height *= 0.5
            # Notice that r*10 is just to guarantee that the values are larger than r
            # and consequently outside the cylinder
            dis = dis.where(self._data_array[axis] <= kwargs.get(axis, 0.0) + height, radius * 10)  # type: ignore
            dis = dis.where(self._data_array[axis] >= kwargs.get(axis, 0.0) - height, radius * 10)  # type: ignore

        return self._data_array.where(dis > radius, remp)

    def box(self, *, remp: bool = True, **kwargs) -> xr.DataArray:
        """Draw a box.

        Parameters
        ----------
        remp : bool
            Adds the geometry to the :obj:`xarray.DataArray` if True and removes
            it if False (the default is True).
        **kwargs : tuple of float
            Box's boundaries.

        Returns
        -------
        :obj:`xarray.DataArray`
            Array with(out) the box

        Raises
        -------
        KeyError
            Boundaries coordinates must be valid dimensions

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> epsi = xcompact3d_toolbox.init_epsi(prm)
        >>> for key in epsi.keys():
        >>>     epsi[key] = epsi[key].geo.box(x=(2,5), y=(0,1))

        .. versionchanged:: 1.2.0
            All arguments changed to keyword-only.
        """

        for key in kwargs:
            if key not in self._data_array.dims:
                raise DimensionNotFoundError(key)

        tmp = xr.zeros_like(self._data_array)

        for key, value in kwargs.items():
            tmp = tmp.where(self._data_array[key] >= value[0], True)
            tmp = tmp.where(self._data_array[key] <= value[1], True)

        return self._data_array.where(tmp, remp)

    def square(self, *, length: float = 1.0, thickness: float = 0.1, remp: bool = True, **kwargs) -> xr.DataArray:
        """Draw a squared frame.

        Parameters
        ----------
        length : float
            Frame's external length (the default is 1).
        thickness : float
            Frames's thickness (the default is 0.1).
        remp : bool
            Adds the geometry to the :obj:`xarray.DataArray` if True and removes
            it if False (the default is True).
        **kwargs : float
            Frames's center.

        Returns
        -------
        :obj:`xarray.DataArray`
            Array with(out) the squared frame

        Raises
        -------
        KeyError
            Center coordinates must be valid dimensions

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> epsi = xcompact3d_toolbox.init_epsi(prm)
        >>> for key in epsi.keys():
        >>>     epsi[key] = epsi[key].geo.square(x=5, y=2, z=1)

        .. versionchanged:: 1.2.0
            All arguments changed to keyword-only.
        """
        for key in kwargs:
            if key not in self._data_array.dims:
                raise DimensionNotFoundError(key)

        center = {key: kwargs.get(key, 0.0) for key in self._data_array.dims}

        boundaries1 = {
            "x": (center["x"] - 0.5 * thickness, center["x"] + 0.5 * thickness),
            "y": (center["y"] - 0.5 * length, center["y"] + 0.5 * length),
            "z": (center["z"] - 0.5 * length, center["z"] + 0.5 * length),
        }
        tmp = self._data_array.geo.box(**boundaries1, remp=True)
        #
        length -= 2 * thickness
        boundaries2 = {
            "x": (center["x"] - 0.5 * thickness, center["x"] + 0.5 * thickness),
            "y": (center["y"] - 0.5 * length, center["y"] + 0.5 * length),
            "z": (center["z"] - 0.5 * length, center["z"] + 0.5 * length),
        }
        tmp = tmp.geo.box(**boundaries2, remp=False)
        #
        return self._data_array.where(tmp, remp)

    def sphere(self, *, radius: float = 0.5, remp: bool = True, **kwargs) -> xr.DataArray:
        """Draw a sphere.

        Parameters
        ----------
        radius : float
            Sphere's radius (the default is 0.5).
        remp : bool
            Adds the geometry to the :obj:`xarray.DataArray` if True and removes
            it if False (the default is True).
        **kwargs : float
            Sphere's center.

        Returns
        -------
        :obj:`xarray.DataArray`
            Array with(out) the sphere

        Raises
        -------
        KeyError
            Center coordinates must be valid dimensions

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> epsi = xcompact3d_toolbox.init_epsi(prm)
        >>> for key in epsi.keys():
        >>>     epsi[key] = epsi[key].geo.sphere(x=1, y=1, z=1)

        .. versionchanged:: 1.2.0
            All arguments changed to keyword-only.
        """
        for key in kwargs:
            if key not in self._data_array.dims:
                raise DimensionNotFoundError(key)

        dis = 0.0
        for d in self._data_array.dims:
            dis = dis + (self._data_array[d] - kwargs.get(d, 0.0)) ** 2.0
        dis = np.sqrt(dis)

        return self._data_array.where(dis > radius, remp)

    def ahmed_body(
        self, *, scale: float = 1.0, angle: float = 45.0, wheels: bool = False, remp: bool = True, **kwargs
    ) -> xr.DataArray:
        """Draw an Ahmed body.

        Parameters
        ----------
        scale : float
            Ahmed body's scale (the default is 1).
        angle : float
            Ahmed body's angle at the back, in degrees (the default is 45).
        wheel : bool
            Draw "wheels" if True (the default is False).
        remp : bool
            Adds the geometry to the :obj:`xarray.DataArray` if True and removes
            it if False (the default is True).
        **kwargs : float
            Ahmed body's center.

        Returns
        -------
        :obj:`xarray.DataArray`
            Array with(out) the Ahmed body

        Raises
        -------
        KeyError
            Center coordinates must be valid dimensions.
        NotImplementedError
            Body must be centered in ``z``.

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> epsi = xcompact3d_toolbox.init_epsi(prm)
        >>> for key in epsi.keys():
        >>>     epsi[key] = epsi[key].geo.ahmed_body(x=2)

        .. versionchanged:: 1.2.0
            All arguments changed to keyword-only.
        """

        import math

        s = scale / 288.0  # adimensional and scale factor

        for key in kwargs:
            if key not in self._data_array.dims:
                raise DimensionNotFoundError(key)

        if "x" not in kwargs:
            kwargs["x"] = 1.0
        if "y" not in kwargs:
            kwargs["y"] = 0.0
        if "z" not in kwargs:
            kwargs["z"] = 0.5 * self._data_array.z[-1].values - ((389.0 * s) / 2.0)
        else:
            # That is because of the mirror in Z
            msg = "Unsupported: Body must be centered in Z"
            raise NotImplementedError(msg)

        if scale != 1:
            msg = "Unsupported: Not prepared yet for scale != 1"
            raise NotImplementedError(msg)

        tmp = xr.zeros_like(self._data_array)
        tmp2 = xr.zeros_like(self._data_array)

        # the "corners" are the intersections between the cylinders

        # horizontal

        tmp = tmp.geo.cylinder(
            x=100.00 * s + kwargs["x"],
            y=150.00 * s + kwargs["y"],
            z=97.25 * s + kwargs["z"],
            axis="z",
            radius=100.00 * s,
            height=194.50 * s,
        )

        tmp = tmp.geo.cylinder(
            x=100.00 * s + kwargs["x"],
            y=238.00 * s + kwargs["y"],
            z=97.25 * s + kwargs["z"],
            axis="z",
            radius=100.00 * s,
            height=194.50 * s,
        )

        # vertical

        tmp2 = tmp2.geo.cylinder(
            x=100.00 * s + kwargs["x"],
            y=194.00 * s + kwargs["y"],
            z=100.00 * s + kwargs["z"],
            axis="y",
            radius=100.00 * s,
            height=288.00 * s,
        )

        # get intersection
        tmp = np.logical_and(tmp == True, tmp2 == True)

        del tmp2

        # now the regular cylinders

        tmp = tmp.geo.cylinder(
            x=100.00 * s + kwargs["x"],
            y=150.00 * s + kwargs["y"],
            z=147.25 * s + kwargs["z"] + 1.0,  # fixing issue #5
            axis="z",
            radius=100.00 * s,
            height=94.50 * s + 2.0,  # fixing issue #5
        )

        tmp = tmp.geo.cylinder(
            x=100.00 * s + kwargs["x"],
            y=238.00 * s + kwargs["y"],
            z=147.25 * s + kwargs["z"] + 1.0,  # fixing issue #5
            axis="z",
            radius=100.00 * s,
            height=94.50 * s + 2.0,  # fixing issue #5
        )

        tmp = tmp.geo.cylinder(
            x=100.00 * s + kwargs["x"],
            y=194.00 * s + kwargs["y"],
            z=100.00 * s + kwargs["z"],
            axis="y",
            radius=100.00 * s,
            height=88.00 * s,
        )

        if wheels:
            tmp = tmp.geo.cylinder(
                x=200.00 * s + kwargs["x"],
                y=25.00 * s + kwargs["y"],
                z=46.50 * s + kwargs["z"],
                axis="y",
                radius=15.00 * s,
                height=50.00 * s,
            )

            tmp = tmp.geo.cylinder(
                x=725.00 * s + kwargs["x"],
                y=25.00 * s + kwargs["y"],
                z=46.50 * s + kwargs["z"],
                axis="y",
                radius=15.00 * s,
                height=50.00 * s,
            )

        # the boxes
        tmp = tmp.geo.box(
            x=(kwargs["x"], 200.00 * s + kwargs["x"]),
            y=(150.00 * s + kwargs["y"], 238.00 * s + kwargs["y"]),
            z=(100.00 * s + kwargs["z"], 194.50 * s + kwargs["z"]),
        )

        tmp = tmp.geo.box(
            x=(100.00 * s + kwargs["x"], 1044.00 * s + kwargs["x"]),
            y=(50.00 * s + kwargs["y"], 338.00 * s + kwargs["y"]),
            z=(kwargs["z"], 194.50 * s + kwargs["z"]),
        )

        # and finally a mirror
        tmp = tmp.geo.mirror(dim="z")

        # Angle in the back
        hipo = (93.80 / math.sin(math.radians(25))) * s
        adj = math.cos(math.radians(angle)) * hipo
        opo = math.sin(math.radians(angle)) * hipo

        x = [1044.00 * s - adj + kwargs["x"], 1044.00 * s + kwargs["x"]]
        y = [338.00 * s + kwargs["y"], 338.00 * s - opo + kwargs["y"]]

        coefficients = np.polyfit(x, y, 1)

        cut = self._data_array.x * coefficients[0] + coefficients[1]

        tmp = tmp.where(self._data_array.y <= cut, False)

        return self._data_array.where(np.logical_not(tmp), remp)

    def mirror(self, dim: str = "x") -> xr.DataArray:
        """Mirror the :math:`\\epsilon` array with respect to the central plane
        in the direction ``dim``.

        Parameters
        ----------
        dim : str
            Reference for the mirror (the default is ``x``).

        Returns
        -------
        :obj:`xarray.DataArray`
            Mirrored array

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> epsi = xcompact3d_toolbox.init_epsi(prm)
        >>> for key in epsi.keys():
        >>>     epsi[key] = epsi[key].geo.cylinder(x=4, y=5).geo.mirror("x")

        """
        return self._data_array.where(
            self._data_array[dim] <= self._data_array[dim][-1] / 2.0,
            self._data_array[{dim: slice(None, None, -1)}].values,
        )


@numba.njit
def _geometry_inside_mesh(triangles, x, y, z, user_tol, lim_x, lim_y, lim_z):
    result = np.zeros((x.size, y.size, z.size), dtype=numba.boolean)

    for i in range(*lim_x):
        for j in range(*lim_y):
            for k in range(*lim_z):
                result[i, j, k] = _point_in_geometry(triangles, x[i], y[j], z[k], user_tol)

    return result


@numba.njit
def _anorm2(x):
    # Compute euclidean norm
    return np.sqrt(np.sum(x**2.0))


@numba.njit
def _adet(x, y, z):
    # Compute 3x3 determinant
    ret = np.multiply(np.multiply(x[0], y[1]), z[2])
    ret += np.multiply(np.multiply(y[0], z[1]), x[2])
    ret += np.multiply(np.multiply(z[0], x[1]), y[2])
    ret -= np.multiply(np.multiply(z[0], y[1]), x[2])
    ret -= np.multiply(np.multiply(y[0], x[1]), z[2])
    ret -= np.multiply(np.multiply(x[0], z[1]), y[2])
    return ret


@numba.njit
def _point_in_geometry(triangles, x, y, z, user_tol):
    """
    The methodology is based on the work of:

    * Jacobson, A., Kavan, L., & Sorkine-Hornung,
        O. (2013). Robust inside-outside segmentation
        using generalized winding numbers. ACM
        Transactions on Graphics (TOG), 32(4), 1-12.

    The Python implementation is an adaptation of
    `inside-3d-mesh <https://github.com/marmakoide/inside-3d-mesh>`_,
    by `Devert Alexandre <https://github.com/marmakoide>`_,
    licensed under the MIT License.
    """
    array_x = np.array((x, y, z), dtype=triangles.dtype)

    # One generalized winding number per input vertex
    ret = triangles.dtype.type(0.0)

    # Accumulate generalized winding number for each triangle
    for array_u, array_v, array_w in triangles:
        array_a, array_b, array_c = array_u - array_x, array_v - array_x, array_w - array_x
        omega = _adet(array_a, array_b, array_c)

        a, b, c = _anorm2(array_a), _anorm2(array_b), _anorm2(array_c)
        d = a * b * c
        d += c * np.sum(np.multiply(array_a, array_b))
        d += a * np.sum(np.multiply(array_b, array_c))
        d += b * np.sum(np.multiply(array_c, array_a))

        ret += np.arctan2(omega, d)

    return ret >= user_tol
