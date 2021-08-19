# -*- coding: utf-8 -*-
"""
Manipulate physical and computational parameters. It contains variables and
methods designed to be a bridge between Xcompact3d and Python applications for
pre and post-processing.
"""

import warnings

import numpy as np
import traitlets

from .io import FilenameProperties, i3d_to_dict, prm_to_dict, read_field
from .io import read_temporal_series, write_xdmf
from .mesh import Mesh3D
from .param import boundary_condition, param


class ParametersBasicParam(traitlets.HasTraits):

    p_row, p_col = [
        traitlets.Int(default_value=0, min=0).tag(
            group="BasicParam",
            desc=f"{name} for domain decomposition and parallel computation",
        )
        for name in ["Row partition", "Column partition"]
    ]
    """int: Defines the domain decomposition for (large-scale) parallel computation.

    Notes
    -----
        The product ``p_row * p_col`` must be equal to the number of
        computational cores where Xcompact3d will run.
        More information can be found at `2DECOMP&FFT`_.

        ``p_row = p_col = 0`` activates auto-tunning.

    .. _2DECOMP&FFT:
        http://www.2decomp.org
    """

    itype = traitlets.Int(default_value=10, min=0, max=10).tag(
        group="BasicParam",
        desc="Flow configuration (1:Lock-exchange, 2:TGV, 3:Channel, and others)",
    )
    """int: Sets the flow configuration, each one is specified in a different
    ``BC.<flow-configuration>.f90`` file (see `Xcompact3d/src`_), they are:

    * 0 - User configuration;
    * 1 - Turbidity Current in Lock-Release;
    * 2 - Taylor-Green Vortex;
    * 3 - Periodic Turbulent Channel;
    * 5 - Flow around a Cylinder;
    * 6 - Debug Schemes (for developers);
    * 7 - Mixing Layer;
    * 9 - Turbulent Boundary Layer;
    * 10 - `Sandbox`_.

    .. _Xcompact3d/src:
        https://github.com/fschuch/Xcompact3d/tree/master/src
    """

    iin = traitlets.Int(default_value=0, min=0, max=2).tag(
        group="BasicParam", desc="Defines perturbation at initial condition"
    )
    """int: Defines perturbation at the initial condition:

    * 0 - No random noise (default);
    * 1 - Random noise with amplitude of :obj:`init_noise`;
    * 2 - Random noise with fixed seed
      (important for reproducibility, development and debugging)
      and amplitude of :obj:`init_noise`.

    Notes
    -----
        The exactly behavior may be different according to each flow configuration.
    """

    nx, ny, nz = [
        traitlets.Int().tag(group="BasicParam", desc=f"{name.upper()}-direction nodes")
        for name in "x y z".split()
    ]
    """int: Number of mesh points.
    """

    xlx, yly, zlz = [
        traitlets.Float().tag(
            group="BasicParam", desc=f"Size of the box in {name}-direction"
        )
        for name in "x y z".split()
    ]
    """float: Domain size.
    """

    nclx1 = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="BasicParam", desc="Velocity boundary condition where x=0"
    )
    """int: Boundary condition for velocity field where :math:`x=0`, the options are:

        * 0 - Periodic;
        * 1 - Free-slip;
        * 2 - Inflow.
    """

    nclxn = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="BasicParam", desc="Velocity boundary condition where x=xlx"
    )
    """int: Boundary condition for velocity field where :math:`x=L_x`, the options are:

    * 0 - Periodic;
    * 1 - Free-slip;
    * 2 - Convective outflow.
    """

    ncly1 = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="BasicParam", desc="Velocity boundary condition where y=0"
    )
    """int: Boundary condition for velocity field where :math:`y=0`, the options are:

        * 0 - Periodic;
        * 1 - Free-slip;
        * 2 - No-slip.
    """

    nclyn = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="BasicParam", desc="Velocity boundary condition where y=yly"
    )
    """int: Boundary condition for velocity field where :math:`y=L_y`, the options are:

        * 0 - Periodic;
        * 1 - Free-slip;
        * 2 - No-slip.
    """

    nclz1 = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="BasicParam", desc="Velocity boundary condition where z=0"
    )
    """int: Boundary condition for velocity field where :math:`z=0`, the options are:

        * 0 - Periodic;
        * 1 - Free-slip;
        * 2 - No-slip.
    """

    nclzn = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="BasicParam", desc="Velocity boundary condition where z=zlz"
    )
    """int: Boundary condition for velocity field where :math:`z=L_z`, the options are:

        * 0 - Periodic;
        * 1 - Free-slip;
        * 2 - No-slip.
    """

    ivisu = traitlets.Int(default_value=1, min=0, max=1).tag(
        group="BasicParam",
        desc="Enable store snapshots at a frequency ioutput (0: No, 1: Yes)",
    )
    """int: Enables store snapshots at a frequency :obj:`ioutput`:

        * 0 - No;
        * 1 - Yes (default).
    """

    ipost = traitlets.Int(default_value=0, min=0, max=1).tag(
        group="BasicParam",
        desc="Enables online postprocessing at a frequency iprocessing (0: No, 1: Yes)",
    )
    """int: Enables online postprocessing at a frequency :obj:`iprocessing`:

        * 0 - No;
        * 1 - Yes (default).

      .. note:: The computation for each case is specified at the ``BC.<flow-configuration>.f90`` file.
    """

    ilesmod = traitlets.Int(default_value=0, min=0, max=1).tag(
        group="BasicParam", desc="Enables Large-Eddy methodologies (0: No, 1: Yes)"
    )
    """int: Enables Large-Eddy methodologies:

        * 0 - No (also forces :obj:`nu0nu` and :obj:`cnu` to 4.0 and 0.44, respectively);
        * 1 - Yes (alse activates the namespace **LESModel**, with variables like :obj:`jles`).
    """

    istret = traitlets.Int(default_value=0, min=0, max=3).tag(
        group="BasicParam",
        desc="y mesh refinement (0:no, 1:center, 2:both sides, 3:bottom)",
    )
    """int: Controls mesh refinement in **y**:

    * 0 - No refinement (default);
    * 1 - Refinement at the center;
    * 2 - Both sides;
    * 3 - Just near the bottom.

    Notes
    -----
        See :obj:`beta`.
    """

    beta = traitlets.Float(default_value=1.0, min=0).tag(
        group="BasicParam", desc="Refinement parameter"
    )
    """float: Refinement factor in **y**.

    Notes
    -----
        Only necessary if :obj:`istret` :math:`\\ne` 0.
    """

    dt = traitlets.Float(default_value=1e-3, min=0.0).tag(
        group="BasicParam", desc="Time step"
    )
    """float: Time step :math:`(\\Delta t)`.
    """

    ifirst = traitlets.Int(default_value=0, min=0).tag(
        group="BasicParam", desc="The number for the first iteration"
    )
    """int: The number for the first iteration.
    """

    ilast = traitlets.Int(default_value=0, min=0).tag(
        group="BasicParam", desc="The number for the last iteration"
    )
    """int: The number for the last iteration.
    """

    re = traitlets.Float(default_value=1e3).tag(
        group="BasicParam", desc="Reynolds number"
    )
    """float: Reynolds number :math:`(Re)`.
    """

    init_noise = traitlets.Float(default_value=0.0).tag(
        group="BasicParam", desc="Turbulence intensity (1=100%) !! Initial condition"
    )
    """float: Random number amplitude at initial condition.

    Notes
    -----
        The exactly behavior may be different according to each flow configuration.

        Only necessary if :obj:`iin` :math:`\\ne` 0.
    """

    inflow_noise = traitlets.Float(default_value=0.0).tag(
        group="BasicParam", desc="Turbulence intensity (1=100%) !! Inflow condition"
    )
    """float: Random number amplitude at inflow boundary (where :math:`x=0`).

    Notes
    -----
        Only necessary if :obj:`nclx1` is equal to 2.
    """

    iibm = traitlets.Int(default_value=0, min=0, max=2).tag(
        group="BasicParam", desc="Flag for immersed boundary method (0: No, 1: Yes)"
    )
    """int: Enables Immersed Boundary Method (IBM):

    * 0 - Off (default);
    * 1 - On with direct forcing method, i.e.,
      it sets velocity to zero inside the solid body;
    * 2 - On with alternating forcing method, i.e, it uses
      Lagrangian Interpolators to define the velocity inside the body
      and imposes no-slip condition at the solid/fluid interface.

    Any option greater than zero activates the namespace **ibmstuff**, for variables like :obj:`nobjmax` and :obj:`nraf`.
    """

    numscalar = traitlets.Int(default_value=0, min=0, max=9).tag(
        group="BasicParam", desc="Number of scalar fractions"
    )
    """int: Number of scalar fraction, which can have different properties.
    Any option greater than zero activates the namespace **numscalar**,
    for variables like :obj:`sc`, :obj:`ri`, :obj:`uset` and others.

    Notes
    -----
        More than 9 will bug Xcompact3d, because it handles the I/O for
        scalar fields with just one digit
    """

    gravx, gravy, gravz = [
        traitlets.Float(default_value=0.0).tag(
            group="BasicParam", desc=f"Gravity unitary vector in {name}-direction"
        )
        for name in "x y z".split()
    ]
    """float: Component of the unitary vector pointing in the gravity's direction.
    """

    def __init__(self):
        super(ParametersBasicParam, self).__init__()


class ParametersNumOptions(traitlets.HasTraits):
    ifirstder = traitlets.Int(default_value=4, min=1, max=4).tag(group="NumOptions")
    """int: Scheme for first order derivative:

    * 1 - 2nd central;
    * 2 - 4th central;
    * 3 - 4th compact;
    * 4 - 6th compact (default).
    """

    isecondder = traitlets.Int(default_value=4, min=1, max=5).tag(
        group="NumOptions", desc="Scheme for first order derivative"
    )
    """int: Scheme for second order derivative:

    * 1 - 2nd central;
    * 2 - 4th central;
    * 3 - 4th compact;
    * 4 - 6th compact (default);
    * 5 - Hyperviscous 6th.
    """

    ipinter = traitlets.Int(3)

    itimescheme = traitlets.Int(default_value=3, min=1, max=7).tag(
        group="NumOptions",
        desc="Time integration scheme (1: Euler, 2: AB2, 3: AB3, 5: RK3)",
    )
    """int: Time integration scheme:

    * 1 - Euler;
    * 2 - AB2;
    * 3 - AB3 (default);
    * 5 - RK3;
    * 7 - Semi-implicit.
    """

    nu0nu = traitlets.Float(default_value=4, min=0.0).tag(
        group="NumOptions",
        desc="Ratio between hyperviscosity/viscosity at nu (dissipation factor intensity)",
    )
    """float: Ratio between hyperviscosity/viscosity at nu.
    """

    cnu = traitlets.Float(default_value=0.44, min=0.0).tag(
        group="NumOptions",
        desc="Ratio between hyperviscosity at km=2/3π and kc=π (dissipation factor range)",
    )
    """float: Ratio between hyperviscosity at :math:`k_m=2/3\\pi` and :math:`k_c= \\pi`.
    """

    def __init__(self):
        super(ParametersNumOptions, self).__init__()


class ParametersInOutParam(traitlets.HasTraits):
    irestart = traitlets.Int(default_value=0, min=0, max=1).tag(
        group="InOutParam", desc="Read initial flow field (0: No, 1: Yes)"
    )
    """int: Reads initial flow field if equals to 1.
    """

    nvisu = traitlets.Int(default_value=1, min=1).tag(
        group="InOutParam", desc="Size for visualization collection"
    )
    """int: Size for visual collection.
    """

    icheckpoint = traitlets.Int(default_value=1000, min=1).tag(
        group="InOutParam", desc="Frequency for writing backup file"
    )
    """int: Frequency for writing restart file.
    """

    ioutput = traitlets.Int(default_value=1000, min=1).tag(
        group="InOutParam", desc="Frequency for visualization file"
    )
    """int: Frequency for visualization (3D snapshots).
    """

    iprocessing = traitlets.Int(default_value=1000, min=1).tag(
        group="InOutParam", desc="Frequency for online postprocessing"
    )
    """int: Frequency for online postprocessing.
    """

    filenamedigits = traitlets.Int(default_value=0, min=0, max=1).tag(
        group="InOutParam",
        desc="Controls the way that the output binary files are enumerated",
    )
    """int: Controls the way that the output binary files are enumerated:

    * 0 - Files receive the number according to the current timestep (default);
    * 1 - Continuous counting.
    """

    ifilenameformat = traitlets.Unicode(default_value="(I9.9)").tag(
        group="InOutParam",
        desc="The number of digits used to name the output binary files",
    )
    """str: The number of digits used to name the output binary files,
    in Fortran format (default is ``(I9.9)``).
    """

    def __init__(self):
        super(ParametersInOutParam, self).__init__()


class ParametersScalarParam(traitlets.HasTraits):
    sc = traitlets.List(trait=traitlets.Float()).tag(
        group="ScalarParam", desc="Schmidt number(s)"
    )
    """:obj:`list` of :obj:`float`: Schmidt number(s).
    """

    ri = traitlets.List(trait=traitlets.Float()).tag(
        group="ScalarParam", desc="Richardson number(s)"
    )
    """:obj:`list` of :obj:`float`: Richardson number(s).
    """

    uset = traitlets.List(trait=traitlets.Float()).tag(
        group="ScalarParam", desc="Settling velocity(ies)"
    )
    """:obj:`list` of :obj:`float`: Settling velocity(s).
    """

    cp = traitlets.List(trait=traitlets.Float()).tag(
        group="ScalarParam", desc="Initial concentration(s)"
    )
    """:obj:`list` of :obj:`float`: Initial concentration(s).
    """

    scalar_lbound = traitlets.List(trait=traitlets.Float(default_value=-1e6)).tag(
        group="ScalarParam", desc="Lower scalar bound(s), for clipping methodology."
    )
    """:obj:`list` of :obj:`float`: Lower scalar bound(s), for clipping methodology.
    """

    scalar_ubound = traitlets.List(trait=traitlets.Float(default_value=1e6)).tag(
        group="ScalarParam", desc="Upper scalar bound(s), for clipping methodology."
    )
    """:obj:`list` of :obj:`float`: Upper scalar bound(s), for clipping methodology.
    """

    iibmS = traitlets.Int(default_value=0, min=0, max=3).tag(
        group="ScalarParam",
        desc="Enables Immersed Boundary Method (IBM) for scalar field(s) (alpha version)",
    )
    """int: Enables Immersed Boundary Method (IBM) for scalar field(s):

    * 0 - Off (default);
    * 1 - On with direct forcing method, i.e.,
      it sets scalar concentration to zero inside the solid body;
    * 2 - On with alternating forcing method, i.e, it uses
      Lagrangian Interpolators to define the scalar field inside the body
      and imposes zero value at the solid/fluid interface.
    * 3 - On with alternating forcing method, but now the Lagrangian
      Interpolators are set to impose no-flux for the scalar field at the
      solid/fluid interface.

      .. note:: It is only recommended if the normal vectors to the object's
            faces are aligned with one of the coordinate axes.
    """

    nclxS1 = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="ScalarParam", desc="Scalar boundary condition where x=0"
    )
    """int: Boundary condition for scalar field(s) where :math:`x=0`, the options are:

    * 0 - Periodic;
    * 1 - No-flux;
    * 2 - Inflow.
    """

    nclxSn = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="ScalarParam", desc="Scalar boundary condition where x=xlx"
    )
    """int: Boundary condition for scalar field(s) where :math:`x=L_x`, the options are:

    * 0 - Periodic;
    * 1 - No-flux;
    * 2 - Convective outflow.
    """

    nclyS1 = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="ScalarParam", desc="Scalar boundary condition where y=0"
    )
    """int: Boundary condition for scalar field(s) where :math:`y=0`, the options are:

    * 0 - Periodic;
    * 1 - No-flux;
    * 2 - Dirichlet.
    """

    nclySn = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="ScalarParam", desc="Scalar boundary condition where y=yly"
    )
    """int: Boundary condition for scalar field(s) where :math:`y=L_y`, the options are:

    * 0 - Periodic;
    * 1 - No-flux;
    * 2 - Dirichlet.
    """

    nclzS1 = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="ScalarParam", desc="Scalar boundary condition where z=0"
    )
    """int: Boundary condition for scalar field(s) where :math:`z=0`, the options are:

    * 0 - Periodic;
    * 1 - No-flux;
    * 2 - Dirichlet.
    """

    nclzSn = traitlets.Int(default_value=2, min=0, max=2).tag(
        group="ScalarParam", desc="Scalar boundary condition where z=zlz"
    )
    """int: Boundary condition for scalar field(s) where :math:`z=L_z`, the options are:

    * 0 - Periodic;
    * 1 - No-flux;
    * 2 - Dirichlet.
    """

    def __init__(self):
        super(ParametersScalarParam, self).__init__()


class ParametersLESModel(traitlets.HasTraits):
    jles = traitlets.Int(default_value=4, min=0, max=4).tag(
        group="LESModel",
        desc="LES Model (1: Phys Smag, 2: Phys WALE, 3: Phys dyn. Smag, 4: iSVV)",
    )
    """int: Chooses LES model, they are:

    * 0 - No model (DNS);
    * 1 - Phys Smag;
    * 2 - Phys WALE;
    * 3 - Phys dyn. Smag;
    * 4 - iSVV.

    """

    def __init__(self):
        super(ParametersLESModel, self).__init__()


class ParametersIbmStuff(traitlets.HasTraits):
    nobjmax = traitlets.Int(default_value=1, min=1).tag(
        group="ibmstuff", desc="Maximum number of objects in any direction"
    )
    """int: Maximum number of objects in any direction. It is defined
        automatically at :obj:`gene_epsi_3D`.
    """

    nraf = traitlets.Int(default_value=10, min=1).tag(
        group="ibmstuff",
        desc="Level of refinement for iibm==2 to find the surface of the immersed object",
    )
    """int: "Level of refinement for :obj:`iibm` equals to 2, to find the surface of the immersed object"
    """

    def __init__(self):
        super(ParametersIbmStuff, self).__init__()


class ParametersExtras(traitlets.HasTraits):
    filename = traitlets.Unicode(default_value="input.i3d").tag()
    """str: Filename for the ``.i3d`` file.
    """

    mesh = traitlets.Instance(klass=Mesh3D)

    filename_properties = traitlets.Instance(klass=FilenameProperties)

    dx, dy, dz = [traitlets.Float().tag() for _ in "x y z".split()]
    """float: Mesh resolution.
    """

    ncores = traitlets.Int(default_value=4, min=1).tag()
    """int: Number of computational cores where Xcompact3d will run.
    """

    size = traitlets.Unicode().tag()
    """str: Auxiliar variable indicating the demanded space in disc
    """

    def __init__(self):
        super(ParametersExtras, self).__init__()
        self.mesh = Mesh3D()
        self.filename_properties = FilenameProperties()
        self._link_mesh_and_parameters()

    def _link_mesh_and_parameters(self):
        for dim in "xyz":
            traitlets.link((getattr(self.mesh, dim), "length"), (self, f"{dim}l{dim}"))
            traitlets.link((getattr(self.mesh, dim), "grid_size"), (self, f"n{dim}"))
            traitlets.link((getattr(self.mesh, dim), "delta"), (self, f"d{dim}"))
        traitlets.link((self.mesh.y, "istret"), (self, "istret"))
        traitlets.link((self.mesh.y, "beta"), (self, "beta"))


class Parameters(
    ParametersBasicParam,
    ParametersNumOptions,
    ParametersInOutParam,
    ParametersScalarParam,
    ParametersLESModel,
    ParametersIbmStuff,
    ParametersExtras,
):
    """The physical and computational parameters are built on top of `traitlets`_.
    It is a framework that lets Python classes have attributes with type checking,
    dynamically calculated default values, and ‘on change’ callbacks.
    So, many of the parameters are validated regarding the type, business rules,
    and the range of values supported by XCompact3d.

    There are methods to handle the parameters file, to read the binary
    arrays produced by XCompact3d and also to write the xdmf file, so the binary
    fields can be opened in any external visualization tool.

    * :obj:`xcompact3d_toolbox.parameters.ParametersBasicParam`;
    * :obj:`xcompact3d_toolbox.parameters.ParametersNumOptions`;
    * :obj:`xcompact3d_toolbox.parameters.ParametersInOutParam`;
    * :obj:`xcompact3d_toolbox.parameters.ParametersScalarParam`;
    * :obj:`xcompact3d_toolbox.parameters.ParametersLESModel`;
    * :obj:`xcompact3d_toolbox.parameters.ParametersIbmStuff`;
    * :obj:`xcompact3d_toolbox.parameters.ParametersExtras`;

    In addition, there are `ipywidgets`_ for a friendly user interface,
    see :obj:`xcompact3d_toolbox.gui.ParametersGui`.

    .. _traitlets:
        https://traitlets.readthedocs.io/en/stable/index.html
    .. _ipywidgets:
        https://ipywidgets.readthedocs.io/en/latest/
    .. _#2:
        https://github.com/fschuch/xcompact3d_toolbox/issues/2

    Notes
    -----
        This is a work in progress, not all parameters are covered yet.
    """

    def __init__(self, **kwargs):
        """Initializes the Parameters Class.

        Parameters
        ----------
        **kwargs
            Keyword arguments for valid attributes.

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid attribute.

        Examples
        -------

        There are a few ways to initialize the class.

        First, calling it with no
        arguments initializes all variables with default value:

        >>> prm = xcompact3d_toolbox.Parameters()

        It is possible to set any values afterwards (including new attributes):

        >>> prm.re = 1e6

        Second, we can specify some values, and let the missing ones be
        initialized with default value:

        >>> prm = x3d.Parameters(
        ...     filename = 'example.i3d',
        ...     itype = 10,
        ...     nx = 257,
        ...     ny = 129,
        ...     nz = 32,
        ...     xlx = 15.0,
        ...     yly = 10.0,
        ...     zlz = 3.0,
        ...     nclx1 = 2,
        ...     nclxn = 2,
        ...     ncly1 = 1,
        ...     nclyn = 1,
        ...     nclz1 = 0,
        ...     nclzn = 0,
        ...     re = 300.0,
        ...     init_noise = 0.0125,
        ...     dt = 0.0025,
        ...     ilast = 45000,
        ...     ioutput = 200,
        ...     iprocessing = 50
        ... )

        And finally, it is possible to read the parameters from the disc:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile = 'example.i3d')

        It also supports the previous parameters file format (see `#7`_):

        >>> prm = xcompact3d_toolbox.Parameters(loadfile = 'incompact3d.prm')

        .. _#7:
            https://github.com/fschuch/xcompact3d_toolbox/issues/7

        """

        super(Parameters, self).__init__()

        if "loadfile" in kwargs.keys():
            self.filename = kwargs.get("loadfile")
            self.load()
            del kwargs["loadfile"]

        self.set(**kwargs)

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for name in self.trait_names():
            group = self.trait_metadata(name, "group")
            if group is None:
                continue
            value = getattr(self, name)
            if value == self.trait_defaults(name):
                continue
            string += f"    {name} = {value},\n"
        string += ")"
        return string

    def __str__(self):
        """Representation of the parameters class, similar to the
        representation of the .i3d file."""
        # These groups are demanded by Xcompact3d, see parameters.f90
        dictionary = dict(
            BasicParam={}, NumOptions={}, InOutParam={}, Statistics={}, CASE={},
        )
        for name in self.trait_names():
            # if skip_default:
            #     if getattr(self, name) == self.trait_defaults(name):
            #         continue
            group = self.trait_metadata(name, "group")
            if group is not None:
                if group not in dictionary.keys():
                    dictionary[group] = {}
                dictionary[group][name] = getattr(self, name)

        # This block is not handled by x3d if ilesmod is off
        if "LESModel" in dictionary.keys() and self.ilesmod == 0:
            del dictionary["LESModel"]

        # This block is not handled by x3d if iibm is off
        if "ibmstuff" in dictionary.keys() and self.iibm == 0:
            del dictionary["ibmstuff"]

        # This block is not handled by x3d if numscalar is 0
        if "ScalarParam" in dictionary.keys() and self.numscalar == 0:
            del dictionary["ScalarParam"]

        string = ""

        string += "! -*- mode: f90 -*-\n"

        for blockkey, block in dictionary.items():

            string += "\n"
            string += "!===================\n"
            string += "&" + blockkey + "\n"
            string += "!===================\n"
            string += "\n"

            for paramkey, param in block.items():
                # get description to print together with the values
                description = self.trait_metadata(paramkey, "desc")
                if description is None:
                    description = ""
                # Check if param is a list or not
                if isinstance(param, list):
                    for n, p in enumerate(param):
                        string += f"{paramkey+'('+str(n+1)+')':>15} = {p:<15} {'! '+description}\n"
                # Check if param is a string
                elif isinstance(param, str):
                    param = "'" + param + "'"
                    string += f"{paramkey:>15} = {param:<15} {'! '+description}\n"
                else:
                    string += f"{paramkey:>15} = {param:<15} {'! '+description}\n"
            string += "\n"
            string += "/End\n"

        return string

    @traitlets.observe(
        "nclx1",
        "nclxn",
        "nclxS1",
        "nclxSn",
        "ncly1",
        "nclyn",
        "nclyS1",
        "nclySn",
        "nclz1",
        "nclzn",
        "nclzS1",
        "nclzSn",
    )
    def _observe_bc(self, change):
        #
        dim = change["name"][3]  # It will be x, y or z
        #
        if change["new"] == 0:
            for BC in f"ncl{dim}1 ncl{dim}n ncl{dim}S1 ncl{dim}Sn".split():
                setattr(self, BC, 0)
                getattr(self.mesh, dim)
            setattr(getattr(self.mesh, dim), "is_periodic", True)
        if change["old"] == 0 and change["new"] != 0:
            for BC in f"ncl{dim}1 ncl{dim}n ncl{dim}S1 ncl{dim}Sn".split():
                setattr(self, BC, change["new"])
            setattr(getattr(self.mesh, dim), "is_periodic", False)

    @traitlets.observe("p_row", "p_col", "ncores")
    def _observe_2Decomp(self, change):
        if change["name"] == "ncores":
            self.p_row, self.p_col = 0, 0
        elif change["name"] == "p_row":
            try:
                self.p_col = self.ncores // self.p_row
            except:
                self.p_col = 0
        elif change["name"] == "p_col":
            try:
                self.p_row = self.ncores // self.p_col
            except:
                self.p_row = 0

    @traitlets.observe("ilesmod")
    def _observe_ilesmod(self, change):
        if change["new"] == 0:
            # It is coded at xcompact3d, look at parameters.f90
            self.nu0nu, self.cnu = 4.0, 0.44

    @traitlets.observe(
        "numscalar",
        "nx",
        "ny",
        "nz",
        "nvisu",
        "icheckpoint",
        "ioutput",
        "iprocessing",
        "ilast",
    )
    def _observe_size(self, change):
        def convert_bytes(num):
            """
            this function will convert bytes to MB.... GB... etc
            """
            step_unit = 1000.0  # 1024 bad the size

            for x in ["bytes", "KB", "MB", "GB", "TB"]:
                if num < step_unit:
                    return "%3.1f %s" % (num, x)
                num /= step_unit

        prec = 4 if param["mytype"] == np.float32 else 8

        # Restart Size from tools.f90
        count = 3 + self.numscalar  # ux, uy, uz, phi
        # Previous time-step if necessary
        if self.itimescheme in [3, 7]:
            count *= 3
        elif self.itimescheme == 2:
            count *= 2
        count += 1  # pp
        count *= (
            self.nx * self.ny * self.nz * prec * (self.ilast // self.icheckpoint - 1)
        )

        # 3D from visu.f90: ux, uy, uz, pp and phi
        count += (
            (4 + self.numscalar)
            * self.nx
            * self.ny
            * self.nz
            * prec
            * self.ilast
            // self.ioutput
        )

        # 2D planes from BC.Sandbox.f90
        if self.itype == 10:
            # xy planes avg and central plane for ux, uy, uz and phi
            count += (
                2
                * (3 + self.numscalar)
                * self.nx
                * self.ny
                * prec
                * self.ilast
                // self.iprocessing
            )
            # xz planes avg, top and bot for ux, uy, uz and phi
            count += (
                3
                * (3 + self.numscalar)
                * self.nx
                * self.nz
                * prec
                * self.ilast
                // self.iprocessing
            )

        self.size = convert_bytes(count)

    def get_boundary_condition(self, var):
        """This method returns the appropriate boundary parameters that are
        expected by the derivative functions.

        Parameters
        ----------
        var : str
            Variable name. The supported options are ``ux``, ``uy``,
            ``uz``, ``pp`` and ``phi``, otherwise the method returns a default
            option.

        Returns
        -------
        dict
            Constants the boundary conditions for the desired variable.

        Examples
        -------

        >>> prm = x3d.Parameters()
        >>> prm.get_boundary_condition('ux')
        {'x': {'ncl1': 1, 'ncln': 1, 'npaire': 0},
        'y': {'ncl1': 1, 'ncln': 2, 'npaire': 1, 'istret': 0, 'beta': 0.75},
        'z': {'ncl1': 0, 'ncln': 0, 'npaire': 1}}

        It is possible to store this information as an attribute in any
        :obj:`xarray.DataArray`:

        >>> DataArray.attrs['BC'] = prm.get_boundary_condition('ux')

        So the correct boundary conditions will be used to compute the derivatives:

        >>> DataArray.x3d.first_derivative('x')
        >>> DataArray.x3d.second_derivative('x')

        Notes
        -----
        The **atribute** ``BC`` is automatically defined for ``ux``, ``uy``,
        ``uz``, ``pp`` and ``phi`` when read from the disc with
        :obj:`xcompact3d_toolbox.Parameters.read_field` and
        :obj:`xcompact3d_toolbox.Parameters.read_all_fields` or initialized at
        :obj:`xcompact3d_toolbox.sandbox.init_dataset`.
        """

        return boundary_condition(self, var)

    def set(self, **kwargs):
        # Boundary conditions are high priority in order to avoid bugs
        for bc in "nclx1 nclxn ncly1 nclyn nclz1 nclzn".split():
            if bc in kwargs:
                setattr(self, bc, kwargs.get(bc))

        if "filename_properties" in kwargs.keys():
            self.filename_properties.set(**kwargs.get("filename_properties"))
            del kwargs["filename_properties"]

        for key, arg in kwargs.items():
            if key not in self.trait_names():
                warnings.warn(f"{key} is not a valid parameter and was not loaded")
            setattr(self, key, arg)

    def load(self):
        """Loads all valid attributes from the parameters file.

        An attribute is considered valid if it has a ``tag`` named ``group``,
        witch assigns it to the respective namespace at the ``.i3d`` file.

        It also includes support for the previous format ``.prm``  (see `#7`_).

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters(filename = 'example.i3d')
        >>> prm.load()

        or just:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile = 'example.i3d')
        >>> prm = xcompact3d_toolbox.Parameters(loadfile = 'incompact3d.prm')

        .. _#7:
            https://github.com/fschuch/xcompact3d_toolbox/issues/7

        """

        if self.filename.split(".")[-1] == "i3d":
            dictionary = {}

            # unpacking the nested dictionary
            for key_out, value_out in i3d_to_dict(self.filename).items():
                for key_in, value_in in value_out.items():
                    dictionary[key_in] = value_in

        elif self.filename.split(".")[-1] == "prm":
            dictionary = prm_to_dict(self.filename)

        else:
            raise IOError(
                f"{self.filename} is invalid. Supported formats are .i3d and .prm."
            )

        self.set(**dictionary)

    def read_field(self, **kwargs):
        """This method reads a binary field from Xcompact3d with :obj:`numpy.fromfile`
        and wraps it into a :obj:`xarray.DataArray` with the appropriate dimensions,
        coordinates and attributes.

        Data type is defined by :obj:`xcompact3d_toolbox.param["mytype"]`.

        Parameters
        ----------
        filename : str
            Name of the file to be read.
        coords : dict of array_like objects or None, optional
            Coordinates (tick labels) to use for indexing along each dimension (see
            :obj:`xarray.DataArray`). If dims=None (default), coordinates are inferred
            from the folder structure.
        name : str, optional
            Name of this array. If name is empty (default), it is inferred
            from filename.
        attrs : dict_like, optional
            Attributes to assign to the new instance. Boundary conditions are
            automatically included in this method, for derivatives routines.

        Returns
        -------
        :obj:`xarray.DataArray`
            Data array containing values read from the disc. Attributes include
            the proper boundary conditions for derivatives if the
            file prefix is ``ux``, ``uy``, ``uz``, ``phi`` or ``pp``.

        Examples
        -------

        >>> prm = x3d.Parameters()

        >>> xcompact3d_toolbox.param["mytype"] = np.float64 # if x3d was compiled with `-DDOUBLE_PREC`
        >>> xcompact3d_toolbox.param["mytype"] = np.float32 # otherwise

        In the following cases, coordinates and dimensions are infered from the
        folder containing the file:

        >>> uy = prm.read_field('./data/xy_planes/uy-00000400.bin')
        >>> uz = prm.read_field('./data/xz_planes/uz-00000400.bin')
        >>> ux = prm.read_field('./data/3d_snapshots/ux-00000400.bin')

        It is possible to handle 3D arrays with the filenames from previous X3d's
        versions:

        >>> ux = x3d.read_field('./data/ux0010')

        If it is a plane and not included in the folder structure presented above,
        just delete the extra coordinate from the dictionary
        returned by :obj:`xcompact3d_toolbox.parameters.Parameters.get_mesh` and
        inform it as an argument. For example, to read a xy-plane:

        >>> mesh = prm.get_mesh
        >>> del mesh['z']
        >>> ux = x3d.read_field('./data/uy0010', coords = prm.get_mesh)

        Notes
        ----

        Take a look at xarray_'s documentation, specially, see `Why xarray?`_.
        :obj:`xarray.DataArray` includes many useful methods for indexing,
        comparisons, reshaping and reorganizing, computations and plotting.

        .. _xarray: http://xarray.pydata.org/en/stable/
        .. _`Why xarray?`: http://xarray.pydata.org/en/stable/why-xarray.html
        """

        return read_field(self, **kwargs)

    def read_temporal_series(self, **kwargs):
        """Reads all files matching the ``filename_pattern`` with
        :obj:`xcompact3d_toolbox.parameters.Parameters.read_field` and
        concatenates them into a time series.

        .. note:: Make sure to have enough memory to load all files at the same time.

        Parameters
        ----------
        filename_pattern : str
            A specified pattern according to the rules used by the Unix shell.
        steep : str
            The variable at the parameters class that controls how many time steps
            are computed between snapshots. Default is ``ioutput``. Only useful
            if ``filenamedigits = 1``.
        progress_function : str
            Activates a progress bar with the options ``tqdm`` or ``notebook``,
            or turn it off when the string is empty (default).
        **kwargs :
            Arguments to be send to :obj:`xcompact3d_toolbox.parameters.Parameters.read_field`,
            like ``coords``, ``name`` and ``attrs``.

        Returns
        -------
        :obj:`xarray.DataArray`
            Data array containing values read from the disc.

        Examples
        -------

        >>> prm = x3d.Parameters()

        >>> x3d.param["mytype"] = np.float64 # if x3d was compiled with `-DDOUBLE_PREC`
        >>> x3d.param["mytype"] = np.float32 # otherwise

        In the following cases, coordinates and dimensions are infered from the
        folder containing the file and time from the filenames:

        >>> ux = prm.read_all_fields('./data/3d_snapshots/ux-*.bin')
        >>> uy = prm.read_all_fields('./data/xy_planes/uy-*.bin')
        >>> uz = prm.read_all_fields('./data/xz_planes/uz-0000??00.bin')

        It is possible to handle 3D arrays with the filenames from previous X3d's
        versions:

        >>> ux = prm.read_all_fields('./data/ux????')

        If it is a plane and not included in the folder structure presented above,
        just delete the extra coordinate from the dictionary
        returned by :obj:`xcompact3d_toolbox.parameters.Parameters.get_mesh` and
        inform it as an argument. For example, to read a xy-plane:

        >>> mesh = prm.get_mesh
        >>> del mesh['z']
        >>> ux = x3d.read_all_fields('./data/uy????', coords = prm.get_mesh)

        """

        return read_temporal_series(self, **kwargs)

    def write(self):
        """Writes all valid attributes to an ``.i3d`` file.

        An attribute is considered valid if it has a ``tag`` named ``group``,
        witch assigns it to the respective namespace at the ``.i3d`` file.

        Examples
        --------

        >>> prm = xcompact3d_toolbox.Parameters(
        ...     filename = 'example.i3d'
        ...     nx = 101,
        ...     ny = 65,
        ...     nz = 11,
        ...     # end so on...
        ... )
        >>> prm.write()

        """
        if self.filename.split(".")[-1] == "i3d":
            with open(self.filename, "w", encoding="utf-8") as file:
                file.write(self.__str__())
        else:
            raise IOError("Format error, only .i3d is supported")

    def write_xdmf(self, **kwargs):
        """Writes four xdmf files:

        * ``./data/3d_snapshots.xdmf`` for 3D snapshots in ``./data/3d_snapshots/*``;
        * ``./data/xy_planes.xdmf`` for planes in ``./data/xy_planes/*``;
        * ``./data/xz_planes.xdmf`` for planes in ``./data/xz_planes/*``;
        * ``./data/yz_planes.xdmf`` for planes in ``./data/yz_planes/*``.

        Shape and time are inferted from folder structure and filenames.
        File list is obtained automatically with :obj:`glob`.

        .. note:: For now, this is only compatible with the new filename structure,
            the conversion is exemplified in `convert_filenames_x3d_toolbox`_.

        .. _`convert_filenames_x3d_toolbox`: https://gist.github.com/fschuch/5a05b8d6e9787d76655ecf25760e7289

        Examples
        -------

        >>> prm = x3d.Parameters()
        >>> prm.write_xdmf()

        """
        write_xdmf(self, **kwargs)

    def get_mesh(self, refined_for_ibm=False):
        """Get mesh point locations for the three coordinates. They are stored
        in a dictionary. It supports mesh refinement in **y** when
        :obj:`istret` :math:`\\ne` 0.

        Returns
        -------
        :obj:`dict` of :obj:`numpy.ndarray`
            It contains the mesh point locations at three dictionary keys,
            for **x**, **y** and **z**.

        Examples
        --------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> prm.get_mesh()
        {'x': array([0.    , 0.0625, 0.125 , 0.1875, 0.25  , 0.3125, 0.375 , 0.4375,
                0.5   , 0.5625, 0.625 , 0.6875, 0.75  , 0.8125, 0.875 , 0.9375,
                1.    ]),
         'y': array([0.    , 0.0625, 0.125 , 0.1875, 0.25  , 0.3125, 0.375 , 0.4375,
                0.5   , 0.5625, 0.625 , 0.6875, 0.75  , 0.8125, 0.875 , 0.9375,
                1.    ]),
         'z': array([0.    , 0.0625, 0.125 , 0.1875, 0.25  , 0.3125, 0.375 , 0.4375,
                0.5   , 0.5625, 0.625 , 0.6875, 0.75  , 0.8125, 0.875 , 0.9375,
                1.    ])}
        """
        if refined_for_ibm and self.iibm != 0:
            copy = self.mesh.copy()
            for dim in copy.trait_names():
                new_grid_size = getattr(self.mesh, dim)._sub_grid_size * self.nraf
                if not getattr(self.mesh, dim).is_periodic:
                    new_grid_size += 1
                getattr(copy, dim).set(grid_size = new_grid_size)
            return copy.get()
        return self.mesh.get()
