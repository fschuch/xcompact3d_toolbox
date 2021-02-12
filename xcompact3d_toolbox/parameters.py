# -*- coding: utf-8 -*-
"""
Manipulate physical and computational parameters. It contains variables and
methods designed to be a bridge between Xcompact3d and Python applications for
pre and post-processing.
"""

import numpy as np
import math
import glob
import traitlets
import os.path
import xarray as xr
import warnings
from .param import boundary_condition, param
from .mesh import get_mesh
from .io import i3d_to_dict, dict_to_i3d, prm_to_dict, write_xdmf

possible_mesh = [
    9,
    11,
    13,
    17,
    19,
    21,
    25,
    31,
    33,
    37,
    41,
    49,
    51,
    55,
    61,
    65,
    73,
    81,
    91,
    97,
    101,
    109,
    121,
    129,
    145,
    151,
    161,
    163,
    181,
    193,
    201,
    217,
    241,
    251,
    257,
    271,
    289,
    301,
    321,
    325,
    361,
    385,
    401,
    433,
    451,
    481,
    487,
    501,
    513,
    541,
    577,
    601,
    641,
    649,
    721,
    751,
    769,
    801,
    811,
    865,
    901,
    961,
    973,
    1001,
    1025,
    1081,
    1153,
    1201,
    1251,
    1281,
    1297,
    1351,
    1441,
    1459,
    1501,
    1537,
    1601,
    1621,
    1729,
    1801,
    1921,
    1945,
    2001,
    2049,
    2161,
    2251,
    2305,
    2401,
    2431,
    2501,
    2561,
    2593,
    2701,
    2881,
    2917,
    3001,
    3073,
    3201,
    3241,
    3457,
    3601,
    3751,
    3841,
    3889,
    4001,
    4051,
    4097,
    4321,
    4375,
    4501,
    4609,
    4801,
    4861,
    5001,
    5121,
    5185,
    5401,
    5761,
    5833,
    6001,
    6145,
    6251,
    6401,
    6481,
    6751,
    6913,
    7201,
    7291,
    7501,
    7681,
    7777,
    8001,
    8101,
    8193,
    8641,
    8749,
    9001,
]
""":obj:`list` of :obj:`int`: Possible number of mesh points for no periodic boundary conditions.

Due to restrictions at the FFT library, they must be equal to:

.. math::
    n_i = 2^{1+a} \\times 3^b \\times 5^c + 1,

where :math:`a`, :math:`b` and :math:`c` are non negative integers, and :math:`i`
representes the three coordinates (**x**, **y** and **z**).

Aditionally, the derivative's stencil imposes that :math:`n_i \\ge 9`.

Notes
-----
There is no upper limit, as long as the restrictions are satisfied.
"""

possible_mesh_p = [i - 1 for i in possible_mesh]
""":obj:`list` of :obj:`int`: Possible number of mesh points for periodic boundary conditions.

Due to restrictions at the FFT library, they must be equal to:

.. math::
    n_i = 2^{1+a} \\times 3^b \\times 5^c,

where :math:`a`, :math:`b` and :math:`c` are non negative integers, and :math:`i`
representes the three coordinates (**x**, **y** and **z**).

Aditionally, the derivative's stencil imposes that :math:`n_i \\ge 8`.

Notes
-----
There is no upper limit, as long as the restrictions are satisfied.
"""


def divisorGenerator(n):
    """Yelds the possibles divisors for ``n``.

    Especially useful to compute the possible values for :obj:`p_row` and :obj:`p_col`
    as functions of the number of computational cores available (:obj:`ncores`).
    Zero is also included in the case of auto-tunning, i.e., ``p_row=p_col=0``.

    Parameters
    ----------
    n : int
        Input value.

    Yields
    ------
    int
        The next possible divisor for ``n``.

    Examples
    -------

    >>> print(list(divisorGenerator(8)))
    [0, 1, 2, 4, 8]

    """
    large_divisors = [0]
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield int(divisor)


class Parameters(traitlets.HasTraits):
    """The physical and computational parameters are built on top of `traitlets`_.
    It is a framework that lets Python classes have attributes with type checking,
    dynamically calculated default values, and ‘on change’ callbacks.
    So, many of the parameters are validated regarding type, norms, and values
    supported by Xcompact3d.

    There are methods to handle the parameters file, to read the binary
    arrays produced by Xcompact3d and also to write the xdmf file, so the binary
    fields can be open in any external visualization tool.

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

    #
    # # BasicParam
    #

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
        group="BasicParam", desc="Defines pertubation at initial condition"
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
        traitlets.Int(default_value=17, min=0).tag(
            group="BasicParam", desc=f"{name.upper()}-direction nodes"
        )
        for name in "x y z".split()
    ]
    """int: Number of mesh points.

    Notes
    -----
        See :obj:`possible_mesh` and :obj:`possible_mesh_p`.
    """

    xlx, yly, zlz = [
        traitlets.Float(default_value=1.0, min=0).tag(
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

    #
    # # NumOptions
    #

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
        desc="Ratio between hypervisvosity at km=2/3π and kc=π (dissipation factor range)",
    )
    """float: Ratio between hypervisvosity at :math:`k_m=2/3\\pi` and :math:`k_c= \\pi`.
    """

    #
    # # InOutParam
    #

    irestart = traitlets.Int(default_value=0, min=0, max=1).tag(
        group="InOutParam", desc="Read initial flow field (0: No, 1: Yes)"
    )
    """int: Reads initial flow field if equals to 1.
    """

    nvisu = traitlets.Int(default_value=1, min=1).tag(
        group="InOutParam", desc="Size for visualisation collection"
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
    #
    # # ScalarParam
    #

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

    #
    # # LESModel
    #

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
    #
    # # ibmstuff
    #

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

    # Auxiliar
    filename = traitlets.Unicode(default_value="input.i3d").tag()
    """str: Filename for the ``.i3d`` file.
    """

    _mx, _my, _mz = [traitlets.Int(default_value=1, min=1) for i in range(3)]

    dx, dy, dz = [
        traitlets.Float(default_value=0.0625, min=0.0).tag() for dir in "x y z".split()
    ]
    """float: Mesh resolution.
    """

    _nclx, _ncly, _nclz = [traitlets.Bool() for i in range(3)]
    """bool: Auxiliar variable for boundary condition,
        it is :obj:`True` if Periodic and :obj:`False` otherwise.
    """

    _possible_mesh_x, _possible_mesh_y, _possible_mesh_z = [
        traitlets.List(trait=traitlets.Int(), default_value=possible_mesh)
        for i in range(3)
    ]
    """:obj:`list` of :obj:`int`: Auxiliar variable for mesh points widgets,
        it stores the avalilable options according to the boudary conditions.
    """

    ncores = traitlets.Int(default_value=4, min=1).tag()
    """int: Number of computational cores where Xcompact3d will run.
    """

    _possible_p_row, _possible_p_col = [
        traitlets.List(trait=traitlets.Int(), default_value=list(divisorGenerator(4)))
        for i in range(2)
    ]
    """:obj:`list` of :obj:`int`: Auxiliar variable for parallel domain decomposition,
        it stores the avalilable options according to :obj:`ncores`.
    """

    # cfl = traitlets.Float(0.0)
    size = traitlets.Unicode().tag()
    """str: Auxiliar variable indicating the demanded space in disc
    """

    def __init__(self, **kwargs):
        """Initializes the Parameters Class.

        Parameters
        ----------
        **kwargs
            Keyword arguments for valid atributes.

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid atribute.

        Examples
        -------

        There are a few ways to initialize the class.

        First, calling it with no
        arguments initializes all variables with default value:

        >>> prm = xcompact3d_toolbox.Parameters()

        It is possible to set any values afterwards (including new atributes):

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

        # Boundary conditions are high priority in order to avoid bugs
        for bc in "nclx1 nclxn ncly1 nclyn nclz1 nclzn".split():
            if bc in kwargs:
                setattr(self, bc, kwargs[bc])

        if "loadfile" in kwargs.keys():
            self.filename = kwargs["loadfile"]
            self.load()
            del kwargs["loadfile"]

        for key, arg in kwargs.items():
            if key not in self.trait_names():
                warnings.warn(f"{key} is not a valid parameter and was not loaded")
            setattr(self, key, arg)

    def __repr__(self):
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

    @traitlets.validate("nx")
    def _validade_mesh_nx(self, proposal):
        _validate_mesh(proposal["value"], self._nclx, self.nclx1, self.nclxn, "x")
        return proposal["value"]

    @traitlets.validate("ny")
    def _validade_mesh_ny(self, proposal):
        _validate_mesh(proposal["value"], self._ncly, self.ncly1, self.nclyn, "y")
        return proposal["value"]

    @traitlets.validate("nz")
    def _validade_mesh_nz(self, proposal):
        _validate_mesh(proposal["value"], self._nclz, self.nclz1, self.nclzn, "z")
        return proposal["value"]

    @traitlets.validate("ifilenameformat")
    def _validade_ifilenameformat(self, proposal):

        if proposal["value"][:2] == "(I" and proposal["value"][-1] == ")":
            i1, i2 = proposal["value"][2:-1].split(".")
            if i1 == i2:
                return proposal["value"]

        raise traitlets.TraitError(
            f"Invalid value for ifilenameformat, try with something like (I3.3) or (I9.9)"
        )

    @traitlets.observe("dx", "nx", "xlx", "dy", "ny", "yly", "dz", "nz", "zlz")
    def _observe_resolution(self, change):
        # for name in "name new old".split():
        #     print(f"    {name:>5} : {change[name]}")

        dim = change["name"][-1]  # It will be x, y or z
        #
        if change["name"] == f"n{dim}":
            if getattr(self, f"_ncl{dim}"):
                setattr(self, f"_m{dim}", change["new"])
            else:
                setattr(self, f"_m{dim}", change["new"] - 1)
            setattr(
                self,
                f"d{dim}",
                getattr(self, f"{dim}l{dim}") / getattr(self, f"_m{dim}"),
            )
        if change["name"] == f"d{dim}":
            new_l = change["new"] * getattr(self, f"_m{dim}")
            if new_l != getattr(self, f"{dim}l{dim}"):
                setattr(self, f"{dim}l{dim}", new_l)
        if change["name"] == f"{dim}l{dim}":
            new_d = change["new"] / getattr(self, f"_m{dim}")
            if new_d != getattr(self, f"d{dim}"):
                setattr(self, f"d{dim}", new_d)

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
            for i in f"ncl{dim}1 ncl{dim}n ncl{dim}S1 ncl{dim}Sn".split():
                setattr(self, i, 0)
            setattr(self, f"_ncl{dim}", True)
        if change["old"] == 0 and change["new"] != 0:
            for i in f"ncl{dim}1 ncl{dim}n ncl{dim}S1 ncl{dim}Sn".split():
                setattr(self, i, change["new"])
            setattr(self, f"_ncl{dim}", False)

    @traitlets.observe("_nclx", "_ncly", "_nclz")
    def _observe_periodicity(self, change):
        #
        dim = change["name"][-1]  # It will be x, y or z
        #
        if change["new"]:
            tmp = getattr(self, f"n{dim}") - 1
            setattr(self, f"_possible_mesh_{dim}", possible_mesh_p)
            setattr(self, f"n{dim}", tmp)
        else:
            tmp = getattr(self, f"n{dim}") + 1
            setattr(self, f"_possible_mesh_{dim}", possible_mesh)
            setattr(self, f"n{dim}", tmp)

    @traitlets.observe("p_row", "p_col", "ncores")
    def _observe_2Decomp(self, change):
        if change["name"] == "ncores":
            possible = list(divisorGenerator(change["new"]))
            self._possible_p_row = possible
            self._possible_p_col = possible
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
            # self.trait_metadata("nu0nu", "widget").disabled = True
            # self.trait_metadata("cnu", "widget").disabled = True
            # self.trait_metadata("isecondder", "widget").disabled = True
        # else:
        # self.trait_metadata("nu0nu", "widget").disabled = False
        # self.trait_metadata("cnu", "widget").disabled = False
        # self.trait_metadata("isecondder", "widget").disabled = False

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

        # Boundary conditions are high priority in order to avoid bugs
        for bc in "nclx1 nclxn ncly1 nclyn nclz1 nclzn".split():
            if bc in dictionary:
                setattr(self, bc, dictionary[bc])

        for key, value in dictionary.items():
            try:
                if self.trait_metadata(key, "group") is not None:
                    setattr(self, key, dictionary[key])
            except:
                warnings.warn(f"{key} is not a valid parameter and was not loaded")

    def read(self):
        """Read is deprecated, use :obj:`xcompact3d_toolbox.parameters.Parameters.load`."""
        warnings.warn("read is deprecated, use load", DeprecationWarning)
        self.load()

    def read_field(self, filename, coords=None, name="", attrs={}):
        """This method reads a binary field from Xcompact3d with :obj:`numpy.fromfile`
        and wraps it into a :obj:`xarray.DataArray` with the appropriate dimensions,
        coordinates and attributes.

        The properties can be inferted automatically if the
        file is inside Xcompact3d's output folders structure, i.e.:

        * 3d_snapshots (nx, ny, nz);
        * xy_planes (nx, ny);
        * xz_planes (nx, nz);
        * yz_planes (ny, nz).

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

        path, file = os.path.split(filename)
        path = os.path.basename(path)

        # If coords is None, we assume a 3d field, and then we cut one coordinate
        # for planes if necessary
        if coords is None:
            coords = self.get_mesh
            if path == "3d_snapshots":
                pass
            elif path == "xy_planes":
                del coords["z"]
            elif path == "xz_planes":
                del coords["y"]
            elif path == "yz_planes":
                del coords["x"]

        # if name is empty, we obtain it from the filename
        if not name:
            try:
                if "-" in file:
                    name = os.path.basename(file.replace(".bin", "")).split("-")[0]
                else:
                    i1, i2 = self.ifilenameformat[2:-1].split(".")
                    name = os.path.basename(file.replace(".bin", ""))[0:-i1]
            except:
                warnings.warn(
                    "Impossible to obtain array name automatically, try to set it manually."
                )

        # Include atributes for boundary conditions, useful to compute the derivatives
        if "phi" in name:
            attrs["BC"] = self.get_boundary_condition("phi")
        else:
            attrs["BC"] = self.get_boundary_condition(name)

        # We obtain the shape for np.fromfile from the coordinates
        shape = []
        for key, value in coords.items():
            shape.append(value.size)

        # This is necessary if the file is a link
        if os.path.islink(filename):
            from os import readlink

            filename = readlink(filename)

        # Finally, we read the array and wrap it into a xarray dataset
        return xr.DataArray(
            np.fromfile(filename, dtype=param["mytype"]).reshape(shape, order="F"),
            dims=coords.keys(),
            coords=coords,
            name=name,
            attrs=attrs,
        )

    def read_all_fields(
        self, filename_pattern, steep="ioutput", progress_function="", **kargs
    ):
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
        filenames = sorted(glob.glob(filename_pattern))

        # is empty
        if not filenames:
            raise IOError(f"No file was found corresponding to {filename_pattern}.")

        # New filename format, see https://github.com/fschuch/Xcompact3d/issues/3
        if self.filenamedigits == 0:
            dt = self.dt
        # Previous filename format
        elif self.filenamedigits == 1:
            dt = self.dt * getattr(self, steep)

        i1, i2 = self.ifilenameformat[2:-1].split(".")
        t = dt * np.array(
            [
                param["mytype"](os.path.basename(file).replace(".bin", "")[-int(i1) :])
                for file in filenames
            ],
            dtype=param["mytype"],
        )

        # <To do> Maybe turn this in a global parameter, because it is useful in
        # many methods
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

        return xr.concat(
            [
                self.read_field(file, **kargs)
                for file in progress_function(filenames, desc=filename_pattern)
            ],
            dim="t",
        ).assign_coords(coords={"t": t})

    def write(self):
        """Writes all valid attributes to an ``.i3d`` file.

        An attribute is considered valid if it has a ``tag`` named ``group``,
        witch assigns it to the respective namespace at the ``.i3d`` file.

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters(filename = 'example.i3d'
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

    def write_xdmf(self):
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
        write_xdmf(self)

    @property
    def get_mesh(self):
        """Get mesh point locations for the three coordinates. They are stored
        in a dictionary. It supports mesh refinement in **y** when
        :obj:`istret` :math:`\\ne` 0.

        Returns
        -------
        :obj:`dict` of :obj:`numpy.ndarray`
            It contains the mesh point locations at three dictionary keys,
            for **x**, **y** and **z**.

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> prm.get_mesh
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
        return get_mesh(self)


def _validate_mesh(n, ncl, ncl1, ncln, dim):

    pmin = 8 if ncl else 9

    if n < pmin:
        # Because of the derivatives' stencil
        raise traitlets.TraitError(f"{n} is invalid, n{dim} must be larger than {pmin}")

    if not ncl:
        n = n - 1

    if n % 2 == 0:
        n //= 2

        for val in [2, 3, 5]:
            while True:
                if n % val == 0:
                    n //= val
                else:
                    break

    if n != 1:
        # Because of the FFT library
        raise traitlets.TraitError(f"Invalid value for mesh points (n{dim})")
