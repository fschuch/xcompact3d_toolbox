"""
Tools to manipulate the physical and computational parameters. It contains variables and
methods designed to be a link between XCompact3d_ and Python applications for
pre and post-processing.

.. _XCompact3d:
    https://github.com/xcompact3d/Incompact3d
"""

from __future__ import annotations

import os.path

import numpy as np
import traitlets
from loguru import logger

from xcompact3d_toolbox.io import Dataset, i3d_to_dict, prm_to_dict
from xcompact3d_toolbox.mesh import Istret, Mesh3D
from xcompact3d_toolbox.param import COORDS, boundary_condition, param


class ParametersBasicParam(traitlets.HasTraits):
    p_row, p_col = (
        traitlets.Int(default_value=0, min=0).tag(
            group="BasicParam",
            desc=f"{name} for domain decomposition and parallel computation",
        )
        for name in ["Row partition", "Column partition"]
    )
    """int: Defines the domain decomposition for (large-scale) parallel computation.

    Notes
    -----
        The product ``p_row * p_col`` must be equal to the number of
        computational cores where XCompact3d will run.
        More information can be found at `2DECOMP&FFT`_.

        ``p_row = p_col = 0`` activates auto-tunning.

    .. _2DECOMP&FFT:
        http://www.2decomp.org
    """

    itype = traitlets.Int(default_value=12, min=0, max=12).tag(
        group="BasicParam",
        desc="Flow configuration (1:Lock-exchange, 2:TGV, 3:Channel, and others)",
    )
    """int: Sets the flow configuration, each one is specified in a different
    ``BC.<flow-configuration>.f90`` file (see `Xcompact3d/src`_), they are:

    * 0 - User configuration;
    * 1 - Turbidity Current in Lock-Release;
    * 2 - Taylor-Green Vortex;
    * 3 - Periodic Turbulent Channel;
    * 4 - Periodic Hill
    * 5 - Flow around a Cylinder;
    * 6 - Debug Schemes (for developers);
    * 7 - Mixing Layer;
    * 8 - Jet;
    * 9 - Turbulent Boundary Layer;
    * 10 - ABL;
    * 11 - Uniform;
    * 12 - `Sandbox`_.

    .. _Xcompact3d/src:
        https://github.com/xcompact3d/Incompact3d/tree/master/src
    """

    iin = traitlets.Int(default_value=0, min=0, max=3).tag(
        group="BasicParam", desc="Defines perturbation at initial condition"
    )
    """int: Defines perturbation at the initial condition:

    * 0 - No random noise (default);
    * 1 - Random noise with amplitude of :obj:`init_noise`;
    * 2 - Random noise with fixed seed
      (important for reproducibility, development and debugging)
      and amplitude of :obj:`init_noise`;
    * 3 - Read inflow planes.

    Notes
    -----
        The exactly behavior may be different according to each flow configuration.
    """

    nx, ny, nz = (traitlets.Int().tag(group="BasicParam", desc=f"{name.upper()}-direction nodes") for name in COORDS)
    """int: Number of mesh points.

    Notes
    -----
        See :obj:`xcompact3d_toolbox.mesh.Coordinate.possible_grid_size`
        for recommended grid sizes.
    """

    xlx, yly, zlz = (
        traitlets.Float().tag(group="BasicParam", desc=f"Size of the box in {name}-direction") for name in COORDS
    )
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
    """int: Enables store snapshots at a frequency :obj:`ParametersInOutParam.ioutput`:

        * 0 - No;
        * 1 - Yes (default).
    """

    ipost = traitlets.Int(default_value=0, min=0, max=1).tag(
        group="BasicParam",
        desc="Enables online postprocessing at a frequency iprocessing (0: No, 1: Yes)",
    )
    """int: Enables online postprocessing at a frequency :obj:`ParametersInOutParam.iprocessing`:

        * 0 - No;
        * 1 - Yes (default).

      .. note:: The computation for each case is specified at the ``BC.<flow-configuration>.f90`` file.
    """

    ilesmod = traitlets.Int(default_value=0, min=0, max=1).tag(
        group="BasicParam", desc="Enables Large-Eddy methodologies (0: No, 1: Yes)"
    )
    """int: Enables Large-Eddy methodologies:

        * 0 - No (also forces :obj:`ParametersNumOptions.nu0nu` and :obj:`ParametersNumOptions.cnu`
          to 4.0 and 0.44, respectively);
        * 1 - Yes (also activates the namespace **LESModel** (see :obj:`ParametersLESModel`).
    """

    istret = traitlets.UseEnum(Istret, default_value=0).tag(
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

    beta = traitlets.Float(default_value=1.0, min=0).tag(group="BasicParam", desc="Refinement parameter")
    """float: Refinement factor in **y**.

    Notes
    -----
        Only necessary if :obj:`istret` :math:`\\ne` 0.
    """

    dt = traitlets.Float(default_value=1e-3, min=0.0).tag(group="BasicParam", desc="Time step")
    """float: Time step :math:`(\\Delta t)`.
    """

    ifirst = traitlets.Int(default_value=0, min=0).tag(group="BasicParam", desc="The number for the first iteration")
    """int: The number for the first iteration.
    """

    ilast = traitlets.Int(default_value=0, min=0).tag(group="BasicParam", desc="The number for the last iteration")
    """int: The number for the last iteration.
    """

    re = traitlets.Float(default_value=1e3).tag(group="BasicParam", desc="Reynolds number")
    """float: Reynolds number :math:`(Re)`.
    """

    init_noise = traitlets.Float(default_value=0.0).tag(
        group="BasicParam", desc="Turbulence intensity (1=100%) !! Initial condition"
    )
    """float: Random number amplitude at initial condition.

    Notes
    -----
        Only necessary if :obj:`iin` :math:`\\ne` 0.
        The exactly behavior may be different according to each flow configuration.

    """

    inflow_noise = traitlets.Float(default_value=0.0).tag(
        group="BasicParam", desc="Turbulence intensity (1=100%) !! Inflow condition"
    )
    """float: Random number amplitude at inflow boundary (where :math:`x=0`).

    Notes
    -----
        Only necessary if :obj:`nclx1` is equal to 2.
    """

    iibm = traitlets.Int(default_value=0, min=0, max=3).tag(
        group="BasicParam", desc="Flag for immersed boundary method (0: No, 1: Yes)"
    )
    """int: Enables Immersed Boundary Method (IBM):

    * 0 - Off (default);
    * 1 - On with direct forcing method, i.e.,
      it sets velocity to zero inside the solid body;
    * 2 - On with alternating forcing method, i.e, it uses
      Lagrangian Interpolators to define the velocity inside the body
      and imposes no-slip condition at the solid/fluid interface;
    * 3 - Cubic Spline Reconstruction;

    Any option greater than zero activates the namespace **ibmstuff**, for variables like
    :obj:`ParametersIbmStuff.nobjmax` and :obj:`ParametersIbmStuff.nraf`.
    """

    numscalar = traitlets.Int(default_value=0, min=0, max=9).tag(group="BasicParam", desc="Number of scalar fractions")
    """int: Number of scalar fraction, which can have different properties.
    Any option greater than zero activates the namespace :obj:`ParametersScalarParam`.

    """

    gravx, gravy, gravz = (
        traitlets.Float(default_value=0.0).tag(group="BasicParam", desc=f"Gravity unitary vector in {name}-direction")
        for name in COORDS
    )
    """float: Component of the unitary vector pointing in the gravity's direction.
    """

    iscalar = traitlets.Int(default_value=0, min=0, max=1).tag(group="BasicParam")

    ilmn = traitlets.Bool(default_value=False).tag(group="BasicParam")

    u1 = traitlets.Float(default_value=2.0).tag(group="BasicParam")

    u2 = traitlets.Float(default_value=1.0).tag(group="BasicParam")

    ifilter = traitlets.Int(default_value=0, min=0, max=3).tag(group="BasicParam")

    C_filter = traitlets.Float(default_value=0.49).tag(group="BasicParam")

    iturbine = traitlets.Int(default_value=0, min=0, max=2).tag(group="BasicParam")

    def __init__(self):
        super().__init__()

    @traitlets.validate("iscalar")
    def _validate_iscalar(self, proposal):
        if proposal.get("value") == 0 and self.numscalar > 0:
            msg = "iscalar can not be zero if numscalar > 0"
            raise traitlets.TraitError(msg)
        return proposal.get("value")

    @traitlets.observe("numscalar")
    def _observe_numscalar(self, change):
        if change.get("new") == 0:
            self.iscalar = 0
        else:
            self.iscalar = 1


class ParametersNumOptions(traitlets.HasTraits):
    ifirstder = traitlets.Int(default_value=4, min=1, max=4).tag(group="NumOptions")
    """int: Scheme for first order derivative:

    * 1 - 2nd central;
    * 2 - 4th central;
    * 3 - 4th compact;
    * 4 - 6th compact (default).
    """

    isecondder = traitlets.Int(default_value=4, min=1, max=5).tag(
        group="NumOptions", desc="Scheme for second order derivative"
    )
    """int: Scheme for second order derivative:

    * 1 - 2nd central;
    * 2 - 4th central;
    * 3 - 4th compact;
    * 4 - 6th compact (default);
    * 5 - Hyperviscous 6th.
    """

    ipinter = traitlets.Int(3)

    itimescheme = traitlets.Int(default_value=3, min=1, max=6).tag(
        group="NumOptions",
        desc="Time integration scheme (1: Euler, 2: AB2, 3: AB3, 5: RK3)",
    )
    """int: Time integration scheme:

    * 1 - Forwards Euler;
    * 2 - Adams-bashforth 2;
    * 3 - Adams-bashforth 3 (default);
    * 4 - Adams-bashforth 4 (Not Implemented);
    * 5 - Runge-kutta 3;
    * 6 - Runge-kutta 4 (Not Implemented).
    """

    iimplicit = traitlets.Int(default_value=0, min=0, max=2).tag(group="NumOptions")
    """int: Time integration scheme:

    * 0 - Off (default);
    * 1 - With backward Euler for Y diffusion;
    * 2 - With CN for Y diffusion.
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
        super().__init__()

    @traitlets.observe("ilesmod")
    def _observe_ilesmod(self, change):
        if change["new"] == 0:
            # It is coded at xcompact3d, look at parameters.f90
            self.nu0nu, self.cnu = 4.0, 0.44

    @traitlets.validate("nu0nu", "cnu")
    def _validate_iscalar(self, proposal):
        if self.ilesmod == 0:
            # It is coded at xcompact3d, look at parameters.f90
            msg = "Can not set new values for nu0nu and cnu if ilesmod = 0"
            raise traitlets.TraitError(msg)
        return proposal.get("value")


class ParametersInOutParam(traitlets.HasTraits):
    irestart = traitlets.Int(default_value=0, min=0, max=1).tag(
        group="InOutParam", desc="Read initial flow field (0: No, 1: Yes)"
    )
    """int: Reads initial flow field if equals to 1.
    """

    nvisu = traitlets.Int(default_value=1, min=1).tag(group="InOutParam", desc="Size for visualization collection")
    """int: Size for visual collection.
    """

    icheckpoint = traitlets.Int(default_value=1000, min=1).tag(
        group="InOutParam", desc="Frequency for writing backup file"
    )
    """int: Frequency for writing restart file.
    """

    ioutput = traitlets.Int(default_value=1000, min=1).tag(group="InOutParam", desc="Frequency for visualization file")
    """int: Frequency for visualization (3D snapshots).
    """

    iprocessing = traitlets.Int(default_value=1000, min=1).tag(
        group="InOutParam", desc="Frequency for online postprocessing"
    )
    """int: Frequency for online postprocessing.
    """

    ioutflow = traitlets.Int(default_value=0).tag(group="InOutParam")

    ninflows = traitlets.Int(default_value=1, min=1).tag(group="InOutParam")

    ntimesteps = traitlets.Int(default_value=1, min=1).tag(group="InOutParam")

    inflowpath = traitlets.Unicode(default_value="./")

    output2D = traitlets.Int(default_value=0).tag(group="InOutParam")  # N815

    nprobes = traitlets.Int(default_value=0, min=0).tag(group="InOutParam")

    def __init__(self):
        super().__init__()


class ParametersScalarParam(traitlets.HasTraits):
    sc = traitlets.List(trait=traitlets.Float()).tag(group="ScalarParam", desc="Schmidt number(s)")
    """:obj:`list` of :obj:`float`: Schmidt number(s).
    """

    ri = traitlets.List(trait=traitlets.Float()).tag(group="ScalarParam", desc="Richardson number(s)")
    """:obj:`list` of :obj:`float`: Richardson number(s).
    """

    uset = traitlets.List(trait=traitlets.Float()).tag(group="ScalarParam", desc="Settling velocity(ies)")
    """:obj:`list` of :obj:`float`: Settling velocity(s).
    """

    cp = traitlets.List(trait=traitlets.Float()).tag(group="ScalarParam", desc="Initial concentration(s)")
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

    sc_even = traitlets.List(trait=traitlets.Bool()).tag(group="ScalarParam")
    sc_skew = traitlets.List(trait=traitlets.Bool()).tag(group="ScalarParam")
    alpha_sc = traitlets.List(trait=traitlets.Float()).tag(group="ScalarParam")
    beta_sc = traitlets.List(trait=traitlets.Float()).tag(group="ScalarParam")
    g_sc = traitlets.List(trait=traitlets.Float()).tag(group="ScalarParam")
    Tref = traitlets.Float().tag(group="ScalarParam")

    def __init__(self):
        super().__init__()


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

    smagcst = traitlets.Float().tag(group="LESModel")
    """float: """

    smagwalldamp = traitlets.Int(default_value=0).tag(group="LESModel")
    """int: """

    nSmag = traitlets.Float(default_value=1.0).tag(group="LESModel")
    """float: """

    walecst = traitlets.Float().tag(group="LESModel")
    """float: """

    maxdsmagcst = traitlets.Float().tag(group="LESModel")
    """float: """

    iwall = traitlets.Int().tag(group="LESModel")
    """int: """

    def __init__(self):
        super().__init__()


class ParametersIbmStuff(traitlets.HasTraits):
    nobjmax = traitlets.Int(default_value=1, min=1).tag(
        group="ibmstuff", desc="Maximum number of objects in any direction"
    )
    """int: Maximum number of objects in any direction. It is defined
        automatically at :obj:`gene_epsi_3d`.
    """

    nraf = traitlets.Int(default_value=10, min=1).tag(
        group="ibmstuff",
        desc="Level of refinement for iibm==2 to find the surface of the immersed object",
    )
    """int: Level of refinement to find the surface of the immersed object, when
        :obj:`ParametersBasicParam.iibm` is equal to 2.
    """

    izap = traitlets.Int(default_value=1, min=0, max=3).tag(
        group="ibmstuff",
        desc="How many points to skip for reconstruction (Range: 0-3) (Recommended: 1)",
    )
    """int: How many points to skip for reconstruction ranging from 0 to 3,
        the recommended is 1.
    """

    npif = traitlets.Int(default_value=2, min=1, max=3).tag(
        group="ibmstuff",
        desc="Number of Points for the Reconstruction (npif=1-3) (Recommended: 2)",
    )
    """int: Number of Points for the reconstruction ranging from 1 to 3,
        the recommended is 2.
    """

    def __init__(self):
        super().__init__()


class ParametersALMParam(traitlets.HasTraits):
    iturboutput = traitlets.Int(default_value=1, min=1).tag(group="ALMParam")

    def __init__(self):
        super().__init__()


class ParametersExtras(traitlets.HasTraits):
    """Extra utilities that are not present at the parameters file,
    but are useful for Python applications.
    """

    filename = traitlets.Unicode(default_value="input.i3d").tag()
    """str: Filename for the parameters file.
    """

    mesh = traitlets.Instance(klass=Mesh3D)
    """:obj:`xcompact3d_toolbox.mesh.Mesh3D`: Mesh object.
    """

    dataset = traitlets.Instance(klass=Dataset)
    """:obj:`xcompact3d_toolbox.io.Dataset`: An object that reads
    and writes the raw binary files from XCompact3d on-demand.

    Notes
    -----

    All arrays are wrapped into Xarray objects (:obj:`xarray.DataArray`
    or :obj:`xarray.Dataset`), take a look at xarray_'s documentation,
    specially, see `Why xarray?`_
    Xarray has many useful methods for indexing, comparisons, reshaping
    and reorganizing, computations and plotting.

    Consider using hvPlot_ to explore your data interactively,
    see how to plot `Gridded Data`_.

    .. _xarray: http://docs.xarray.dev/en/stable
    .. _`Why xarray?`: http://docs.xarray.dev/en/stable/why-xarray.html
    .. _hvPlot: https://hvplot.holoviz.org/
    .. _`Gridded Data`: https://hvplot.holoviz.org/user_guide/Gridded_Data.html

    Examples
    --------

    The first step is specify the filename properties.
    If the simulated fields are named like ``ux-000.bin``, they are in the default
    configuration, there is no need to specify filename properties. But just in case,
    it would be like:

    >>> prm = xcompact3d_toolbox.Parameters()
    >>> prm.dataset.filename_properties.set(
    ...     separator = "-",
    ...     file_extension = ".bin",
    ...     number_of_digits = 3
    ... )

    If the simulated fields are named like ``ux0000``, the parameters are:

    >>> prm = xcompact3d_toolbox.Parameters()
    >>> prm.dataset.filename_properties.set(
    ...     separator = "",
    ...     file_extension = "",
    ...     number_of_digits = 4
    ... )

    Data type is defined by :obj:`xcompact3d_toolbox.param`:

    >>> import numpy
    >>> xcompact3d_toolbox.param["mytype] = numpy.float64 # if double precision
    >>> xcompact3d_toolbox.param["mytype] = numpy.float32 # if single precision

    Now it is possible to customize the way the dataset
    will be handled:

    >>> prm.dataset.set(
    ...     data_path = "./data/",
    ...     drop_coords = "",
    ...     set_of_variables = {"ux", "uy", "uz"},
    ...     snapshot_step = "ioutput",
    ...     snapshot_counting = "ilast",
    ...     stack_scalar = True,
    ...     stack_velocity = False,
    ... )

    .. note :: For convenience, ``data_path`` is set as
       ``"./data/"`` relative to the ``filename`` of the parameters file
       when creating a new instance of :obj:`Parameters`
       (i.g., if ``filename = "./example/input.i3d"`` then
       ``data_path = "./example/data/"``).

    There are many ways to load the arrays produced by
    your numerical simulation, so you can choose what
    best suits your post-processing application.
    See the examples:

    * Load one array from the disc:

      >>> ux = prm.dataset.load_array("ux-0000.bin")

    * Load the entire time series for a given variable:

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

    * Load all variables from a given snapshot:

      >>> snapshot = prm.dataset.load_snapshot(10)

      or just:

      >>> snapshot = prm.dataset[10]

    * Loop through all snapshots, loading them one by one:

      >>> for ds in prm.dataset:
      ...     vort = ds.uy.x3d.first_derivative("x") - ds.ux.x3d.first_derivative("y")
      ...     prm.dataset.write(data = vort, file_prefix = "w3")

    * Loop through some snapshots, loading them one by one, with the same arguments
      of a classic Python :obj:`range`, for instance, from 0 to 100 with a step of 5:

      >>> for ds in prm.dataset(0, 101, 5):
      ...     vort = ds.uy.x3d.first_derivative("x") - ds.ux.x3d.first_derivative("y")
      ...     prm.dataset.write(data = vort, file_prefix = "w3")

    * Or simply load all snapshots at once (if you have enough memory):

      >>> ds = prm.dataset[:]

    And finally, it is possible to produce a new xdmf file, so all data
    can be visualized on any external tool:

    >>> prm.dataset.write_xdmf()
    """

    dx, dy, dz = (traitlets.Float().tag() for _ in COORDS)
    """float: Mesh resolution.
    """

    ncores = traitlets.Int(default_value=4, min=1).tag()
    """int: Number of computational cores where XCompact3d will run.
    """

    size = traitlets.Unicode().tag()
    """str: Auxiliary variable indicating the demand for storage.
    """

    def __init__(self):
        super().__init__()
        self.mesh = Mesh3D()

        self._link_mesh_and_parameters()
        self.dataset = Dataset(_mesh=self.mesh, _prm=self)

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
    ParametersALMParam,
    ParametersExtras,
):
    """The physical and computational parameters are built on top of `traitlets`_.
    It is a framework that lets Python classes have attributes with type checking,
    dynamically calculated default values, and "on change" callbacks.
    In this way, many of the parameters are validated regarding the type,
    business rules, and the range of values supported by XCompact3d_.
    There are methods to handle the parameters file (``.i3d`` and ``.prm``).

    The parameters are arranged in different classes, but just for organization purposes,
    this class inherits from all of them.

    In addition, there are `ipywidgets`_ for a friendly user interface,
    see :obj:`xcompact3d_toolbox.gui.ParametersGui`.

    After that, it is time to read the binary arrays produced by XCompact3d_ and also
    to write a new xdmf file, so the  binary fields can be opened in any external
    visualization tool. See more details in :obj:`xcompact3d_toolbox.parameters.ParametersExtras.dataset`.

    .. _XCompact3d:
        https://github.com/xcompact3d/Incompact3d
    .. _traitlets:
        https://traitlets.readthedocs.io/en/stable/index.html
    .. _ipywidgets:
        https://ipywidgets.readthedocs.io/en/latest/

    .. note:: This is a work in progress, not all parameters are covered yet.
    """

    def __init__(self, *, raise_warning: bool = False, **kwargs):
        """Initializes the Parameters Class.

        Parameters
        ----------
        raise_warning : bool, optional
            Raise a warning instead of an error if an invalid parameter is found.
            By default False.
        **kwargs
            Keyword arguments for valid attributes, like ``nx``, ``re`` and so on.

        Raises
        -------
        KeyError
            Exception is raised when an Keyword arguments is not a valid attribute.

        Examples
        --------

        There are a few ways to initialize the class.

        First, calling it with no arguments initializes all variables with default value:

        >>> prm = xcompact3d_toolbox.Parameters()

        It is possible to set any value afterwards:

        >>> prm.re = 1e6
        >>> prm.set(
        ...     iibm=0,
        ...     p_row=4,
        ...     p_col=2,
        ... )

        Second, we can specify some values, and let the missing ones be
        initialized with default value:

        >>> prm = x3d.Parameters(
        ...     filename="example.i3d",
        ...     itype=12,
        ...     nx=257,
        ...     ny=129,
        ...     nz=32,
        ...     xlx=15.0,
        ...     yly=10.0,
        ...     zlz=3.0,
        ...     nclx1=2,
        ...     nclxn=2,
        ...     ncly1=1,
        ...     nclyn=1,
        ...     nclz1=0,
        ...     nclzn=0,
        ...     re=300.0,
        ...     init_noise=0.0125,
        ...     dt=0.0025,
        ...     ilast=45000,
        ...     ioutput=200,
        ...     iprocessing=50,
        ... )

        And finally, it is possible to read the parameters from the disc:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="example.i3d")

        It also supports the previous parameters file format (see `#7`_):

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="incompact3d.prm")

        .. _#7:
            https://github.com/fschuch/xcompact3d_toolbox/issues/7

        .. versionchanged:: 1.2.0
            The argument ``raise_warning`` changed to keyword-only.

        """

        super().__init__()

        if "loadfile" in kwargs:
            self.filename = kwargs.pop("loadfile")
            self.load(raise_warning=raise_warning)

        self.set(raise_warning=raise_warning, **kwargs)

        data_path = os.path.join(os.path.dirname(self.filename), "data")
        self.dataset.set(data_path=data_path)

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
        representation of the ``.i3d`` file."""
        # These groups are demanded by Xcompact3d, see parameters.f90
        dictionary = {
            "BasicParam": {},
            "NumOptions": {},
            "InOutParam": {},
            "Statistics": {},
            "CASE": {},
        }
        for name in self.trait_names():
            # if skip_default:
            #     if getattr(self, name) == self.trait_defaults(name):
            #         continue
            group = self.trait_metadata(name, "group")
            if group is not None:
                if group not in dictionary:
                    dictionary[group] = {}
                dictionary[group][name] = getattr(self, name)

        # This block is not handled by x3d if ilesmod is off
        if "LESModel" in dictionary and self.ilesmod == 0:
            del dictionary["LESModel"]

        # This block is not handled by x3d if iibm is off
        if "ibmstuff" in dictionary and self.iibm == 0:
            del dictionary["ibmstuff"]

        # This block is not handled by x3d if numscalar is 0
        if "ScalarParam" in dictionary and self.numscalar == 0:
            del dictionary["ScalarParam"]

        # This block is not handled by x3d if iturbine is not 1
        if "ALMParam" in dictionary and self.iturbine != 1:
            del dictionary["ALMParam"]

        string = ""

        string += "! -*- mode: f90 -*-\n"

        for blockkey, block in dictionary.items():
            string += "\n"
            string += "!===================\n"
            string += "&" + blockkey + "\n"
            string += "!===================\n"
            string += "\n"

            for paramkey, paramvalue in block.items():
                # get description to print together with the values
                description = self.trait_metadata(paramkey, "desc")
                if description is None:
                    description = ""
                # Check if param is a list or not
                if isinstance(paramvalue, list):
                    for n, p in enumerate(paramvalue):
                        string += f"{paramkey+'('+str(n+1)+')':>15} = {p:<15} {'! '+description}\n"
                # Check if param is a string
                elif isinstance(paramvalue, str):
                    new_paramvalue = "'" + paramvalue + "'"
                    string += f"{paramkey:>15} = {new_paramvalue:<15} {'! '+description}\n"
                elif isinstance(paramvalue, bool):
                    new_paramvalue = ".true." if paramvalue else ".false."
                    string += f"{paramkey:>15} = {new_paramvalue:<15} {'! '+description}\n"
                else:
                    string += f"{paramkey:>15} = {paramvalue:<15} {'! '+description}\n"
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
            for boundary_condition in f"ncl{dim}1 ncl{dim}n ncl{dim}S1 ncl{dim}Sn".split():
                setattr(self, boundary_condition, 0)
                getattr(self.mesh, dim)
            getattr(self.mesh, dim).is_periodic = True
        if change["old"] == 0 and change["new"] != 0:
            for boundary_condition in f"ncl{dim}1 ncl{dim}n ncl{dim}S1 ncl{dim}Sn".split():
                setattr(self, boundary_condition, change["new"])
            getattr(self.mesh, dim).is_periodic = False

    @traitlets.observe("p_row", "p_col", "ncores")
    def _observe_2decomp(self, change):
        if change["name"] == "ncores":
            self.p_row, self.p_col = 0, 0
        elif change["name"] == "p_row":
            try:
                self.p_col = self.ncores // self.p_row
            except ZeroDivisionError:
                self.p_col = 0
        elif change["name"] == "p_col":
            try:
                self.p_row = self.ncores // self.p_col
            except ZeroDivisionError:
                self.p_row = 0

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
    def _observe_size(self, _):
        def convert_bytes(num):
            """
            this function will convert bytes to MB.... GB... etc
            """
            step_unit = 1000.0  # 1024 bad the size

            for x in ["bytes", "KB", "MB", "GB", "TB"]:
                if num < step_unit:
                    return f"{num:3.1f} {x}"
                num /= step_unit
            return None

        prec = 4 if param["mytype"] == np.float32 else 8

        # Restart Size from tools.f90
        count = 3 + self.numscalar  # ux, uy, uz, phi
        # Previous time-step if necessary
        if self.itimescheme in [3, 7]:
            count *= 3
        elif self.itimescheme == 2:  # noqa: PLR2004
            count *= 2
        count += 1  # pp
        count *= self.nx * self.ny * self.nz * prec * (self.ilast // self.icheckpoint - 1)

        # 3D from visu.f90: ux, uy, uz, pp and phi
        count += (4 + self.numscalar) * self.nx * self.ny * self.nz * prec * self.ilast // self.ioutput

        # 2D planes from BC.Sandbox.f90
        if self.itype == 10:  # noqa: PLR2004
            # xy planes avg and central plane for ux, uy, uz and phi
            count += 2 * (3 + self.numscalar) * self.nx * self.ny * prec * self.ilast // self.iprocessing
            # xz planes avg, top and bot for ux, uy, uz and phi
            count += 3 * (3 + self.numscalar) * self.nx * self.nz * prec * self.ilast // self.iprocessing

        self.size = convert_bytes(count)

    def get_boundary_condition(self, variable_name: str) -> dict:
        """This method returns the appropriate boundary parameters that are
        expected by the derivatives methods.

        Parameters
        ----------
        variable_name : str
            Variable name. The supported options are ``ux``, ``uy``,
            ``uz``, ``pp`` and ``phi``, otherwise the method returns the default
            option.

        Returns
        -------
        :obj:`dict` of :obj:`dict`
            A dict containing the boundary conditions for the variable specified.

        Examples
        --------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> prm.get_boundary_condition("ux")
        {'x': {'ncl1': 1, 'ncln': 1, 'npaire': 0},
        'y': {'ncl1': 1, 'ncln': 2, 'npaire': 1, 'istret': 0, 'beta': 0.75},
        'z': {'ncl1': 0, 'ncln': 0, 'npaire': 1}}

        It is possible to store this information as an attribute in any
        :obj:`xarray.DataArray`:

        >>> DataArray.attrs["BC"] = prm.get_boundary_condition("ux")

        So the correct boundary conditions will be used to compute the derivatives:

        >>> DataArray.x3d.first_derivative("x")
        >>> DataArray.x3d.second_derivative("x")

        """

        return boundary_condition(self, variable_name)

    def set(self, *, raise_warning: bool = False, **kwargs) -> None:
        """Set a new value for any parameter after the initialization.

        Parameters
        ----------
        raise_warning : bool, optional
            Raise a warning instead of an error if an invalid parameter is found.
            By default False.
        **kwargs
            Keyword arguments for valid attributes, like ``nx``, ``re`` and so on.

        Raises
        ------
        KeyError
            Exception is raised when an Keyword arguments is not a valid attribute.

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> prm.set(
        ...     iibm=0,
        ...     p_row=4,
        ...     p_col=2,
        ... )

        .. versionchanged:: 1.2.0
            The argument ``raise_warning`` changed to keyword-only.
        """
        # They are high priority in order to avoid errors with validations and observations
        for bc in "nclx1 nclxn ncly1 nclyn nclz1 nclzn numscalar ilesmod".split():
            if bc in kwargs:
                setattr(self, bc, kwargs.get(bc))

        for key, arg in kwargs.items():
            if key not in self.trait_names():
                if raise_warning:
                    logger.warning(f"{key} is not a valid parameter and was not loaded")
                else:
                    msg = f"{key} is not a valid parameter"
                    raise KeyError(msg)
            setattr(self, key, arg)

    def from_string(self, string: str, *, raise_warning: bool = False) -> None:
        """Loads the attributes from a string.

        Parameters
        ----------
        filename : str, optional
            The filename for the parameters file. If None, it uses the filename specified
            in the class (default is :obj:`None`).
        raise_warning : bool, optional
            Raise a warning instead of an error if an invalid parameter is found.
            By default False.

        Raises
        ------
        KeyError
            Exception is raised when an attributes is invalid.
        """

        dictionary = {}

        # unpacking the nested dictionary
        for value_out in i3d_to_dict(string=string).values():
            for key_in, value_in in value_out.items():
                dictionary[key_in] = value_in

        self.set(raise_warning=raise_warning, **dictionary)

    def from_file(self, filename: str | None = None, *, raise_warning: bool = False) -> None:
        """Loads the attributes from the parameters file.

        It also includes support for the previous format :obj:`.prm`  (see `#7`_).

        Parameters
        ----------
        filename : str, optional
            The filename for the parameters file. If None, it uses the filename specified
            in the class (default is :obj:`None`).
        raise_warning : bool, optional
            Raise a warning instead of an error if an invalid parameter is found.
            By default False.

        Raises
        ------
        IOError
            If the file format is not :obj:`.i3d` or :obj:`.prm`.
        KeyError
            Exception is raised when an attributes is invalid.

        Examples
        -------

        >>> prm = xcompact3d_toolbox.Parameters(filename="example.i3d")
        >>> prm.load()

        >>> prm = xcompact3d_toolbox.Parameters()
        >>> prm.load("example.i3d")

        or just:

        >>> prm = xcompact3d_toolbox.Parameters(loadfile="example.i3d")
        >>> prm = xcompact3d_toolbox.Parameters(loadfile="incompact3d.prm")

        .. _#7:
            https://github.com/fschuch/xcompact3d_toolbox/issues/7

        """
        if filename is None:
            filename = self.filename
        if self.filename.split(".")[-1] == "i3d":
            dictionary = {}

            # unpacking the nested dictionary
            for value_out in i3d_to_dict(self.filename).values():
                for key_in, value_in in value_out.items():
                    dictionary[key_in] = value_in

        elif self.filename.split(".")[-1] == "prm":
            dictionary = prm_to_dict(self.filename)

        else:
            msg = f"{self.filename} is invalid. Supported formats are .i3d and .prm."
            raise OSError(msg)

        self.set(raise_warning=raise_warning, **dictionary)

    def load(self, *arg, **kwarg) -> None:
        """An alias for :obj:`Parameters.from_file`"""
        self.from_file(*arg, **kwarg)

    def write(self, filename: str | None = None) -> None:
        """Write all valid attributes to an :obj:`.i3d` file.

        An attribute is considered valid if it has a ``tag`` named ``group``,
        witch assigns it to the respective namespace at the :obj:`.i3d` file.

        Parameters
        ----------
        filename : str, optional
            The filename for the :obj:`.i3d` file. If None, it uses the filename specified
            in the class (default is :obj:`None`).

        Examples
        --------

        >>> prm = xcompact3d_toolbox.Parameters(
        ...     filename="example.i3d",
        ...     nx=101,
        ...     ny=65,
        ...     nz=11,
        ...     # and so on...
        ... )
        >>> prm.write()

        or just:

        >>> prm.write("example.i3d")

        """
        if filename is None:
            filename = self.filename
        if filename.split(".")[-1] == "i3d":
            with open(filename, "w", encoding="utf-8") as file:
                file.write(self.__str__())
        else:
            msg = "Format error, only .i3d is supported"
            raise OSError(msg)

    def get_mesh(self, *, refined_for_ibm: bool = False) -> dict:
        """Get mesh the three-dimensional coordinate system. The coordinates are stored
        in a dictionary. It supports mesh refinement in **y** when
        :obj:`ParametersBasicParam.istret` :math:`\\ne` 0.

        Parameters
        ----------
        refined_for_ibm : bool
            If True, it returns a refined mesh as a function of :obj:`ParametersIbmStuff.nraf` (default is False).

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
                new_grid_size = getattr(self.mesh, dim)._sub_grid_size * self.nraf  # noqa: SLF001
                if not getattr(self.mesh, dim).is_periodic:
                    new_grid_size += 1
                getattr(copy, dim).set(grid_size=new_grid_size)
            return copy.get()
        return self.mesh.get()
