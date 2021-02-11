# -*- coding: utf-8 -*-
"""
Manipulate the physical and computational parameters, just like :obj:`xcompact3d_toolbox.parameters.Parameters`,
but with `ipywidgets`_.

.. _ipywidgets:
    https://ipywidgets.readthedocs.io/en/latest/
"""

import ipywidgets as widgets
from traitlets import link
from IPython.display import display
from .parameters import Parameters, possible_mesh, possible_mesh_p


class ParametersGui(Parameters):
    """This class is derivated from :obj:`xcompact3d_toolbox.parameters.Parameters`,
    including all its features. In addition, there is a two way link between the
    parameters and their widgets. Control them with code and/or with the graphical
    user interface.

    """

    def __init__(self, **kargs):
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

        >>> prm = xcompact3d_toolbox.ParametersGui()

        It is possible to set any values afterwards (including new atributes):

        >>> prm.re = 1e6

        Second, we can specify some values, and let the missing ones be
        initialized with default value:

        >>> prm = x3d.ParametersGui(
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

        >>> prm = xcompact3d_toolbox.ParametersGui(loadfile = 'example.i3d')

        It also supports the previous parameters file format (see `#7`_):

        >>> prm = xcompact3d_toolbox.ParametersGui(loadfile = 'incompact3d.prm')

        .. _#7:
            https://github.com/fschuch/xcompact3d_toolbox/issues/7

        """
        super(ParametersGui, self).__init__(**kargs)

        self._widgets = dict(
            #
            # # BasicParam
            #
            beta=widgets.BoundedFloatText(min=0.0, max=1e9),
            dt=widgets.BoundedFloatText(min=1e-9, max=1e9),
            ifirst=widgets.BoundedIntText(min=0, max=1e9),
            iibm=widgets.Dropdown(
                options=[
                    ("Off", 0),
                    ("Forced to zero", 1),
                    ("Interpolated to zero", 2),
                ],
            ),
            iin=widgets.Dropdown(
                options=[
                    ("No random noise", 0),
                    ("Random noise", 1),
                    ("Random noise with fixed seed", 2),
                ],
            ),
            ilast=widgets.BoundedIntText(min=0, max=1e9),
            inflow_noise=widgets.FloatText(min=-1e9, max=1e9),
            init_noise=widgets.FloatText(min=-1e9, max=1e9),
            istret=widgets.Dropdown(
                options=[
                    ("No refinement", 0),
                    ("Refinement at the center", 1),
                    ("Both sides", 2),
                    ("Just near the bottom", 3),
                ],
            ),
            itype=widgets.Dropdown(
                options=[
                    ("User", 0),
                    ("Lock-exchange", 1),
                    ("Taylor-Green Vortex", 2),
                    ("Channel", 3),
                    ("Periodic Hill", 4),
                    ("Cylinder", 5),
                    ("Debug Schemes", 6),
                    ("Mixing Layer", 7),
                    ("Turbulent Jet", 8),
                    ("Turbulent Boundary Layer", 9),
                    ("Sandbox", 10),
                ],
            ),
            nclxn=widgets.Dropdown(
                options=[("Periodic", 0), ("Free-slip", 1), ("Outflow", 2)],
            ),
            nclx1=widgets.Dropdown(
                options=[("Periodic", 0), ("Free-slip", 1), ("Inflow", 2)],
            ),
            numscalar=widgets.IntSlider(min=0, max=9, continuous_update=False),
            p_col=widgets.Dropdown(options=self._possible_p_col),
            p_row=widgets.Dropdown(options=self._possible_p_row),
            re=widgets.FloatText(min=0.0, max=1e9),
            #
            # # NumOptions
            #
            cnu=widgets.BoundedFloatText(min=0.0, max=1e6),  # , disabled=True
            ifirstder=widgets.Dropdown(
                options=[
                    ("2nd central", 1),
                    ("4th central", 1),
                    ("4th compact", 1),
                    ("6th compact", 4),
                ],
            ),
            isecondder=widgets.Dropdown(
                disabled=True,
                options=[
                    # '2nd central', 1),
                    ("6th compact", 4),
                    ("hyperviscous 6th", 5),
                ],
            ),
            itimescheme=widgets.Dropdown(
                options=[
                    ("Euler", 1),
                    ("AB2", 2),
                    ("AB3", 3),
                    ("RK3", 5),
                    ("Semi-implicit", 7),
                ],
            ),
            nu0nu=widgets.BoundedFloatText(min=0.0, max=1e6),  # , disabled=True
            #
            # # InOutParam
            #
            irestart=widgets.Dropdown(options=[("Off", 0), ("On", 1)]),
            nvisu=widgets.BoundedIntText(min=1, max=1e9, disabled=True),
            #
            # # ScalarParam
            #
            iibmS=widgets.Dropdown(
                options=[
                    ("Off", 0),
                    ("Forced to zero", 1),
                    ("Interpolated to zero", 2),
                    ("Interpolated to no-flux", 3),
                ],
            ),
            #
            # # LESModel
            #
            jles=widgets.Dropdown(
                options=[
                    ("DNS", 0),
                    ("Phys Smag", 1),
                    ("Phys WALE", 2),
                    ("Phys dyn. Smag", 3),
                    ("iSVV", 4),
                ],
            ),
            #
            # # ibmstuff
            #
            # Disabled because nobjmax is defined automatically at genepsi.py
            nobjmax=widgets.IntText(disabled=True, min=0, max=1e9),
            nraf=widgets.IntSlider(min=1, max=25),
            #
            # # Auxiliar for user interface, not included at the .i3d file
            #
            filename=widgets.Text(),
            ncores=widgets.BoundedIntText(value=0, min=0, max=1e9),
            size=widgets.Text(value="", disabled=True),
        )

        for name in "gravx gravy gravz".split():
            self._widgets[name] = widgets.FloatText(min=-1.0, max=1.0)

        for name in "nx ny nz".split():
            self._widgets[name] = widgets.Dropdown(options=possible_mesh)

        for name in "xlx yly zlz".split():
            self._widgets[name] = widgets.BoundedFloatText(min=0.0, max=1e9)

        for name in "ncly1 nclyn nclz1 nclzn".split():
            self._widgets[name] = widgets.Dropdown(
                options=[("Periodic", 0), ("Free-slip", 1), ("No-slip", 2)],
            )

        for name in "ilesmod ivisu ipost".split():
            self._widgets[name] = widgets.Dropdown(options=[("Off", 0), ("On", 1)])

        for name in "icheckpoint ioutput iprocessing".split():
            self._widgets[name] = widgets.BoundedIntText(min=1, max=1e9, step=100)

        for name in "nclxS1 nclxSn nclyS1 nclySn nclzS1 nclzSn".split():
            self._widgets[name] = widgets.Dropdown(
                options=[("Periodic", 0), ("No-flux", 1), ("Dirichlet", 2)],
            )

        for name in "dx dy dz".split():
            self._widgets[name] = widgets.BoundedFloatText(min=0.0, max=1e6)

        # Add a name to all widgets (same as dictionary key)
        for name in self._widgets.keys():
            self._widgets[name].description = name

        # Try to add a description
        for name in self._widgets.keys():
            # get description to include together with widgets
            description = self.trait_metadata(name, "desc")
            if description is not None:
                self._widgets[name].description_tooltip = description

        # Creating an arrange with all widgets
        dim = "x y z".split()

        self.ipyview = widgets.VBox(
            [
                widgets.HTML(value="<h1>Xcompact3d Parameters</h1>"),
                widgets.HBox(
                    [
                        self._widgets["filename"],
                        widgets.Button(
                            description="Read", disabled=True, icon="file-upload"
                        ),
                        widgets.Button(
                            description="Write", disabled=True, icon="file-download"
                        ),
                        widgets.Button(description="Run", disabled=True, icon="rocket"),
                        widgets.Button(description="Sync", disabled=True, icon="sync"),
                    ]
                ),
                widgets.HTML(value="<h2>BasicParam</h2>"),
                widgets.HBox([self._widgets[d] for d in "itype re".split()]),
                widgets.HBox(
                    [self._widgets[d] for d in "iin init_noise inflow_noise".split()]
                ),
                widgets.HTML(value="<h3>Domain Decomposition</h3>"),
                widgets.HBox([self._widgets[d] for d in "ncores p_row p_col".split()]),
                widgets.HTML(value="<h3>Temporal discretization</h3>"),
                widgets.HBox([self._widgets[d] for d in "ifirst ilast dt".split()]),
                widgets.HTML(value="<h3>InOutParam</h3>"),
                widgets.HBox([self._widgets[d] for d in "irestart nvisu size".split()]),
                widgets.HBox(
                    [
                        self._widgets[d]
                        for d in "icheckpoint ioutput iprocessing".split()
                    ]
                ),
                widgets.HTML(value="<h3>Spatial discretization</h3>"),
                widgets.HBox([self._widgets[f"n{d}"] for d in dim]),
                widgets.HBox([self._widgets[f"{d}l{d}"] for d in dim]),
                widgets.HBox([self._widgets[f"d{d}"] for d in dim]),
                widgets.HBox([self._widgets[f"ncl{d}1"] for d in dim]),
                widgets.HBox([self._widgets[f"ncl{d}n"] for d in dim]),
                widgets.HBox([self._widgets[d] for d in "istret beta".split()]),
                widgets.HTML(value="<h2>NumOptions</h2>"),
                widgets.HBox(
                    [
                        self._widgets[d]
                        for d in "ifirstder isecondder itimescheme".split()
                    ]
                ),
                widgets.HBox([self._widgets[d] for d in "ilesmod nu0nu cnu".split()]),
                widgets.HTML(value="<h2>ScalarParam</h2>"),
                widgets.HBox([self._widgets["numscalar"]]),
                widgets.HBox([self._widgets[f"ncl{d}S1"] for d in dim]),
                widgets.HBox([self._widgets[f"ncl{d}Sn"] for d in dim]),
                widgets.HBox([self._widgets[f"grav{d}"] for d in dim]),
                widgets.HBox([self._widgets[d] for d in "iibmS".split()]),
                widgets.HTML(
                    value="<strong>cp, us, sc, ri, scalar_lbound & scalar_ubound</strong> are lists with length numscalar, set them properly on the code."
                ),
                widgets.HTML(value="<h2>IBMStuff</h2>"),
                widgets.HBox([self._widgets[d] for d in "iibm nraf nobjmax".split()]),
            ]
        )

        self.link_widgets()

    def __call__(self, *args):
        """Returns widgets on demand.
        Parameters
        ----------
        *args : str
            Name(s) for the desired widget(s).
        Returns
        -------
        :obj:`ipywidgets.VBox`
            Widgets for an user friendly interface.

        Examples
        -------
        >>> prm = xcompact3d_toolbox.ParametersGui()
        >>> prm('nx', 'xlx', 'dx', 'nclx1', 'nclxn')
        """

        return widgets.VBox([self._widgets[name] for name in args])

    def _ipython_display_(self):
        display(self.ipyview)

    def link_widgets(self):
        """Creates a two-way link between the value of an attribute and its widget.
        This method is called at initialization, but provides an easy way to link
        any new variable.

        Examples
        -------
        >>> prm = xcompact3d_toolbox.ParametersGui(loadfile = 'example.i3d')
        >>> prm.link_widgets()
        """

        # Link the possible mesh values with the respective dropdown widget
        for dim in ["x", "y", "z"]:
            link(
                (self, f"_possible_mesh_{dim}"), (self._widgets[f"n{dim}"], "options"),
            )

        # Link the possible domain decomposition values with the respective dropdown widget
        for name in ["p_row", "p_col"]:
            link(
                (self, f"_possible_{name}"), (self._widgets[f"{name}"], "options"),
            )

        # Create two-way link between variables and widgets
        for name in self._widgets.keys():
            link((self, name), (self._widgets[name], "value"))

        # for name in self._widgets.keys():
        #     if name == "numscalar":
        #         continue
        #     group = self.trait_metadata(name, "group")
        #     if group == "ScalarParam":
        #         link(
        #             (self, "_iscalar"), (self._widgets[name], "disabled"),
        #         )
