import numpy as np
import math
import traitlets
import ipywidgets as widgets
from .param import param
from .mesh import get_mesh
from .io import i3d_to_dict, dict_to_i3d

possible_mesh = [
    9, 11, 13, 17, 19, 21, 25, 31, 33, 37, 41, 49, 51, 55, 61, 65, 73, 81, 91, 97,
    101, 109, 121, 129, 145, 151, 161, 163, 181, 193, 201, 217, 241, 251, 257,
    271, 289, 301, 321, 325, 361, 385, 401, 433, 451, 481, 487, 501, 513, 541,
    577, 601, 641, 649, 721, 751, 769, 801, 811, 865, 901, 961, 973, 1001, 1025,
    1081, 1153, 1201, 1251, 1281, 1297, 1351, 1441, 1459, 1501, 1537, 1601,
    1621, 1729, 1801, 1921, 1945, 2001, 2049, 2161, 2251, 2305, 2401, 2431,
    2501, 2561, 2593, 2701, 2881, 2917, 3001, 3073, 3201, 3241, 3457, 3601,
    3751, 3841, 3889, 4001, 4051, 4097, 4321, 4375, 4501, 4609, 4801, 4861,
    5001, 5121, 5185, 5401, 5761, 5833, 6001, 6145, 6251, 6401, 6481, 6751,
    6913, 7201, 7291, 7501, 7681, 7777, 8001, 8101, 8193, 8641, 8749, 9001
]

possible_mesh_p = [i - 1 for i in possible_mesh]

def divisorGenerator(n):
    large_divisors = [0]
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield int(divisor)

class Parameters(traitlets.HasTraits):
    #
    # # BasicParam
    #
    p_row, p_col = [
        traitlets.Int(default_value=0, min=0).tag(
            group = 'BasicParam',
            widget = widgets.Dropdown(description=name, options=[0])
        )
        for name in ['p_row', 'p_col']
    ]

    itype = traitlets.Int(default_value=10, min=0, max=10).tag(
        group = 'BasicParam',
        widget = widgets.Dropdown(
            description='itype',
            disabled=True,
            options=[('User', 0),
                     ('Lock-exchange', 1),
                     ('Taylor-Green Vortex', 2),
                     ('Channel', 3),
                     ('Periodic Hill', 4),
                     ('Cylinder', 5),
                     ('Debug Schemes', 6),
                     ('Mixing Layer', 7),
                     ('Turbulent Jet', 8),
                     ('Turbulent Boundary Layer', 9),
                     ('Sandbox', 10)
                    ])
    )

    iin = traitlets.Int(default_value=0, min=0, max=2).tag(
        group = 'BasicParam',
        widget = widgets.Dropdown(
            description='iin',
            options=[('No random noise', 0),
                     ('Random noise', 1),
                     ('Random noise with fixed seed', 2)
                    ])
    )

    nx, ny, nz = [
        traitlets.Int(default_value=17, min=0).tag(
            group='BasicParam',
            widget = widgets.Dropdown(description=name, options=possible_mesh)
        )
        for name in ['nx', 'ny', 'nz']
    ]

    xlx, yly, zlz = [
        traitlets.Float(default_value=1.0, min=0).tag(
            group='BasicParam',
            widget = widgets.BoundedFloatText(description=name, min=0.0, max=1e6)
        )
        for name in ['xlx', 'yly', 'zlz']
    ]

    nclx1 = traitlets.Int(default_value=2, min=0, max=2).tag(
        group='BasicParam',
        widget = widgets.Dropdown(
            description='nclx1',
            options=[
                ('Periodic', 0),
                ('Free-slip', 1),
                ('Inflow', 2)
            ]
        )
    )

    nclxn = traitlets.Int(default_value=2, min=0, max=2).tag(
        group='BasicParam',
        widget = widgets.Dropdown(
            description='nclxn',
            options=[
                ('Periodic', 0),
                ('Free-slip', 1),
                ('Outflow', 2)
            ]
        )
    )

    ncly1, nclyn, nclz1, nclzn = [
        traitlets.Int(default_value=2, min=0, max=2).tag(
            group='BasicParam',
            widget = widgets.Dropdown(
                description=name,
                options=[
                    ('Periodic', 0),
                    ('Free-slip', 1),
                    ('No-slip', 2)
                ]
            )
        )
        for name in 'ncly1 nclyn nclz1 nclzn'.split()
    ]

    ivisu, ipost, ilesmod = [traitlets.Bool(default_value=True) for i in range(3)]

    istret = traitlets.Int(default_value=0, min=0, max=3).tag(
        group='BasicParam',
        widget = widgets.Dropdown(
            description='istret',
            disabled=True,
            options=[
                ('No refinement', 0),
                ('Refinement at the center', 1),
                ('Both sides', 2),
                ('Just near the bottom', 3)
            ]
        )
    )

    beta = traitlets.Float(default_value=1.0, min=0).tag(
        group='BasicParam',
        widget = widgets.BoundedFloatText(
            description='beta', min=0.0, max=1e6, disabled=True)
    )

    dt = traitlets.Float(default_value=1e-3, min=0.0).tag(
        group='BasicParam',
        widget = widgets.BoundedFloatText(description='dt', min=0.0, max=1e6)
    )

    ifirst, ilast = [
        traitlets.Int(default_value=0, min=0).tag(
            group='BasicParam',
            widget = widgets.IntText(description=name)
        )
        for name in ['ifirst', 'ilast']
    ]

    re = traitlets.Float(default_value=1e3).tag(
        group='BasicParam',
        widget = widgets.FloatText(description='re')
    )

    init_noise = traitlets.Float(default_value=0.0).tag(
        group='BasicParam',
        widget = widgets.FloatText(description='init_noise')
    )

    inflow_noise = traitlets.Float(default_value=0.0).tag(
        group='BasicParam',
        widget = widgets.FloatText(description='inflow_noise')
    )

    ilesmod, ivisu, ipost = [
        traitlets.Int(default_value=1, min=0, max=1).tag(
            group='BasicParam',
            widget = widgets.Dropdown(
                description=name,
                options=[
                    ('Off', 0),
                    ('On', 1)
                ]
            )
        )
        for name in ['ilesmod', 'ivisu', 'ipost']
    ]

    iibm = traitlets.Int(default_value=0, min=0, max=2).tag(
            group='BasicParam',
            widget = widgets.Dropdown(
                description='iibm',
                options=[
                    ('Off', 0),
                    ('Forced to zero', 1),
                    ('Interpolated to zero', 2)
                ]
            )
    )

    # More than 9 will bug Xcompact3d, because it handles scalar fields with
    # just one digit
    numscalar = traitlets.Int(default_value=0, min=0, max=9).tag(
        group='BasicParam',
        widget = widgets.IntSlider(min=0, max=9,
            description='numscalar', continuous_update=False)
    )

    gravx, gravy, gravz = [
        traitlets.Float(default_value=0.0).tag(
            group='BasicParam',
            widget = widgets.FloatText(description=name, disabled=True)
        )
        for name in ['gravx', 'gravy', 'gravz']
    ]

    #
    # # NumOptions
    #

    ifirstder = traitlets.Int(default_value=4, min=1, max=4).tag(
        group = 'NumOptions',
        widget = widgets.Dropdown(
            description='ifirstder',
            disabled=True,
            options=[
                ('2nd central', 1),
                ('6th compact', 4)
            ])
    )

    isecondder = traitlets.Int(default_value=4, min=1, max=5).tag(
        group = 'NumOptions',
        widget = widgets.Dropdown(
            description='isecondder',
            disabled=True,
            options=[
                #'2nd central', 1),
                ('6th compact', 4),
                ('hyperviscous 6th', 5),
            ])
    )

    ipinter = traitlets.Int(3)

    itimescheme = traitlets.Int(default_value=3, min=1, max=7).tag(
        group = 'NumOptions',
        widget = widgets.Dropdown(
            description='itimescheme',
            options=[
                ('Euler', 1),
                ('AB2', 2),
                ('AB3', 3),
                ('RK3', 5),
                ('Semi-implicit', 7),
            ])
    )

    nu0nu = traitlets.Float(default_value=4, min=0.0).tag(
            group = 'NumOptions',
            widget = widgets.BoundedFloatText(
                description='nu0nu', min=0.0, max=1e6, disabled=True)
    )

    cnu = traitlets.Float(default_value=0.44, min=0.0).tag(
            group = 'NumOptions',
            widget = widgets.BoundedFloatText(
                description='cnu', min=0.0, max=1e6, disabled=True)
    )

    #
    # # InOutParam
    #

    irestart = traitlets.Int(default_value=0, min=0, max=1).tag(
            group='InOutParam',
            widget = widgets.Dropdown(
                description='irestart',
                options=[
                    ('Off', 0),
                    ('On', 1)
                ]
            )
    )

    nvisu = traitlets.Int(default_value=1, min=1).tag(
        group='InOutParam',
        widget = widgets.BoundedIntText(
            description = 'nvisu', min=1, max=1e9, disabled=True)
    )

    icheckpoint, ioutput, iprocessing = [
        traitlets.Int(default_value=1000, min=1).tag(
            group='InOutParam',
            widget = widgets.BoundedIntText(description=name,min=1, max=1e9)
        )
        for name in ['icheckpoint', 'ioutput', 'iprocessing']
    ]

    ifilenameformat = traitlets.Int(default_value=9, min=1)

    #
    # # ScalarParam
    #

    iscalar = traitlets.Bool(False)

    # Include widgets for list demands some planning about code design
    sc = traitlets.List(trait=traitlets.Float()).tag(group='ScalarParam')
    ri = traitlets.List(trait=traitlets.Float()).tag(group='ScalarParam')
    uset = traitlets.List(trait=traitlets.Float()).tag(group='ScalarParam')
    cp = traitlets.List(trait=traitlets.Float()).tag(group='ScalarParam')
    scalar_lbound = traitlets.List(trait=traitlets.Float(default_value = -1e6)).tag(group='ScalarParam')
    scalar_ubound = traitlets.List(trait=traitlets.Float(default_value = 1e6)).tag(group='ScalarParam')

    iibmS = traitlets.Int(default_value=0, min=0, max=3).tag(
            group='ScalarParam',
            widget = widgets.Dropdown(
                description='iibmS',
                options=[
                    ('Off', 0),
                    ('Forced to zero', 1),
                    ('Interpolated to zero', 2),
                    ('Interpolated to no-flux', 3)
                ]
            )
    )

    nclxS1, nclxSn, nclyS1, nclySn, nclzS1, nclzSn = [
        traitlets.Int(default_value=2, min=0, max=2).tag(
            group='ScalarParam',
            widget = widgets.Dropdown(
                description=name,
                options=[
                    ('Periodic', 0),
                    ('No-flux', 1),
                    ('Dirichlet', 2)
                ]
            )
        )
        for name in ['nclxS1', 'nclxSn', 'nclyS1', 'nclySn', 'nclzS1', 'nclzSn']
    ]

    #
    # # LESModel
    #

    jles = traitlets.Int(default_value=0, min=0, max=4).tag(
            group='LESModel',
            widget = widgets.Dropdown(
                description='ilesmod',
                options=[
                    ('DNS', 0),
                    ('Phys Smag', 1),
                    ('Phys WALE', 2),
                    ('Phys dyn. Smag', 3),
                    ('iSVV', 4)
                ],
            )
    )

    #
    # # ibmstuff
    #

    nobjmax = traitlets.Int(default_value=1, min=0).tag(
            group='ibmstuff',
            widget = widgets.IntText(description='nobjmax', disabled=True)
    )
    nraf = traitlets.Int(default_value=10, min=1).tag(
            group='ibmstuff',
            widget = widgets.IntSlider(min=1, max=25, description='nraf')
    )

    # Auxiliar
    filename = traitlets.Unicode(default_value='input.i3d').tag(
        widget = widgets.Text(description='filename')
    )

    i3d = traitlets.Dict(
        default_value = {
            'BasicParam': {},
            'NumOptions': {},
            'InOutParam': {},
            'Statistics': {},
            'ScalarParam': {},
            'LESModel': {},
            'WallModel': {},
            'ibmstuff': {},
            'ForceCVs': {},
            'CASE': {}
        }
    )

    mx, my, mz = [traitlets.Int(default_value=1, min=1) for i in range(3)]

    dx, dy, dz = [
        traitlets.Float(default_value=0.0625, min=0.0).tag(
            widget = widgets.BoundedFloatText(description=name, min=0.0, max=1e6)
        )
        for name in ['dx', 'dy', 'dz']
    ]

    nclx, ncly, nclz = [traitlets.Bool() for i in range(3)]

    _possible_mesh_x, _possible_mesh_y, _possible_mesh_z =[
        traitlets.List(trait=traitlets.Int(), default_value=possible_mesh) for i in range(3)
    ]

    ncores = traitlets.Int(default_value=4, min=1).tag(
            widget = widgets.BoundedIntText(value=0, min=0,description='ncores',max=1e9)
    )
    _possible_p_row, _possible_p_col = [
        traitlets.List(trait=traitlets.Int(), default_value=[0]) for i in range(2)
    ]
    #cfl = traitlets.Float(0.0)
    _size_in_disc = traitlets.Unicode().tag(
            widget = widgets.Text(value='', description='Size', disabled=True)
    )

    def __init__(self, **kwargs):
        #(filename='input.i3d', dictionary=None):
        super(Parameters, self).__init__()

        # Boundary conditions are high priority in order to avoid bugs
        for bc in 'nclx1 nclxn ncly1 nclyn nclz1 nclzn'.split():
            if bc in kwargs:
                setattr(self, bc, kwargs[bc])

        for key, arg in kwargs.items():
            if key not in self.trait_names():
                raise KeyError(f'There is no parameter named {key}!')
            setattr(self, key, arg)

        #self.link_widgets()

    def __call__(self, *args):

        if len(args) == 0:
            dim = 'x y z'.split()

            return widgets.VBox([
                widgets.HTML(value='<h1>Xcompact3d Parameters</h1>'),
                widgets.HBox([
                    self.trait_metadata('filename', 'widget'),
                    widgets.Button(description="Read", disabled=True, icon='file-upload'),
                    widgets.Button(description="Write", disabled=True, icon='file-download'),
                    widgets.Button(description="Run", disabled=True, icon='rocket'),
                    widgets.Button(description="Sync", disabled=True, icon='sync'),
                ]),
                widgets.HTML(value='<h2>BasicParam</h2>'),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'itype re'.split()]),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'iin init_noise inflow_noise'.split()]),
                widgets.HTML(value='<h3>Domain Decomposition</h3>'),
                widgets.HBox([self.trait_metadata(f'{d}', 'widget') for d in 'ncores p_row p_col'.split()]),
                widgets.HTML(value='<h3>Temporal discretization</h3>'),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'ifirst ilast dt'.split()]),
                widgets.HTML(value='<h3>InOutParam</h3>'),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'irestart nvisu _size_in_disc'.split()]),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'icheckpoint ioutput iprocessing'.split()]),
                widgets.HTML(value='<h3>Spatial discretization</h3>'),
                widgets.HBox([self.trait_metadata(f'n{d}', 'widget') for d in dim]),
                widgets.HBox([self.trait_metadata(f'{d}l{d}', 'widget') for d in dim]),
                widgets.HBox([self.trait_metadata(f'd{d}', 'widget') for d in dim]),
                widgets.HBox([self.trait_metadata(f'ncl{d}1', 'widget') for d in dim]),
                widgets.HBox([self.trait_metadata(f'ncl{d}n', 'widget') for d in dim]),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'istret beta'.split()]),
                widgets.HTML(value='<h2>NumOptions</h2>'),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'ifirstder isecondder itimescheme'.split()]),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'ilesmod nu0nu cnu'.split()]),
                widgets.HTML(value='<h2>ScalarParam</h2>'),
                widgets.HBox([self.trait_metadata('numscalar', 'widget')]),
                widgets.HBox([self.trait_metadata(f'ncl{d}S1', 'widget') for d in dim]),
                widgets.HBox([self.trait_metadata(f'ncl{d}Sn', 'widget') for d in dim]),
                widgets.HBox([self.trait_metadata(f'grav{d}', 'widget') for d in dim]),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'iibmS'.split()]),
                widgets.HTML(value='<strong>cp, us, sc, ri, scalar_lbound & scalar_ubound</strong> are lists with length numscalar, set them properly on the code.'),
                widgets.HTML(value='<h2>IBMStuff</h2>'),
                widgets.HBox([self.trait_metadata(d, 'widget') for d in 'iibm nraf nobjmax'.split()]),
            ])

        widgets_list = []
        for name in args:
            widget = self.trait_metadata(name, 'widget')
            if widget != None:
                widgets_list.append(widget)

        return widgets.VBox(widgets_list)

    @traitlets.validate('nx')
    def _validade_mesh_nx(self, proposal):
        _validate_mesh(proposal['value'], self.nclx, self.nclx1, self.nclxn, 'x')
        return proposal['value']

    @traitlets.validate('ny')
    def _validade_mesh_ny(self, proposal):
        _validate_mesh(proposal['value'], self.ncly, self.ncly1, self.nclyn, 'y')
        return proposal['value']

    @traitlets.validate('nz')
    def _validade_mesh_nz(self, proposal):
        _validate_mesh(proposal['value'], self.nclz, self.nclz1, self.nclzn, 'z')
        return proposal['value']

    @traitlets.observe('dx', 'nx', 'xlx', 'dy', 'ny', 'yly', 'dz', 'nz', 'zlz')
    def _observe_resolution(self, change):
        # for name in 'name new old'.split():
        #     print(f'    {name}:{change[name]}')
        #
        dim = change['name'][-1] # It will be x, y or z
        #
        if change['name'] == f'n{dim}':
            if getattr(self, f'ncl{dim}'):
                setattr(self, f'm{dim}', change['new'])
            else:
                setattr(self, f'm{dim}', change['new'] - 1)
            setattr(
                self, f'd{dim}',
                getattr(self, f'{dim}l{dim}') / getattr(self, f'm{dim}')
            )
        if change['name'] == f'd{dim}':
            new_l = change['new'] * getattr(self, f'm{dim}')
            if new_l != getattr(self, f'{dim}l{dim}'):
                setattr(self, f'{dim}l{dim}',new_l)
        if change['name'] == f'{dim}l{dim}':
            new_d = change['new'] / getattr(self, f'm{dim}')
            if new_d != getattr(self, f'd{dim}'):
                setattr(self, f'd{dim}', new_d)

    @traitlets.observe(
        'nclx1', 'nclxn', 'nclxS1', 'nclxSn',
        'ncly1', 'nclyn', 'nclyS1', 'nclySn',
        'nclz1', 'nclzn', 'nclzS1', 'nclzSn'
    )
    def _observe_bc(self, change):
        #
        dim = change['name'][3] # It will be x, y or z
        #
        if change['new'] == 0:
            for i in f'ncl{dim}1 ncl{dim}n ncl{dim}S1 ncl{dim}Sn'.split():
                setattr(self, i, 0)
            setattr(self, f'ncl{dim}', True)
        if change['old'] == 0 and change['new'] != 0:
            for i in f'ncl{dim}1 ncl{dim}n ncl{dim}S1 ncl{dim}Sn'.split():
                setattr(self, i, change['new'])
            setattr(self, f'ncl{dim}', False)

    @traitlets.observe('nclx', 'ncly', 'nclz')
    def _observe_periodicity(self, change):
        #
        dim = change['name'][-1] # It will be x, y or z
        #
        if change['new']:
            tmp = getattr(self, f'n{dim}') - 1
            setattr(self, f'_possible_mesh_{dim}', possible_mesh_p)
            setattr(self, f'n{dim}', tmp)
        else:
            tmp = getattr(self, f'n{dim}') + 1
            setattr(self, f'_possible_mesh_{dim}', possible_mesh)
            setattr(self, f'n{dim}', tmp)

    @traitlets.observe('p_row', 'p_col', 'ncores')
    def _observe_2Decomp(self, change):
        if change['name'] == 'ncores':
            possible = list(divisorGenerator(change['new']))
            self._possible_p_row = possible
            self._possible_p_col = possible
            self.p_row, self.p_col = 0, 0
        elif change['name'] == 'p_row':
            try:
                self.p_col = self.ncores // self.p_row
            except:
                self.p_col = 0
        elif change['name'] == 'p_col':
            try:
                self.p_row = self.ncores // self.p_col
            except:
                self.p_row = 0

    @traitlets.observe('ilesmod')
    def _observe_ilesmod(self, change):
        if change['new'] == 0:
            self.nu0nu, self.cnu, self.isecondder = 4.0, 0.44, 4
            self.trait_metadata('nu0nu', 'widget').disabled = True
            self.trait_metadata('cnu', 'widget').disabled = True
            self.trait_metadata('isecondder', 'widget').disabled = True
        else:
            self.trait_metadata('nu0nu', 'widget').disabled = False
            self.trait_metadata('cnu', 'widget').disabled = False
            self.trait_metadata('isecondder', 'widget').disabled = False


    @traitlets.observe('numscalar')
    def _observe_numscalar(self, change):
        self.iscalar = True if change['new'] == 0 else False

    @traitlets.observe('numscalar', 'nx', 'ny', 'nz', 'nvisu', 'icheckpoint',
        'ioutput', 'iprocessing', 'ilast')
    def _observe_size_in_disc(self, change):

        def convert_bytes(num):
            """
            this function will convert bytes to MB.... GB... etc
            """
            step_unit = 1000.0 #1024 bad the size

            for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
                if num < step_unit:
                    return "%3.1f %s" % (num, x)
                num /= step_unit

        prec = 4 if param['mytype'] == np.float32 else 8

        # Restart Size from tools.f90
        count = 3 + self.numscalar #ux, uy, uz, phi
        # Previous time-step if necessary
        if self.itimescheme in [3, 7]:
            count *= 3
        elif self.itimescheme == 2:
            count *= 2
        count  += 1 #pp
        count *= self.nx*self.ny*self.nz*prec*(self.ilast//self.icheckpoint - 1)

        # 3D from visu.f90: ux, uy, uz, pp and phi
        count += (4 + self.numscalar)*self.nx*self.ny*self.nz*prec*self.ilast//self.ioutput

        # 2D planes from BC.Sandbox.f90
        if self.itype == 10:
            # xy planes avg and central plane for ux, uy, uz and phi
            count += 2*(3 + self.numscalar)*self.nx*self.ny*prec*self.ilast//self.iprocessing
            # xz planes avg, top and bot for ux, uy, uz and phi
            count += 3*(3 + self.numscalar)*self.nx*self.nz*prec*self.ilast//self.iprocessing

        self._size_in_disc = convert_bytes(count)

    def _class_to_dict(self):
        for name in self.trait_names():
            group = self.trait_metadata(name, 'group')
            if group != None:
                if group not in self.i3d.keys():
                    self.i3d[group] = {}
                self.i3d[group][name] = getattr(self, name)

    def _dict_to_class(self):

        # Boundary conditions are high priority in order to avoid bugs
        for bc in 'nclx1 nclxn ncly1 nclyn nclz1 nclzn'.split():
            if bc in self.i3d['BasicParam']:
                setattr(self, bc, self.i3d['BasicParam'][bc])

        for name in self.trait_names():
            try:
                group = self.trait_metadata(name, 'group')
                setattr(self, name, self.i3d[group][name])
            except:
                #print(f'{name} not in dictionary')
                pass

    def read(self):
        self.i3d = i3d_to_dict(self.filename)
        self._dict_to_class()

    def write(self):
        self._class_to_dict()
        dict_to_i3d(self.i3d, self.filename)

    def link_widgets(self, silence=True):
        # Create two-way link between variable and widget
        for name in self.trait_names():
            try:
                traitlets.link((self, name), (self.trait_metadata(name, 'widget'), 'value'))
            except:
                if not silence:
                    print(f'Widget not linked for {name}')

        #
        # for dim in ['x', 'y', 'z']:
        #     traitlets.link(
        #         (self, f'_possible_mesh_{dim}'), (self.trait_metadata(f'n{dim}', 'widget'), 'options')
        #     )
        # for name in ['p_row', 'p_col']:
        #     traitlets.link(
        #         (self, f'_possible_{name}'), (self.trait_metadata(f'{name}', 'widget'), 'options')
        #     )
        #
        for name in self.trait_names():
            if name == 'numscalar':
                continue
            group = self.trait_metadata(name, 'group')
            if group == 'ScalarParam':
                try:
                    traitlets.link((self, 'iscalar'), (self.trait_metadata(name, 'widget'), 'disabled'))
                except:
                    if not silence:
                        print(f'Widget not linked to numscalar for {name}')
        # Try adding a description
        for name in self.trait_names():
            if name in description:
                try:
                    self.trait_metadata(name, 'widget').description_tooltip = description[name]
                except:
                    pass

    def get_mesh(self):
        return get_mesh()

def _validate_mesh(n, ncl, ncl1, ncln, dim):

    pmin = 8 if ncl else 9

    if n < pmin:
        # Because of the derivatives' stencil
        raise traitlets.TraitError(
            f'{n} is invalid, n{dim} must be larger than {pmin}')

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
        raise traitlets.TraitError(f'Invalid value for mesh points (n{dim})')

description = dict(
    # Basic Parameters
    p_row = 'Domain decomposition for parallel computation',
    p_col = 'Domain decomposition for parallel computation',
    itype = 'Flow configuration (Taylor-Green Vortex,  Flow around a Cylinder...)',
    nx = 'Number of mesh points in x direction',
    ny = 'Number of mesh points in y direction',
    nz = 'Number of mesh points in z direction',
    xlx = 'Domain size in x direction',
    yly = 'Domain size in y direction',
    zlz = 'Domain size in z direction',
    nclx1 = 'Velocidy boundary condition at begin of x direction',
    nclxn = 'Velocidy boundary condition at end of x direction',
    ncly1 = 'Velocidy boundary condition at begin of y direction',
    nclyn = 'Velocidy boundary condition at end of y direction',
    nclz1 = 'Velocidy boundary condition at begin of z direction',
    nclzn = 'Velocidy boundary condition at end of z direction',
    istret = 'Mesh refinement in y direction at certain location',
    beta = 'Refinement parameter',
    iin = 'Defines pertubation at initial condition',
    re = 'Reynolds number',
    init_noise = 'Value to initial noise, turbulence intensity',
    inflow_noise = 'Random amplitude value at inflow boundary, turbulence intensity',
    dt = 'Value to time step',
    ifirst = 'Value to first iteration',
    ilast = 'Value to last iteration',
    ilesmod = 'Enables Large-Eddy methodologies',
    iscalar = 'Enables scalar fields',
    numscalar = 'Number of scalar fractions',
    iibm = 'Immersed boundary configuration for velocity',
    ilmn = 'Enables Low Mach Number methodology (compressible flows)',
    ivisu = 'Enable store snapshots',
    ipost = ' Enbalbes online postprocessing',
    gravx = 'Value to x component in gravity unitary vector',
    gravy = 'Value to y component in gravity unitary vector',
    gravz = 'Value to z component in gravity unitary vector',
    # Numeric Options
    ifirstder = 'Scheme for first order derivative',
    isecondder = 'Scheme for second order derivative',
    itimescheme = 'Scheme for time integration',
    nu0nu = 'Ratio between hyperviscosity/viscosity at nu (dissipation factor intensity)',
    cnu = 'Ratio between hypervisvosity at km=2/3π and kc=π (dissipation factor range)',
    # In/Out Parameters
    irestart = 'Flag to read initial flow field',
    icheckpoint = 'Frequency for writing backup file',
    ioutput =  'Frequency for visualization file',
    nvisu = 'Size for visual collection',
    iprocessing = 'Frequency for online postprocessing',
    # Statistics
    spinup_time = 'Time in seconds after which statistics are collected',
    nstat = 'Statistic array size',
    wrotation = 'Rotation speed to trigger turbulence',
    # Scalar Parameters
    nclxS1 = 'Scalar boundary condition at begin of x direction',
    nclxSn = 'Scalar boundary condition at end of x direction',
    nclyS1 = 'Scalar boundary condition at begin of y direction',
    nclySn = 'Scalar boundary condition at end of y direction',
    nclzS1 = 'Scalar boundary condition at begin of z direction',
    nclzSn = 'Scalar boundary condition at end of z direction',
    sc = 'Schmidt number(s)',
    ri = 'Richardson number(s)',
    uset = 'Settling velocity(ies)',
    cp = 'Initial concentration(s)',
    iibmS = 'Immersed boundary configuration for scalar (alpha version)',
    scalar_lbound = 'Lower scalar bound',
    scalar_ubound = 'Upper scalar bound',
    # LES Model
    jles = 'LES model',
    smagcst = 'Value to Smagorinsky constant',
    #IBM stuff
    nraf = 'Refinement constant which each axys will be multiplicated',
    nobjmax = 'Maximum number of objects in any direction',
    # Force balance
    iforces = 'Flag to drag and sustentation coefficients',
    nvol = 'Number of volumes for computing force balance',
    xld = 'Volume left bound',
    xrd = 'Volume right bound',
    yld = 'Volume lower bound',
    yud = 'Volume upper bound',
)
