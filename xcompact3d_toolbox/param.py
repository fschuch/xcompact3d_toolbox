from numpy import float64

mytype = float64
""":obj:`numpy.float32` or :obj:`numpy.float64`: Defines the float point precision

This precision is applyed for I/O and for initialization at
:obj:`xcompact3d_toolbox.sandbox`.

Use :obj:`numpy.float64` (default) if Xcompact3d was compiled with the flag
``-DDOUBLE_PREC``, or use ``np.float32`` otherwise (including ``-DSAVE_SINGLE``).

"""


def boundary_condition(prm, var=None):

    default = {d: {"ncl1": 2, "ncln": 2, "npaire": 1} for d in ["x", "y", "z"]}

    BC = {
        "ux": {
            "x": {"ncl1": prm.nclx1, "ncln": prm.nclxn, "npaire": 0},
            "y": {"ncl1": prm.ncly1, "ncln": prm.nclyn, "npaire": 1},
            "z": {"ncl1": prm.nclz1, "ncln": prm.nclzn, "npaire": 1},
        },
        "uy": {
            "x": {"ncl1": prm.nclx1, "ncln": prm.nclxn, "npaire": 1},
            "y": {"ncl1": prm.ncly1, "ncln": prm.nclyn, "npaire": 0},
            "z": {"ncl1": prm.nclz1, "ncln": prm.nclzn, "npaire": 1},
        },
        "uz": {
            "x": {"ncl1": prm.nclx1, "ncln": prm.nclxn, "npaire": 1},
            "y": {"ncl1": prm.ncly1, "ncln": prm.nclyn, "npaire": 1},
            "z": {"ncl1": prm.nclz1, "ncln": prm.nclzn, "npaire": 0},
        },
        "pp": {
            "x": {
                "ncl1": 0 if prm.nclx else 1,
                "ncln": 0 if prm.ncln else 1,
                "npaire": 1,
            },
            "y": {
                "ncl1": 0 if prm.ncly else 1,
                "ncln": 0 if prm.ncly else 1,
                "npaire": 1,
            },
            "z": {
                "ncl1": 0 if prm.nclz else 1,
                "ncln": 0 if prm.nclz else 1,
                "npaire": 1,
            },
        },
    }

    if prm.numscalar > 0:

        BC["phi"] = {
            "x": {"ncl1": prm.nclxS1, "ncln": prm.nclnS1, "npaire": 1},
            "y": {"ncl1": prm.nclyS1, "ncln": prm.nclySn, "npaire": 1},
            "z": {"ncl1": prm.nclzS1, "ncln": prm.nclzSn, "npaire": 1},
        }

    return BC.get(var, default)
