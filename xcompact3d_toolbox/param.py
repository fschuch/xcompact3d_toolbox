from numpy import float64

param = {"mytype": float64}


def boundary_condition(prm, var=None):

    default = {d: {"ncl1": 2, "ncln": 2, "npaire": 1} for d in ["x", "y", "z"]}
    default["y"]["istret"] = prm.istret
    default["y"]["beta"] = prm.beta

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
                "ncl1": 0 if prm._nclx else 1,
                "ncln": 0 if prm._nclx else 1,
                "npaire": 1,
            },
            "y": {
                "ncl1": 0 if prm._ncly else 1,
                "ncln": 0 if prm._ncly else 1,
                "npaire": 1,
            },
            "z": {
                "ncl1": 0 if prm._nclz else 1,
                "ncln": 0 if prm._nclz else 1,
                "npaire": 1,
            },
        },
    }

    if prm.numscalar > 0:

        BC["phi"] = {
            "x": {"ncl1": prm.nclxS1, "ncln": prm.nclxS1, "npaire": 1},
            "y": {"ncl1": prm.nclyS1, "ncln": prm.nclySn, "npaire": 1},
            "z": {"ncl1": prm.nclzS1, "ncln": prm.nclzSn, "npaire": 1},
        }

    for key, value in BC.items():
        value["y"]["istret"] = prm.istret
        value["y"]["beta"] = prm.beta

    return BC.get(var.lower(), default)
