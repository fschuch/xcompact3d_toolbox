from .param import mytype
import numpy as np


def get_mesh(prm, raf=False, staggered=False):
    #
    dim = ["x", "y", "z"]
    #
    n = {d: getattr(prm, f"n{d}") for d in dim}
    m = n.copy()
    #
    l = {d: getattr(prm, f"{d}l{d}") for d in dim}
    #
    # BC, True if periodic, false otherwise
    ncl = {d: getattr(prm, f"_ncl{d}") for d in dim}
    #
    if prm.iibm == 2 and raf:
        for d in dim:
            n[d] = int(m[d] * prm.nraf + 1)
    # Mesh
    mesh = {
        d: np.linspace(start=0.0, stop=l[d], num=n[d], endpoint=not ncl[d]) for d in dim
    }
    #
    # Half-staggered for pressure
    #
    if staggered:
        for d in dim:
            delta = mesh[d][1] - mesh[d][0]
            mesh[d] = np.linspace(
                start=0.5 * delta,
                stop=l[d] - 0.5 * delta,
                num=m[d],
                endpoint=not ncl[d],
            )
    # #
    # #stretching
    # #
    # istret = prm.istret
    # beta = prm.beta
    # #
    # if istret != 0:
    #     raise NotImplementedError(
    #             "Unsupported: Not prepared yet for istret != 0")

    return mesh
