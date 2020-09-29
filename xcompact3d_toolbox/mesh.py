from .param import param
import numpy as np


def get_mesh(prm, raf=False, staggered=False):
    #
    dim = ["x", "y", "z"]
    #
    n = {d: getattr(prm, f"n{d}") for d in dim}
    m = {d: getattr(prm, f"_m{d}") for d in dim}
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
        d: np.linspace(
            start=0.0, stop=l[d], num=n[d], endpoint=not ncl[d], dtype=param["mytype"]
        )
        for d in dim
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
                dtype=param["mytype"],
            )
    #
    # stretching
    #
    if prm.istret != 0:
        #
        istret = prm.istret
        beta = prm.beta
        yp = np.zeros_like(mesh["y"])
        yeta = np.zeros_like(mesh["y"])
        #
        if staggered:
            raise NotImplementedError("Unsupported: Not prepared yet for istret != 0")

        yinf = -0.5 * prm.yly
        den = 2.0 * beta * yinf
        xnum = -yinf - np.sqrt(np.pi * np.pi * beta * beta + yinf * yinf)
        alpha = np.abs(xnum / den)
        xcx = 1.0 / beta / alpha
        if alpha != 0.0:
            if istret == 1:
                yp[0] = 0.0
            if istret == 2:
                yp[0] = 0.0
            if istret == 1:
                yeta[0] = 0.0
            if istret == 2:
                yeta[0] = -0.5
            if istret == 3:
                yp[0] = 0.0
            if istret == 3:
                yeta[0] = -0.5
            for j in range(1, prm.ny):
                if istret == 1:
                    yeta[j] = j / prm._my
                if istret == 2:
                    yeta[j] = j / prm._my - 0.5
                if istret == 3:
                    yeta[j] = 0.5 * j / prm._my - 0.5
                den1 = np.sqrt(alpha * beta + 1.0)
                xnum = den1 / np.sqrt(alpha / np.pi) / np.sqrt(beta) / np.sqrt(np.pi)
                den = (
                    2.0
                    * np.sqrt(alpha / np.pi)
                    * np.sqrt(beta)
                    * np.pi
                    * np.sqrt(np.pi)
                )
                den3 = (
                    (np.sin(np.pi * yeta[j])) * (np.sin(np.pi * yeta[j])) / beta / np.pi
                ) + alpha / np.pi
                den4 = 2.0 * alpha * beta - np.cos(2.0 * np.pi * yeta[j]) + 1.0
                xnum1 = (
                    (np.arctan(xnum * np.tan(np.pi * yeta[j])))
                    * den4
                    / den1
                    / den3
                    / den
                )
                cst = (
                    np.sqrt(beta)
                    * np.pi
                    / (2.0 * np.sqrt(alpha) * np.sqrt(alpha * beta + 1.0))
                )
                if istret == 1:
                    if yeta[j] < 0.5:
                        yp[j] = xnum1 - cst - yinf
                    if yeta[j] == 0.5:
                        yp[j] = -yinf
                    if yeta[j] > 0.5:
                        yp[j] = xnum1 + cst - yinf
                elif istret == 2:
                    if yeta[j] < 0.5:
                        yp[j] = xnum1 - cst + prm.yly
                    if yeta[j] == 0.5:
                        yp[j] = prm.yly
                    if yeta[j] > 0.5:
                        yp[j] = xnum1 + cst + prm.yly
                elif istret == 3:
                    if yeta[j] < 0.5:
                        yp[j] = (xnum1 - cst + prm.yly) * 2.0
                    if yeta[j] == 0.5:
                        yp[j] = prm.yly * 2.0
                    if yeta[j] > 0.5:
                        yp[j] = (xnum1 + cst + prm.yly) * 2.0
                else:
                    raise NotImplementedError("Unsupported: invalid value for istret")
        if alpha == 0.0:
            yp[0] = -1.0e10
            for j in range(1, prm.ny):
                yeta[j] = j / prm.ny
                yp[j] = -beta * np.cos(np.pi * yeta[j]) / np.sin(yeta[j] * np.pi)

        mesh["y"] = yp

    return mesh
