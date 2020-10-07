from .param import param
import numpy as np


def stretching(istret, beta, yly, my, ny):

    yp = np.zeros(ny, dtype=param["mytype"])
    yeta = np.zeros_like(yp)
    ypi = np.zeros_like(yp)
    yetai = np.zeros_like(yp)
    ppy = np.zeros_like(yp)
    pp2y = np.zeros_like(yp)
    pp4y = np.zeros_like(yp)
    ppyi = np.zeros_like(yp)
    pp2yi = np.zeros_like(yp)
    pp4yi = np.zeros_like(yp)
    #
    yinf = -0.5 * yly
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
        for j in range(1, ny):
            if istret == 1:
                yeta[j] = j / my
            if istret == 2:
                yeta[j] = j / my - 0.5
            if istret == 3:
                yeta[j] = 0.5 * j / my - 0.5
            den1 = np.sqrt(alpha * beta + 1.0)
            xnum = den1 / np.sqrt(alpha / np.pi) / np.sqrt(beta) / np.sqrt(np.pi)
            den = 2.0 * np.sqrt(alpha / np.pi) * np.sqrt(beta) * np.pi * np.sqrt(np.pi)
            den3 = (
                (np.sin(np.pi * yeta[j])) * (np.sin(np.pi * yeta[j])) / beta / np.pi
            ) + alpha / np.pi
            den4 = 2.0 * alpha * beta - np.cos(2.0 * np.pi * yeta[j]) + 1.0
            xnum1 = (
                (np.arctan(xnum * np.tan(np.pi * yeta[j]))) * den4 / den1 / den3 / den
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
                    yp[j] = xnum1 - cst + yly
                if yeta[j] == 0.5:
                    yp[j] = yly
                if yeta[j] > 0.5:
                    yp[j] = xnum1 + cst + yly
            elif istret == 3:
                if yeta[j] < 0.5:
                    yp[j] = (xnum1 - cst + yly) * 2.0
                if yeta[j] == 0.5:
                    yp[j] = yly * 2.0
                if yeta[j] > 0.5:
                    yp[j] = (xnum1 + cst + yly) * 2.0
            else:
                raise NotImplementedError("Unsupported: invalid value for istret")
    if alpha == 0.0:
        yp[0] = -1.0e10
        for j in range(1, ny):
            yeta[j] = j / ny
            yp[j] = -beta * np.cos(np.pi * yeta[j]) / np.sin(yeta[j] * np.pi)

    if alpha != 0.0:
        for j in range(ny):
            if istret == 1:
                yetai[j] = (j + 0.5) * (1.0 / my)
            if istret == 2:
                yetai[j] = (j + 0.5) * (1.0 / my) - 0.5
            if istret == 3:
                yetai[j] = (j + 0.5) * (0.5 / my) - 0.5
            den1 = np.sqrt(alpha * beta + 1.0)
            xnum = den1 / np.sqrt(alpha / np.pi) / np.sqrt(beta) / np.sqrt(np.pi)
            den = 2.0 * np.sqrt(alpha / np.pi) * np.sqrt(beta) * np.pi * np.sqrt(np.pi)
            den3 = (
                (np.sin(np.pi * yetai[j])) * (np.sin(np.pi * yetai[j])) / beta / np.pi
            ) + alpha / np.pi
            den4 = 2.0 * alpha * beta - np.cos(2.0 * np.pi * yetai[j]) + 1.0
            xnum1 = (
                (np.arctan(xnum * np.tan(np.pi * yetai[j]))) * den4 / den1 / den3 / den
            )
            cst = (
                np.sqrt(beta)
                * np.pi
                / (2.0 * np.sqrt(alpha) * np.sqrt(alpha * beta + 1.0))
            )
            if istret == 1:
                if yetai[j] < 0.5:
                    ypi[j] = xnum1 - cst - yinf
                elif yetai[j] == 0.5:
                    ypi[j] = 0.0 - yinf
                elif yetai[j] > 0.5:
                    ypi[j] = xnum1 + cst - yinf
            elif istret == 2:
                if yetai[j] < 0.5:
                    ypi[j] = xnum1 - cst + yly
                elif yetai[j] == 0.5:
                    ypi[j] = 0.0 + yly
                elif yetai[j] > 0.5:
                    ypi[j] = xnum1 + cst + yly
            elif istret == 3:
                if yetai[j] < 0.5:
                    ypi[j] = (xnum1 - cst + yly) * 2.0
                elif yetai[j] == 0.5:
                    ypi[j] = (0.0 + yly) * 2.0
                elif yetai[j] > 0.5:
                    ypi[j] = (xnum1 + cst + yly) * 2.0

    if alpha == 0.0:
        ypi[0] = -1e10
        for j in range(1, ny):
            yetai[j] = j * (1.0 / ny)
            ypi[j] = -beta * np.cos(np.pi * yetai[j]) / np.sin(yetai[j] * np.pi)

    # Mapping!!, metric terms
    if istret != 3:
        for j in range(ny):
            ppy[j] = yly * (
                alpha / np.pi
                + (1.0 / np.pi / beta)
                * np.sin(np.pi * yeta[j])
                * np.sin(np.pi * yeta[j])
            )
            pp2y[j] = ppy[j] * ppy[j]
            pp4y[j] = -2.0 / beta * np.cos(np.pi * yeta[j]) * np.sin(np.pi * yeta[j])
        for j in range(ny):
            ppyi[j] = yly * (
                alpha / np.pi
                + (1.0 / np.pi / beta)
                * np.sin(np.pi * yetai[j])
                * np.sin(np.pi * yetai[j])
            )
            pp2yi[j] = ppyi[j] * ppyi[j]
            pp4yi[j] = -2.0 / beta * np.cos(np.pi * yetai[j]) * np.sin(np.pi * yetai[j])

    if istret == 3:
        for j in range(ny):
            ppy[j] = yly * (
                alpha / np.pi
                + (1.0 / np.pi / beta)
                * np.sin(np.pi * yeta[j])
                * np.sin(np.pi * yeta[j])
            )
            pp2y[j] = ppy[j] * ppy[j]
            pp4y[j] = (
                -2.0 / beta * np.cos(np.pi * yeta[j]) * np.sin(np.pi * yeta[j])
            ) / 2.0
        for j in range(ny):
            ppyi[j] = yly * (
                alpha / np.pi
                + (1.0 / np.pi / beta)
                * np.sin(np.pi * yetai[j])
                * np.sin(np.pi * yetai[j])
            )
            pp2yi[j] = ppyi[j] * ppyi[j]
            pp4yi[j] = (
                -2.0 / beta * np.cos(np.pi * yetai[j]) * np.sin(np.pi * yetai[j])
            ) / 2.0

    return yp, ppy, pp2y, pp4y


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
        mesh["y"] = stretching(prm.istret, prm.beta, prm.yly, prm._my, prm.ny)[0]

    return mesh
