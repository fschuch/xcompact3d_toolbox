"""This module generates all the files necessary for our
customized Immersed Boundary Method, based on Lagrange
reconstructions. It is an adaptation to Python from the
original Fortran code and methods from:

* Gautier R., Laizet S. & Lamballais E., 2014, A DNS study of
  jet control with microjets using an alternating direction forcing
  strategy, Int. J. of Computational Fluid Dynamics, 28, 393--410.

:obj:`gene_epsi_3d` is powered by `Numba`_, it translates Python functions to
optimized machine code at runtime. Numba-compiled numerical algorithms in Python
can approach the speeds of C or FORTRAN.

.. _Numba:
    http://numba.pydata.org/

"""

import os.path

import numba
import numpy as np
import xarray as xr
from loguru import logger


def gene_epsi_3d(epsi_in_dict, prm):
    """This function generates all the Auxiliary files necessary for our
    customize IBM, based on Lagrange reconstructions. The arrays can be
    initialized with :obj:`xcompact3d_toolbox.sandbox.init_epsi()`, then,
    some standard geometries are provided by the accessor
    :obj:`xcompact3d_toolbox.sandbox.Geometry`.
    Notice that you can apply our own routines for your own objects.
    The main outputs of the function are written to disc at the files
    ``epsilon.bin``, ``nobjx.dat``, ``nobjy.dat``, ``nobjz.dat``,
    ``nxifpif.dat``, ``nyifpif.dat``, ``nzifpif.dat``, ``xixf.dat``,
    ``yiyf.dat`` and ``zizf.dat``.
    They will be used by Xcompact3d and the sandbox
    flow configuration.

    Parameters
    ----------
    epsi_in_dict : :obj:`dict` of :obj:`xarray.DataArray`
        A dictionary containing the epsi(s) array(s).
    prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
        Contains the computational and physical parameters.

    Returns
    -------
    :obj:`xarray.Dataset` or :obj:`None`
        All computed variables are returned in a Dataset if ``prm.iibm >= 2``,
        but just for reference, since all the relevant values are written
        to the disc.

    Examples
    -------

    >>> prm = x3d.Parameters()
    >>> epsi = x3d.sandbox.init_epsi(prm)
    >>> for key in epsi.keys():
    ...     epsi[key] = epsi[key].geo.cylinder(x=4, y=5)
    >>> dataset = x3d.gene_epsi_3d(epsi, prm)

    Remember to set the number of objects after that if ``prm.iibm >= 2``:

    >>> if prm.iibm >= 2:
    ...     prm.nobjmax = dataset.obj.size

    """

    def obj_count(data_array, dim):
        """Counts the number of objects in a given direction"""

        @numba.jit
        def count(array):
            nobj = np.int64(array[0])
            for i in range(array.size - 1):
                if not array[i] and array[i + 1]:
                    nobj += 1
            return nobj

        return xr.apply_ufunc(
            count,
            data_array,
            input_core_dims=[[dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int64],
        )

    def get_boundaries(data_array, dim, max_obj, length):
        """Gets the boundaries in a given direction"""

        @numba.jit
        def pos(array, x):
            xi = np.zeros((max_obj), dtype=np.float64)
            xf = np.zeros_like(xi)

            inum = 0
            if array[0]:
                xi[inum] = -x[1]
            for i in range(array.size - 1):
                if not array[i] and array[i + 1]:
                    xi[inum] = (x[i] + x[i + 1]) / 2.0
                elif array[i] and not array[i + 1]:
                    xf[inum] = (x[i] + x[i + 1]) / 2.0
                    inum += 1
            if array[-1]:
                xf[inum] = length + (x[-1] - x[-2]) / 2.0

            return xi, xf

        return xr.apply_ufunc(
            pos,
            data_array,
            data_array[dim],
            input_core_dims=[[dim], [dim]],
            output_core_dims=[["obj"], ["obj"]],
            dask_gufunc_kwargs={"output_sizes": {"obj": max_obj}},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64, np.float64],
        )

    def fix_bug(xi, xf, nraf, nobjmax, nobjx, nobjxraf, epsi, refepsi, dim):
        @numba.njit
        def fix(xi_in, xf_in, nobjx, nobjxraf, epsi, refepsi, nraf, nobjmax):
            xi = xi_in.copy()
            xf = xf_in.copy()
            if nobjx != nobjxraf:
                iobj = -1
                isol = 0
                if epsi[0]:
                    iobj += 1
                for i in range(epsi.size - 1):
                    idebraf = 0
                    ifinraf = 0
                    iflu = 0
                    if not epsi[i] and epsi[i + 1]:
                        iobj += 1
                    if not epsi[i] and not epsi[i + 1]:
                        iflu = 1
                    if epsi[i] and epsi[i + 1]:
                        isol = 1
                    for iraf in range(nraf):
                        iiraf = iraf + nraf * i
                        if not refepsi[iiraf] and refepsi[iiraf + 1]:
                            idebraf = iiraf + 1
                        if refepsi[iiraf] and not refepsi[iiraf + 1]:
                            ifinraf = iiraf + 1
                    if idebraf != 0 and ifinraf != 0 and idebraf < ifinraf and iflu == 1:
                        iobj += 1
                        for ii in range(iobj, nobjmax - 1):
                            xi[ii] = xi[ii + 1]
                            xf[ii] = xf[ii + 1]
                        iobj -= 1
                    if idebraf != 0 and ifinraf != 0 and idebraf > ifinraf and isol == 1:
                        iobj += 1
                        for ii in range(iobj, nobjmax - 1):
                            xi[ii] = xi[ii + 1]
                        iobj -= 1
                        for ii in range(iobj, nobjmax - 1):
                            xf[ii] = xf[ii + 1]

            return xi, xf

        return xr.apply_ufunc(
            fix,
            xi,
            xf,
            nobjx,
            nobjxraf,
            epsi,
            refepsi,
            kwargs={"nraf": nraf, "nobjmax": nobjmax},
            input_core_dims=[["obj"], ["obj"], [], [], [dim], [dim + "_raf"]],
            output_core_dims=[["obj"], ["obj"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64, np.float64],
        )

    def verif_epsi(epsi, dim):
        @numba.jit
        def verif(epsi):
            nxipif = npif * np.ones((max_obj + 1), dtype=np.int64)
            nxfpif = nxipif.copy()
            ising, inum, iflu = 0, 0, 0
            if epsi[0]:
                inum += 1
            else:
                iflu += 1
            for i in range(1, epsi.size):
                if not epsi[i]:
                    iflu += 1
                if (not epsi[i - 1]) and (epsi[i]):
                    inum += 1
                    if inum == 0:
                        nxipif[inum] = iflu - izap
                        if iflu - izap < npif:
                            ising += 1
                        else:
                            nxipif[inum] = npif
                        iflu = 0
                    else:
                        nxipif[inum] = iflu - izap
                        nxfpif[inum - 1] = iflu - izap
                        if iflu - izap < npif:
                            ising += 1
                        else:
                            nxipif[inum] = npif
                            nxfpif[inum - 1] = npif
                        iflu = 0
                if epsi[i]:
                    iflu = 0
            if not epsi[-1]:
                nxfpif[inum] = iflu - izap
                if iflu - izap < npif:
                    ising += 1
                    nxfpif[inum] = npif

            return nxipif, nxfpif, np.array([ising], dtype=np.int64)

        return xr.apply_ufunc(
            verif,
            epsi,
            input_core_dims=[[dim]],
            output_core_dims=[["obj_aux"], ["obj_aux"], ["c"]],
            dask_gufunc_kwargs={"output_sizes": {"obj_aux": max_obj + 1, "c": 1}},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int64, np.int64, np.int64],
        )

    if prm.iibm <= 1:
        prm.dataset.write(epsi_in_dict["epsi"])
        return None

    izap = prm.izap
    npif = prm.npif
    nraf = prm.nraf

    epsi = epsi_in_dict["epsi"]
    xepsi = epsi_in_dict["xepsi"]
    yepsi = epsi_in_dict["yepsi"]
    zepsi = epsi_in_dict["zepsi"]

    ds = epsi.to_dataset(name="epsi")

    for direction, ep in zip(["x", "y", "z"], [xepsi, yepsi, zepsi]):
        ds[f"nobj_{direction}"] = obj_count(epsi, direction)
        ds[f"nobjmax_{direction}"] = ds[f"nobj_{direction}"].max()
        ds[f"nobjraf_{direction}"] = obj_count(ep, direction)
        ds[f"nobjmaxraf_{direction}"] = ds[f"nobjraf_{direction}"].max()

        ds[f"ibug_{direction}"] = (
            xr.zeros_like(ds[f"nobj_{direction}"]).where(ds[f"nobj_{direction}"] == ds[f"nobjraf_{direction}"], 1).sum()
        )

        logger.debug(
            f"{direction}\n"
            f"       nobjraf : {ds[f'nobjmax_{direction}'].values}\n"
            f"    nobjmaxraf : {ds[f'nobjmaxraf_{direction}'].values}\n"
            f"           bug : {ds[f'ibug_{direction}'].values}\n"
        )

    max_obj = np.max([ds.nobjmax_x.values, ds.nobjmax_y.values, ds.nobjmax_z.values])
    ds = ds.assign_coords(obj=range(max_obj), obj_aux=range(-1, max_obj))

    for direction, ep, length in zip(["x", "y", "z"], [xepsi, yepsi, zepsi], [prm.xlx, prm.yly, prm.zlz]):
        ds[f"xi_{direction}"], ds[f"xf_{direction}"] = get_boundaries(ep, direction, max_obj, length)

        if ds[f"ibug_{direction}"] != 0:
            ds[f"xi_{direction}"], ds[f"xf_{direction}"] = fix_bug(
                ds[f"xi_{direction}"],
                ds[f"xf_{direction}"],
                nraf,
                int(max_obj),
                ds[f"nobj_{direction}"],
                ds[f"nobjmaxraf_{direction}"],
                epsi,
                ep.rename(**{direction: direction + "_raf"}),
                direction,
            )

        ds[f"nxipif_{direction}"], ds[f"nxfpif_{direction}"], ising = verif_epsi(epsi, direction)

        logger.debug(f"number of points with potential problem in {direction} : {ising.sum().values}")

    write_geomcomplex(prm, ds)

    return ds


def write_geomcomplex(prm, ds) -> None:
    def write_nobj(array, dim) -> None:
        with open(os.path.join(data_path, f"nobj{dim}.dat"), "w", newline="\n") as file:
            for value in transpose_n_flatten(array):
                file.write(f"{value:12d}\n")

    def write_nxipif(array1, array2, dim) -> None:
        _array1 = transpose_n_flatten(array1)
        _array2 = transpose_n_flatten(array2)
        with open(os.path.join(data_path, f"n{dim}ifpif.dat"), "w", newline="\n") as file:
            for value1, value2 in zip(_array1, _array2):
                file.write(f"{value1:12d}{value2:12d}\n")

    def write_xixf(array1, array2, dim) -> None:
        _array1 = transpose_n_flatten(array1)
        _array2 = transpose_n_flatten(array2)
        with open(os.path.join(data_path, f"{dim}i{dim}f.dat"), "w", newline="\n") as file:
            for value1, value2 in zip(_array1, _array2):
                file.write(f"{value1:24.16E}{value2:24.16E}\n")

    def transpose_n_flatten(array):
        if len(array.coords) == 3:  # noqa: PLR2004
            return array.values.transpose(1, 0, 2).flatten()
        return array.values.T.flatten()

    data_path = os.path.join(prm.dataset.data_path, "geometry")
    prm.dataset.write(ds["epsi"])
    for direction in ["x", "y", "z"]:
        write_nobj(ds[f"nobj_{direction}"], direction)
        write_nxipif(ds[f"nxipif_{direction}"], ds[f"nxfpif_{direction}"], direction)
        write_xixf(ds[f"xi_{direction}"], ds[f"xf_{direction}"], direction)
