# -*- coding: utf-8 -*-
"""This module generates all the files necessary for our
customized Immersed Boundary Method, based on Lagrange
reconstructions. It is an adaptation to Python from the
original Fortran code and methods from:

* Gautier R., Laizet S. & Lamballais E., 2014, A DNS study of
  jet control with microjets using an alternating direction forcing
  strategy, Int. J. of Computational Fluid Dynamics, 28, 393--410.

:obj:`gene_epsi_3D` is powered by `Numba`_, it translates Python functions to
optimized machine code at runtime. Numba-compiled numerical algorithms in Python
can approach the speeds of C or FORTRAN.

.. _Numba:
    http://numba.pydata.org/

"""

import os.path

import numba
import numpy as np
import xarray as xr


def gene_epsi_3D(epsi_in_dict, prm):
    """This function generates all the auxiliar files necessary for our
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
    >>> dataset = x3d.gene_epsi_3D(epsi, prm)

    Remember to set the number of objects after that if ``prm.iibm >= 2``:

    >>> if prm.iibm >= 2:
    ...     prm.nobjmax = dataset.obj.size

    """

    def obj_count(dataArray, dim):
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
            dataArray,
            input_core_dims=[[dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int64],
        )

    def get_boundaries(dataArray, dim, max_obj, length):
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
            dataArray,
            dataArray[dim],
            input_core_dims=[[dim], [dim]],
            output_core_dims=[["obj"], ["obj"]],
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
                    if (
                        idebraf != 0
                        and ifinraf != 0
                        and idebraf < ifinraf
                        and iflu == 1
                    ):
                        iobj += 1
                        for ii in range(iobj, nobjmax - 1):
                            xi[ii] = xi[ii + 1]
                            xf[ii] = xf[ii + 1]
                        iobj -= 1
                    if (
                        idebraf != 0
                        and ifinraf != 0
                        and idebraf > ifinraf
                        and isol == 1
                    ):
                        iobj += 1
                        for ii in range(iobj, nobjmax - 1):
                            xi[ii] = xi[ii + 1]
                        iobj -= 1
                        for i in range(iobj, nobjmax - 1):
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
            kwargs=dict(nraf=nraf, nobjmax=nobjmax),
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

    # Dask cannot go further
    epsi = epsi_in_dict["epsi"].compute()
    xepsi = epsi_in_dict["xepsi"].compute()
    yepsi = epsi_in_dict["yepsi"].compute()
    zepsi = epsi_in_dict["zepsi"].compute()

    ds = epsi.to_dataset(name="epsi")

    for dir, ep in zip(["x", "y", "z"], [xepsi, yepsi, zepsi]):

        ds[f"nobj_{dir}"] = obj_count(epsi, dir)
        ds[f"nobjmax_{dir}"] = ds[f"nobj_{dir}"].max()
        ds[f"nobjraf_{dir}"] = obj_count(ep, dir)
        ds[f"nobjmaxraf_{dir}"] = ds[f"nobjraf_{dir}"].max()

        ds[f"ibug_{dir}"] = (
            xr.zeros_like(ds[f"nobj_{dir}"])
            .where(ds[f"nobj_{dir}"] == ds[f"nobjraf_{dir}"], 1)
            .sum()
        )

        print(f"{dir}")
        print(f'       nobjraf : {ds[f"nobjmax_{dir}"].values}')
        print(f'    nobjmaxraf : {ds[f"nobjmaxraf_{dir}"].values}')
        print(f'           bug : {ds[f"ibug_{dir}"].values}\n')

    max_obj = np.max([ds.nobjmax_x.values, ds.nobjmax_y.values, ds.nobjmax_z.values])
    ds = ds.assign_coords(obj=range(max_obj), obj_aux=range(-1, max_obj))

    for dir, ep, l in zip(
        ["x", "y", "z"], [xepsi, yepsi, zepsi], [prm.xlx, prm.yly, prm.zlz]
    ):

        ds[f"xi_{dir}"], ds[f"xf_{dir}"] = get_boundaries(ep, dir, max_obj, l)

        if ds[f"ibug_{dir}"] != 0:
            ds[f"xi_{dir}"], ds[f"xf_{dir}"] = fix_bug(
                ds[f"xi_{dir}"],
                ds[f"xf_{dir}"],
                nraf,
                int(max_obj),
                ds[f"nobj_{dir}"],
                ds[f"nobjmaxraf_{dir}"],
                epsi,
                ep.rename(**{dir: dir + "_raf"}),
                dir,
            )

        ds[f"nxipif_{dir}"], ds[f"nxfpif_{dir}"], ising = verif_epsi(epsi, dir)

        print(
            f"number of points with potential problem in {dir} : {ising.sum().values}"
        )

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
        with open(
            os.path.join(data_path, f"n{dim}ifpif.dat"), "w", newline="\n"
        ) as file:
            for value1, value2 in zip(_array1, _array2):
                file.write(f"{value1:12d}{value2:12d}\n")

    def write_xixf(array1, array2, dim) -> None:
        _array1 = transpose_n_flatten(array1)
        _array2 = transpose_n_flatten(array2)
        with open(
            os.path.join(data_path, f"{dim}i{dim}f.dat"), "w", newline="\n"
        ) as file:
            for value1, value2 in zip(_array1, _array2):
                file.write(f"{value1:24.16E}{value2:24.16E}\n")

    def transpose_n_flatten(array):
        if len(array.coords) == 3:
            return array.values.transpose(1, 0, 2).flatten()
        return array.values.T.flatten()

    print("\nWriting...")
    data_path = os.path.join(prm.dataset.data_path, "geometry")
    prm.dataset.write(ds["epsi"])
    for dir in ["x", "y", "z"]:
        write_nobj(ds[f"nobj_{dir}"], dir)
        write_nxipif(ds[f"nxipif_{dir}"], ds[f"nxfpif_{dir}"], dir)
        write_xixf(ds[f"xi_{dir}"], ds[f"xf_{dir}"], dir)
