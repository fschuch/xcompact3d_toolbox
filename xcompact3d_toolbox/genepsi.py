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

import numba
import numpy as np
import xarray as xr
from .array import X3dDataArray, X3dDataset
from .mesh import get_mesh
from .param import mytype


def gene_epsi_3D(epsi_in_dict, prm):
    """This function generates all the auxiliar files necessary for our
    customize IBM, based on Lagrange reconstructions. The arrays can be
    initialized with :obj:`xcompact3d_toolbox.sandbox.init_epsi()`, then, some
    standard geometries are provided by the accessor
    :obj:`xcompact3d_toolbox.sandbox.geo`.
    Notice that you can apply our own routines for your own objects.
    The main outputs of the function are written to disc at files obj-x.csv,
    obj-y.csv and obj-z.csv, they will be used by Xcompact3d and the sandbox
    flow configuration. The function computes the maximum number of objects
    in a given direction and updates this value at prm, so, make sure to
    write the ``.i3d`` file to disc afterwards.


    Parameters
    ----------
    epsi_in_dict : :obj:`dict` of :obj:`xarray.DataArray`
        A dictionary containing the epsi(s) array(s).
    prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
        Contains the computational and physical parameters.

    Returns
    -------
    :obj:`xarray.Dataset`
        All computed variables are returned in a Data set, but just for
        reference, since all the relevant values are written to the disc.

    Examples
    -------

    >>> prm = x3d.Parameters()
    >>> epsi = x3d.sandbox.init_epsi(prm)
    >>> for key in epsi.keys():
    >>>     epsi[key] = epsi[key].geo.cylinder(x=4, y=5)
    >>> x3d.gene_epsi_3D(epsi, prm)

    """

    def obj_count(da, dim):
        """Counts the number of objects in a given direction"""

        @numba.jit
        def count(data):
            nojb = np.int64(data[0])
            for i in range(data.size - 1):
                if not data[i] and data[i + 1]:
                    nojb += 1
            return nojb

        return xr.apply_ufunc(
            count,
            da,
            input_core_dims=[[dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int64],
        )

    def get_boundaries(epsiraf, dim, max_obj, l):
        """Gets the boundaries in a given direction"""

        @numba.jit
        def pos(epraf, x):
            xi = np.zeros((max_obj), dtype=np.float64)
            xf = np.zeros_like(xi)

            inum = 0
            if epraf[0]:
                xi[inum] = -x[1]
            for i in range(epraf.size - 1):
                if not epraf[i] and epraf[i + 1]:
                    xi[inum] = (x[i] + x[i + 1]) / 2.0
                elif epraf[i] and not epraf[i + 1]:
                    xf[inum] = (x[i] + x[i + 1]) / 2.0
                    inum += 1
            if epraf[-1]:
                xf[inum] = l + (x[-1] - x[-2]) / 2.0

            return xi, xf

        return xr.apply_ufunc(
            pos,
            epsiraf,
            epsiraf[dim],
            input_core_dims=[[dim], [dim]],
            output_core_dims=[["obj"], ["obj"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64, np.float64],
        )

    def fix_bug(xi, xf, nobjx, nobjxraf, epsi, epsiraf, nraf, max_obj, dim):
        @numba.jit
        def fix(xi, xf, nobj, nobjraf, epsi, epsiraf, nraf, max_obj):

            if nobj != nobjraf:
                iobj = -1
                if epsi[0]:
                    iobj += 1
                for i in range(epsi.size - 1):
                    if not epsi[i] and epsi[i + 1]:
                        iobj += 1
                    if not epsi[i] and not epsi[i + 1]:
                        iflu = 1
                    if epsi[i] and epsi[i + 1]:
                        isol = 1
                    for iraf in range(nraf):
                        if (
                            not epsiraf[iraf + nraf * i]
                            and epsiraf[iraf + nraf * i + 1]
                        ):
                            idebraf = iraf + 1 + nraf * i + 1
                        if (
                            epsiraf[iraf + nraf * i]
                            and not epsiraf[iraf + nraf * i + 1]
                        ):
                            ifinraf = iraf + 1 + nraf * i + 1
                    if (
                        idebraf != 0
                        and ifinraf != 0
                        and idebraf < ifinraf
                        and iflu == 1
                    ):
                        iobj += 1
                        for ii in range(iobj, max_obj):
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
                        for ii in range(iobj, max_obj):
                            xi[ii] = xi[ii + 1]
                        iobj -= 1
                        for ii in range(iobj, max_obj):
                            xf[ii] = xf[ii + 1]
                    idebraf, ifinraf, iful = 0, 0, 0

        return xr.apply_ufunc(
            fix,
            epsiraf,
            epsiraf[dim],
            input_core_dims=[[dim], [dim]],
            output_core_dims=[["obj"], ["obj"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64, np.float64],
        )

    def verif_epsi(epsi, dim):
        """Gets the indexes where Lagrangian interpolator starts/ends
           at each side of each object
        """

        @numba.jit
        def verif(epsi):
            nxipif = npif * np.ones((max_obj), dtype=np.int64)
            nxfpif = npif * np.ones_like(nxipif)
            ising, inum, iflu = 0, -1, 0
            if epsi[0]:
                inum += 1
            if not epsi[0]:
                iflu += 1
            for i in range(1, epsi.size):
                if not epsi[i]:
                    iflu += 1
                if not epsi[i - 1] and epsi[i]:
                    inum += 1
                    if inum == 0:
                        nxipif[inum] = iflu - izap
                        if iflu - izap < npif:
                            ising += 1
                        if iflu - izap >= npif:
                            nxipif[inum] = npif
                        iflu = 0
                    else:
                        nxipif[inum] = iflu - izap
                        nxfpif[inum - 1] = iflu - izap
                        if iflu - izap < npif:
                            ising += 1
                        if iflu - izap >= npif:
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
            output_core_dims=[["obj"], ["obj"], ["c"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int64, np.int64, np.int64],
        )

    izap = getattr(prm, "izap", 1)
    npif = getattr(prm, "npif", 2)
    nraf = getattr(prm, "nraf", 10)

    xlx = prm.xlx
    yly = prm.yly
    zlz = prm.zlz

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
        print(f'       nobjraf : {ds["nobjmax_"+dir].values}')
        print(f'    nobjmaxraf : {ds["nobjmaxraf_"+dir].values}')
        print(f'           bug : {ds["ibug_"+dir].values}\n')

    max_obj = np.max([ds.nobjmax_x.values, ds.nobjmax_y.values, ds.nobjmax_z.values])
    prm.nobjmax = int(max_obj)  # using int to be consistent with traitlets types
    ds = ds.assign_coords({"obj": np.arange(0, max_obj)})

    for dir, ep, l in zip(["x", "y", "z"], [xepsi, yepsi, zepsi], [xlx, yly, zlz]):

        ds[f"xi_{dir}"], ds[f"xf_{dir}"] = get_boundaries(ep, dir, max_obj, l)

        if ds[f"ibug_{dir}"] != 0:
            ds[f"xi_{dir}"], ds[f"xf_{dir}"] = fix_bug(
                ds[f"xi_{dir}"],
                ds[f"xf_{dir}"],
                ds[f"nobj_{dir}"],
                ds[f"nobjmaxraf_{dir}"],
                epsi,
                ep,
                nraf,
                max_obj,
                dir,
            )

        ds[f"nxipif_{dir}"], ds[f"nxfpif_{dir}"], ising = verif_epsi(epsi, dir)

        print(
            f"number of points with potential problem in {dir} : {ising.sum().values}"
        )

    print("\nWriting...")

    ds.x3d.write(prm)

    Y, Z = np.meshgrid(ds.y.values, ds.z.values)
    with open(f"./data/geometry/obj-x.csv", "w") as f:
        fmt = (2 + 2 * ds.obj.size) * "%26.16e" + (1 + 2 * ds.obj.size) * "  %06d"
        np.savetxt(
            f,
            np.column_stack(
                (
                    Z.flatten(),
                    Y.flatten(),
                    ds.xi_x.values.transpose(1, 0, 2).reshape(
                        ds.y.size * ds.z.size, ds.obj.size
                    ),
                    ds.xf_x.values.transpose(1, 0, 2).reshape(
                        ds.y.size * ds.z.size, ds.obj.size
                    ),
                    ds.nobj_x.values.T.flatten(),
                    ds.nxipif_x.values.transpose(1, 0, 2).reshape(
                        ds.y.size * ds.z.size, ds.obj.size
                    ),
                    ds.nxfpif_x.values.transpose(1, 0, 2).reshape(
                        ds.y.size * ds.z.size, ds.obj.size
                    ),
                )
            ),
            fmt=fmt,
        )

    X, Z = np.meshgrid(ds.x.values, ds.z.values)
    with open(f"./data/geometry/obj-y.csv", "w") as f:
        np.savetxt(
            f,
            np.column_stack(
                (
                    Z.flatten(),
                    X.flatten(),
                    ds.xi_y.values.T.reshape(ds.x.size * ds.z.size, ds.obj.size),
                    ds.xf_y.values.T.reshape(ds.x.size * ds.z.size, ds.obj.size),
                    ds.nobj_y.values.T.flatten(),
                    ds.nxipif_y.values.reshape(ds.x.size * ds.z.size, ds.obj.size),
                    ds.nxfpif_y.values.reshape(ds.x.size * ds.z.size, ds.obj.size),
                )
            ),
            fmt=fmt,
        )

    X, Y = np.meshgrid(ds.x.values, ds.y.values)
    with open(f"./data/geometry/obj-z.csv", "w") as f:
        np.savetxt(
            f,
            np.column_stack(
                (
                    Y.flatten(),
                    X.flatten(),
                    ds.xi_z.values.transpose(1, 0, 2).reshape(
                        ds.x.size * ds.y.size, ds.obj.size
                    ),
                    ds.xf_z.values.transpose(1, 0, 2).reshape(
                        ds.x.size * ds.y.size, ds.obj.size
                    ),
                    ds.nobj_z.values.T.flatten(),
                    ds.nxipif_z.values.transpose(1, 0, 2).reshape(
                        ds.x.size * ds.y.size, ds.obj.size
                    ),
                    ds.nxfpif_z.values.transpose(1, 0, 2).reshape(
                        ds.x.size * ds.y.size, ds.obj.size
                    ),
                )
            ),
            fmt=fmt,
        )

    ds.to_netcdf("./data/geometry/obj.nc")

    print("Done!")

    return ds
