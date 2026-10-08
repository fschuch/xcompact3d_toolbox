"""
Lazy, Dask-backed access to the raw binary files produced by XCompact3d,
built on `xarray-binfile`_.

This module describes the files of a simulation as an xarray-binfile
convention, assembled from a :obj:`xcompact3d_toolbox.parameters.Parameters`
instance. It coexists with :obj:`xcompact3d_toolbox.io.Dataset`, the
on-demand loader, and both read and write the same files.

.. _xarray-binfile: https://docs.fschuch.com/xarray-binfile/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray_binfile  # noqa: F401  (registers the .binary_engine accessors)
from xarray_binfile.conventions import Layout

from xcompact3d_toolbox.param import param

if TYPE_CHECKING:
    from xcompact3d_toolbox.parameters import Parameters

COORD_ATTRS: dict[str, dict[str, str]] = {
    "x": {"name": "Streamwise coordinate", "long_name": r"$x_1$"},
    "y": {"name": "Vertical coordinate", "long_name": r"$x_2$"},
    "z": {"name": "Spanwise coordinate", "long_name": r"$x_3$"},
    "t": {"name": "Time", "long_name": r"$t$"},
    "i": {"name": "Velocity component", "long_name": r"$i$"},
    "n": {"name": "Scalar fraction", "long_name": r"$\ell$"},
}
"""Attributes attached to the coordinates of the lazy dataset, matching :obj:`xcompact3d_toolbox.sandbox.init_dataset`."""


@dataclass(frozen=True)
class Xcompact3dConvention:
    """Describes the binary files of a simulation for `xarray-binfile`_.

    Build it with :obj:`from_parameters`.

    Parameters
    ----------
    layout : :obj:`xarray_binfile.conventions.Layout`
        Dimension order, coordinates, dtype and memory order of one file on disk.

    .. _xarray-binfile: https://docs.fschuch.com/xarray-binfile/
    """

    layout: Layout

    @classmethod
    def from_parameters(
        cls,
        prm: Parameters,
        *,
        dtype: Any = None,
        drop_coords: str = "",
    ) -> Xcompact3dConvention:
        """Build the convention from a :obj:`xcompact3d_toolbox.parameters.Parameters` instance.

        Parameters
        ----------
        prm : :obj:`xcompact3d_toolbox.parameters.Parameters`
            The simulation parameters; mesh, time step and output frequency are read from it.
        dtype : dtype_like, optional
            On-disk data type of every file. Defaults to ``xcompact3d_toolbox.param["mytype"]``
            at call time.
        drop_coords : str, optional
            Coordinate to drop when the files hold 2D planes: ``"x"``, ``"y"`` or ``"z"``
            (default is ``""``, for 3D fields).

        Returns
        -------
        :obj:`Xcompact3dConvention`
            The convention.
        """
        layout = Layout(
            prm.mesh.drop(*drop_coords),
            dtype=np.dtype(param["mytype"] if dtype is None else dtype),
            order="F",
            coord_attrs=COORD_ATTRS,
        )
        return cls(layout=layout)
